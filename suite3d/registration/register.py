import os
import numpy as n
import time
import traceback
import gc
import threading

from ..s2p_registration import nonrigid_transform_data, register_frames
from .. import svd_utils as svu

from .. import utils
from .. import register_gpu as reg_gpu
from .. import reg_3d as reg_3d
from .. import reference_image as ref
from .. import quality_metrics as qm
from ..utils import default_log
from ..io import s3dio
from ..job import Job
from .loading import ThreadedBatchLoader, Batches

try:
    import cupy as cp
except ImportError:
    import numpy as cp


class ParamsConfig:
    def __init__(
        self,
        job: Job,
        summary: dict,
        log_cb: callable = default_log,
    ):
        """Pydantic model for registering summary variables with defaults"""
        self.log_cb = log_cb

        self.refs_and_masks = summary["refs_and_masks"]
        self.ref_img_3d = summary["ref_img_3d"]
        self.min_pix_vals = summary["min_pix_vals"]
        self.crosstalk_coeff = summary["crosstalk_coeff"]
        self.xpad = summary["xpad"]
        self.ypad = summary["ypad"]
        self.plane_shifts = summary["plane_shifts"]
        self.new_xs = summary["new_xs"]
        self.old_xs = summary["og_xs"]

        if job.params["fuse_shift_override"] is not None:
            self.fuse_shift = job.params["fuse_shift_override"]
            log_cb("Overriding fuse shift value to %d" % self.fuse_shift)
        else:
            self.fuse_shift = summary["fuse_shift"]

        # new parameters
        reference_params = summary["reference_params"]
        self.rmins = reference_params.get("plane_mins", None)
        self.rmaxs = reference_params.get("plane_maxs", None)
        self.snr_thresh = 1.2  # TODO add values to a default params dictionary
        self.NRsm = reference_params["NRsm"]
        self.yblocks, self.xblocks = (
            reference_params["yblock"],
            reference_params["xblock"],
        )
        self.nblocks = reference_params["nblocks"]

        # for 3D GPU registration
        self.pc_size = job.params.get("pc_size", (2, 20, 20))
        self.frate_hz = job.params.get("fs", 4)

        self.tif_batch_size = job.params["tif_batch_size"]
        self.planes = job.params["planes"]
        self.notch_filt = job.params["notch_filt"]
        self.enforce_positivity = job.params.get("enforce_positivity", False)
        self.mov_dtype = job.params["dtype"]
        self.split_tif_size = job.params.get("split_tif_size", None)
        self.n_ch_tif = job.params.get("n_ch_tif", 30)
        self.max_rigid_shift = job.params.get("max_rigid_shift_pix", 75)
        self.gpu_reg_batchsize = job.params.get("gpu_reg_batchsize", 10)
        self.max_shift_nr = job.params.get("max_shift_nr", 3)
        self.nr_npad = job.params.get("nr_npad", 3)
        self.nr_subpixel = job.params.get("nr_subpixel", 10)
        self.nr_smooth_iters = job.params.get("nr_smooth_iters", 2)
        self.fuse_strips = job.params.get("fuse_strips", True)
        self.fix_fastZ = job.params.get("fix_fastZ", False)
        self.reg_norm_frames = job.params.get("reg_norm_frames", True)
        self.cavity_size = job.params.get("cavity_size", 15)
        self.nonrigid = job.params.get("nonrigid", True)

        _save_dtype = job.params.get("save_dtype", "float32")
        if _save_dtype == "float32":
            self.save_dtype = n.float32
        elif _save_dtype == "float16":
            self.save_dtype = n.float16
        else:
            raise ValueError(f"Invalid save_dtype: {_save_dtype}")


def register(
    job,
    params,
    summary,
    log_cb=default_log,
    max_gpu_batches=None,
    use_gpu: bool = False,
    use_3d: bool = False,
):
    """Skeleton for registering a dataset"""
    jobio = s3dio(job)

    # Load summary variables (or maybe just use summary object and config?)
    cfg = ParamsConfig(job, params, summary, log_cb)

    # Prepare batches
    batches = Batches(job)


def register_dataset_gpu(
    job,
    summary,
    log_cb=default_log,
    max_gpu_batches=None,
):
    jobio = s3dio(job)
    cfg = ParamsConfig(job, summary, log_cb)
    batches = Batches(job)
    threaded_loader = ThreadedBatchLoader(jobio, batches.batches, log_cb=log_cb)
    reg_data_paths = []

    mask_mul, mask_offset, ref_2ds = n.stack(
        [r[:3] for r in cfg.refs_and_masks], axis=1
    )
    mask_mul_nr, mask_offset_nr, ref_nr = n.stack(
        [r[3:] for r in cfg.refs_and_masks], axis=1
    )
    max_shift_nr = 5

    # catch if rmins/rmaxs where not calculate in init_pass
    if rmins is None and rmaxs is None:
        log_cb("Not clipping frames for registration")
        rmins = n.array([None for i in range(cfg.n_ch_tif)])
        rmaxs = n.array([None for i in range(cfg.n_ch_tif)])
    else:
        if not cfg.reg_norm_frames:
            log_cb("Not clipping frames for registration")
            rmins = n.array([None for i in range(len(rmins))])
            rmaxs = n.array([None for i in range(len(rmaxs))])

    if max_rigid_shift < n.ceil(n.max(n.abs(summary["plane_shifts"]))) + 5:
        max_rigid_shift = n.ceil(n.max(n.abs(summary["plane_shifts"]))) + 5

    log_cb(
        "Will analyze %d tifs in %d batches"
        % (len(n.concatenate(batches.batches)), len(batches.batches)),
        0,
    )
    if cfg.enforce_positivity:
        log_cb("Enforcing positivity", 1)

    file_idx = 0
    for batch_idx, mov_cpu in enumerate(threaded_loader):
        nt = mov_cpu.shape[1]
        ymaxs_rr = []
        xmaxs_rr = []
        mov_shifted = []
        ymaxs_nr = []
        xmaxs_nr = []

        mov_shifted = None
        log_cb("Loaded batch of size %s" % ((str(mov_cpu.shape))), 2)
        for gpu_batch_idx in range(int(n.ceil(nt / cfg.gpu_reg_batchsize))):
            if max_gpu_batches is not None:
                if gpu_batch_idx >= max_gpu_batches:
                    break
            idx0 = cfg.gpu_reg_batchsize * gpu_batch_idx
            idx1 = min(idx0 + cfg.gpu_reg_batchsize, nt)
            log_cb(
                "Sending frames %d-%d to GPU for rigid registration" % (idx0, idx1), 3
            )
            tic_rigid = time.time()

            mov_shifted_gpu, ymaxs_rr_gpu, xmaxs_rr_gpu, __ = reg_gpu.rigid_2d_reg_gpu(
                mov_cpu[:, idx0:idx1],
                mask_mul,
                mask_offset,
                ref_2ds,
                max_reg_xy=max_rigid_shift,
                min_pix_vals=cfg.min_pix_vals,
                rmins=rmins,
                rmaxs=rmaxs,
                crosstalk_coeff=cfg.crosstalk_coeff,
                shift=True,
                xpad=cfg.xpad,
                ypad=cfg.ypad,
                fuse_shift=cfg.fuse_shift,
                new_xs=cfg.new_xs,
                old_xs=cfg.old_xs,
                fuse_and_pad=True,
                cavity_size=cfg.cavity_size,
                log_cb=log_cb,
            )

            mov_shifted_cpu = mov_shifted_gpu.get()
            log_cb(
                "Completed rigid registration in %.2f sec" % (time.time() - tic_rigid),
                3,
            )
            tic_nonrigid = time.time()
            if cfg.nonrigid:
                ymaxs_nr_gpu, xmaxs_nr_gpu, snrs = reg_gpu.nonrigid_2d_reg_gpu(
                    mov_shifted_gpu,
                    mask_mul_nr[:, :, 0],
                    mask_offset_nr[:, :, 0],
                    ref_nr[:, :, 0],
                    yblocks,
                    xblocks,
                    snr_thresh,
                    NRsm,
                    rmins,
                    rmaxs,
                    max_shift=max_shift_nr,
                    npad=nr_npad,
                    n_smooth_iters=nr_smooth_iters,
                    subpixel=nr_subpixel,
                    log_cb=log_cb,
                )
                log_cb(
                    "Computed non-rigid shifts in %.2f sec" % (time.time() - tic_rigid),
                    3,
                )

                tic_get = time.time()
                ymaxs_nr_cpu = ymaxs_nr_gpu.get()
                xmaxs_nr_cpu = xmaxs_nr_gpu.get()
            else:
                print("NO NONRIGID\n\n\n")
                tic_get = time.time()
                xmaxs_nr_cpu = n.zeros_like(ymaxs_rr_gpu)
                ymaxs_nr_cpu = n.zeros_like(ymaxs_rr_gpu)

            ymaxs_rr_cpu = ymaxs_rr_gpu.get()
            xmaxs_rr_cpu = xmaxs_rr_gpu.get()
            # print("######\n\nAFter RIGID: 0.5p: %.3f 99.5p: %.3f, Mean: %.3f, Min: %.3f, Max:%.3f" %
            #    (n.percentile(mov_shifted_cpu[:,10],0.5), n.percentile(mov_shifted_cpu[:,10],99.5),
            # mov_shifted_cpu[:,10].mean(), mov_shifted_cpu[:,10].min(),
            # mov_shifted_cpu[:,10].max()))
            # print("SHAPE")
            # print(mov_shifted_cpu.shape)
            del mov_shifted_gpu
            log_cb(
                "Transferred shifted mov of shape %s to CPU in %.2f sec"
                % (str(mov_shifted_cpu.shape), time.time() - tic_get),
                3,
            )

            if mov_shifted is None:
                mov_shifted = n.zeros(
                    (
                        mov_shifted_cpu.shape[1],
                        nt,
                        mov_shifted_cpu.shape[2],
                        mov_shifted_cpu.shape[3],
                    ),
                    n.float32,
                )
                log_cb(
                    "Allocated array of shape %s to store CPU movie"
                    % str(mov_shifted.shape),
                    3,
                )
                log_cb("After array alloc:", level=3, log_mem_usage=True)

            shift_tic = time.time()
            nz = mov_shifted_cpu.shape[1]
            for zidx in range(nz):
                if nonrigid:
                    # print("SHIFITNG: %d" % zidx)
                    # TODO migrate to suite3D?

                    mov_shifted[zidx, idx0:idx1] = nonrigid_transform_data(
                        mov_shifted_cpu[:, zidx],
                        nblocks,
                        xblock=xblocks,
                        yblock=yblocks,
                        ymax1=ymaxs_nr_cpu[:, zidx],
                        xmax1=xmaxs_nr_cpu[:, zidx],
                    )
                else:
                    mov_shifted[zidx, idx0:idx1] = mov_shifted_cpu[:, zidx]

            # print("######\n\nAFter NONRIGID: 0.5p: %.3f 99.5p: %.3f, Mean: %.3f, Min: %.3f, Max:%.3f" %
            #        (n.percentile(mov_shifted[10,idx0:idx1],0.5), n.percentile(mov_shifted[10,idx0:idx1],99.5),
            #         mov_shifted[10,idx0:idx1].mean(), mov_shifted[10,idx0:idx1].min(),
            #         mov_shifted[10,idx0:idx1].max()))
            log_cb(
                "Non rigid transformed (on CPU) in %.2f sec"
                % (time.time() - shift_tic),
                3,
            )

            # mov_shifted.append(mov_shifted_cpu)
            ymaxs_rr.append(ymaxs_rr_cpu.T)
            xmaxs_rr.append(xmaxs_rr_cpu.T)
            ymaxs_nr.append(ymaxs_nr_cpu)
            xmaxs_nr.append(xmaxs_nr_cpu)

            mempool = cp.get_default_memory_pool()
            mempool.free_all_blocks()

            log_cb("After GPU Batch:", level=3, log_mem_usage=True)

        concat_t = time.time()
        log_cb("Concatenating movie", 2)
        # mov_shifted = mov_shifted_cpu # n.concatenate(mov_shifted,axis=0)
        # print("CONCAT")
        # print(mov_shifted.shape)
        log_cb("Concat in %.2f sec" % (time.time() - concat_t), 3)
        all_offsets = {}
        all_offsets["xmaxs_rr"] = n.concatenate(xmaxs_rr, axis=0)
        all_offsets["ymaxs_rr"] = n.concatenate(ymaxs_rr, axis=0)
        all_offsets["xmaxs_nr"] = n.concatenate(xmaxs_nr, axis=0)
        all_offsets["ymaxs_nr"] = n.concatenate(ymaxs_nr, axis=0)

        log_cb("After all GPU Batches:", level=3, log_mem_usage=True)

        if split_tif_size is None:
            split_tif_size = mov_shifted.shape[0]
        for i in range(0, mov_shifted.shape[1], split_tif_size):
            reg_data_path = os.path.join(
                job.dirs["registered_fused_data"], "fused_reg_data%04d.npy" % file_idx
            )
            reg_data_paths.append(reg_data_path)
            end_idx = min(mov_shifted.shape[1], i + split_tif_size)
            mov_save = mov_shifted[:, i:end_idx]
            if max_gpu_batches is not None:
                if i > max_gpu_batches * gpu_reg_batchsize:
                    break
            # mov_save = n.swapaxes(mov_save, 0, 1)
            save_t = time.time()
            log_cb(
                "Saving fused, registered file of shape %s to %s"
                % (str(mov_save.shape), reg_data_path),
                2,
            )
            n.save(reg_data_path, mov_save.astype(save_dtype))
            log_cb("Saved in %.2f sec" % (time.time() - save_t), 3)
            file_idx += 1
        n.save(batches.reg_data_paths[batch_idx], all_offsets)

        log_cb("After full batch saving:", level=3, log_mem_usage=True)


def register_dataset_gpu_3d(
    job, tifs, params, dirs, summary, log_cb=default_log, max_gpu_batches=None
):
    jobio = s3dio(job)

    refs_and_masks = summary["refs_and_masks"]
    ref_img_3d = summary["ref_img_3d"]
    min_pix_vals = summary["min_pix_vals"]
    crosstalk_coeff = summary["crosstalk_coeff"]
    xpad = summary["xpad"]
    ypad = summary["ypad"]
    plane_shifts = summary["plane_shifts"]
    fuse_shift = summary["fuse_shift"]
    new_xs = summary["new_xs"]
    old_xs = summary["og_xs"]

    # new parameters
    reference_params = summary["reference_params"]
    rmins = reference_params.get("plane_mins", None)
    rmaxs = reference_params.get("plane_maxs", None)
    snr_thresh = 1.2  # TODO add values to a default params dictionary
    NRsm = reference_params["NRsm"]
    yblocks, xblocks = reference_params["yblock"], reference_params["xblock"]
    nblocks = reference_params["nblocks"]
    pc_size = params.get("pc_size", (2, 20, 20))
    frate_hz = params.get("fs", 4)

    # choose the top 2% of pix in each plane to run
    # quality metrics on
    top_pix = qm.choose_top_pix(ref_img_3d)

    # NOTE TODO the current mask_mul etc is uncropped, so currently calculated here but should be changed in reference_image.py
    # when updating to full 3D

    # mask_mul, mask_offset, ref_2ds = n.stack([r[:3] for r in refs_and_masks],axis=1)

    # Current hack to get cropped ref + maks
    sigma = reference_params["sigma"]
    ref_img = ref_img_3d.copy()
    if ypad > 0:
        ref_img = ref_img[:, int(ypad) : int(-ypad)]
    if xpad > 0:
        ref_img = ref_img[:, :, int(xpad) : int(-xpad)]

    # ref_img = ref_img_3d[:, int(ypad):int(-ypad), int(xpad): int(-xpad)]
    mask_mul, mask_offset = ref.compute_masks3D(ref_img, sigma)
    ref_2ds = reg_3d.mask_filter_fft_ref(ref_img, mask_mul, mask_offset, smooth=0.5)

    if params["fuse_shift_override"] is not None:
        fuse_shift = params["fuse_shift_override"]
        log_cb("Overriding fuse shift value to %d" % fuse_shift)

    job_iter_dir = dirs["iters"]
    job_reg_data_dir = dirs["registered_fused_data"]

    n_tifs_to_analyze = params.get("total_tifs_to_analyze", len(tifs))
    tif_batch_size = params["tif_batch_size"]
    planes = params["planes"]
    notch_filt = params["notch_filt"]
    enforce_positivity = params.get("enforce_positivity", False)
    mov_dtype = params["dtype"]
    split_tif_size = params.get("split_tif_size", None)
    n_ch_tif = params.get("n_ch_tif", 30)
    max_rigid_shift = params.get("max_rigid_shift_pix", 75)
    apply_z_shift = params.get("apply_z_shift", False)
    gpu_reg_batchsize = params.get("gpu_reg_batchsize", 10)
    max_shift_nr = params.get("max_shift_nr", 3)
    nr_npad = params.get("nr_npad", 3)
    nr_subpixel = params.get("nr_subpixel", 10)
    nr_smooth_iters = params.get("nr_smooth_iters", 2)
    fuse_strips = params.get("fuse_strips", True)
    fix_fastZ = params.get("fix_fastZ", False)
    reg_norm_frames = params.get("reg_norm_frames", True)
    cavity_size = params.get("cavity_size", 15)
    save_dtype_str = params.get("save_dtype", "float32")
    save_dtype = None
    if save_dtype_str == "float32":
        save_dtype = n.float32
    elif save_dtype_str == "float16":
        save_dtype = n.float16

    # catch if rmins/rmaxs where not calculate in init_pass
    if rmins is None and rmaxs is None:
        log_cb("Not clipping frames for registration")
        rmins = n.array([None for i in range(n_ch_tif)])
        rmaxs = n.array([None for i in range(n_ch_tif)])
    else:
        if not reg_norm_frames:
            log_cb("Not clipping frames for registration")
            rmins = n.array([None for i in range(len(rmins))])
            rmaxs = n.array([None for i in range(len(rmaxs))])

    if max_rigid_shift < n.ceil(n.max(n.abs(summary["plane_shifts"]))) + 5:
        max_rigid_shift = n.ceil(n.max(n.abs(summary["plane_shifts"]))) + 5

    convert_plane_ids_to_channel_ids = params.get(
        "convert_plane_ids_to_channel_ids", True
    )

    batches = init_batches(tifs, tif_batch_size, n_tifs_to_analyze)
    n_batches = len(batches)
    __, offset_paths = init_batch_files(
        job_iter_dir, job_reg_data_dir, n_batches, makedirs=False, filename="offsets"
    )
    reg_data_paths = []

    log_cb(
        "Will analyze %d tifs in %d batches"
        % (len(n.concatenate(batches)), len(batches)),
        0,
    )
    if enforce_positivity:
        log_cb("Enforcing positivity", 1)

    loaded_movs = [n.zeros(1)]

    def io_thread_loader(tifs, batch_idx):
        tic_thread = time.time()
        log_cb("[Thread] Loading batch %d \n" % batch_idx, 5)
        log_cb("   [Thread] Before load %d \n" % batch_idx, 5, log_mem_usage=True)
        loaded_mov = jobio.load_data(tifs)
        # loaded_mov = lbmio.load_and_stitch_tifs(tifs, planes, filt = notch_filt, concat=True,n_ch=n_ch_tif, fix_fastZ=fix_fastZ,
        #                                         convert_plane_ids_to_channel_ids=convert_plane_ids_to_channel_ids, log_cb=log_cb)
        loaded_movs[0] = loaded_mov
        log_cb(
            "[Thread] Thread for batch %d ready to join after %2.2f sec \n"
            % (batch_idx, time.time() - tic_thread),
            5,
        )
        log_cb("   [Thread] After load %d \n" % batch_idx, 5, log_mem_usage=True)
        # log_cb("loaded mov: ")
        # log_cb(str(loaded_mov.shape))

    log_cb("Launching IO thread")
    io_thread = threading.Thread(target=io_thread_loader, args=(batches[0], 0))
    io_thread.start()

    file_idx = 0
    for batch_idx in range(n_batches):
        log_cb("Memory at batch %d." % batch_idx, level=3, log_mem_usage=True)
        offset_path = offset_paths[batch_idx]
        log_cb("Loading Batch %d of %d" % (batch_idx, n_batches - 1), 0)
        io_thread.join()
        log_cb("Batch %d IO thread joined" % (batch_idx))
        log_cb("Memory after IO thread join", level=3, log_mem_usage=True)

        mov_cpu = loaded_movs[0].copy()
        log_cb("Memory after movie copied from thread", level=3, log_mem_usage=True)
        loaded_movs[0] = n.zeros(1)
        gc.collect()
        log_cb("Memory after thread memory cleared", level=3, log_mem_usage=True)

        if batch_idx + 1 < n_batches:
            log_cb("Launching IO thread for next batch")
            io_thread = threading.Thread(
                target=io_thread_loader, args=(batches[batch_idx + 1], batch_idx + 1)
            )
            io_thread.start()
            log_cb("After IO thread launch:", level=3, log_mem_usage=True)
        nt = mov_cpu.shape[1]
        # Change to new kept info
        mov_shifted = []

        mov_shifted = None
        log_cb("Loaded batch of size %s" % ((str(mov_cpu.shape))), 2)
        # New function has loop over batches as part of registration

        time_pre_reg = time.time()
        # log time it takes
        phase_corr_shifted, int_shift, pc_peak_loc, sub_pixel_shifts, mov_cpu = (
            reg_3d.rigid_3d_ref_gpu(
                mov_cpu,
                mask_mul,
                mask_offset,
                ref_2ds,
                pc_size,
                batch_size=gpu_reg_batchsize,  # TODO make xpad/ypad automatically integers
                rmins=rmins,
                rmaxs=rmaxs,
                crosstalk_coeff=crosstalk_coeff,
                shift_reg=False,
                xpad=int(xpad),
                ypad=int(ypad),
                fuse_shift=fuse_shift,
                new_xs=new_xs,
                old_xs=old_xs,
                plane_shifts=plane_shifts,
                process_mov=True,
                cavity_size=cavity_size,
            )
        )

        log_cb(f"Completed rigid reg on batch in :{time.time() - time_pre_reg}s")

        time_shift = time.time()
        # shift entire abtch on cpu at once
        # log this info
        mov_shifted = reg_3d.shift_mov_fast(mov_cpu, -int_shift)

        if apply_z_shift:
            # if there is at least one
            if n.max(int_shift[0]) > 1:
                mov_shifted = reg_3d.shift_mov_z(mov_shifted, int_shift)
        log_cb(f"Shifted the mov in: {time.time() - time_shift}s")

        # NOTE changed this so gets int_shifts + sub_pixel shifts etc
        all_offsets = {}
        all_offsets["phase_corr_shifted"] = phase_corr_shifted
        all_offsets["int_shift"] = int_shift
        all_offsets["pc_peak_loc"] = pc_peak_loc
        all_offsets["sub_pixel_shifts"] = sub_pixel_shifts

        log_cb("After all GPU Batches:", level=3, log_mem_usage=True)

        if split_tif_size is None:
            split_tif_size = mov_shifted.shape[0]
        for i in range(0, mov_shifted.shape[1], split_tif_size):
            reg_data_path = os.path.join(
                job_reg_data_dir, "fused_reg_data%04d.npy" % file_idx
            )
            reg_data_paths.append(reg_data_path)
            end_idx = min(mov_shifted.shape[1], i + split_tif_size)
            mov_save = mov_shifted[:, i:end_idx]
            if max_gpu_batches is not None:
                if i > max_gpu_batches * gpu_reg_batchsize:
                    break
            # mov_save = n.swapaxes(mov_save, 0, 1)
            save_t = time.time()
            log_cb(
                "Saving fused, registered file of shape %s to %s"
                % (str(mov_save.shape), reg_data_path),
                2,
            )
            n.save(reg_data_path, mov_save.astype(save_dtype))
            log_cb("Saved in %.2f sec" % (time.time() - save_t), 3)

            metrics_path = os.path.join(
                job_reg_data_dir, "reg_metrics_%04d.npy" % file_idx
            )
            mean_img_path = os.path.join(
                job_reg_data_dir, "mean_img_%04d.npy" % file_idx
            )
            log_cb("Computing quality metrics and saving", 2)

            mean_img, metrics = qm.compute_metrics_for_movie(
                mov_save, frate_hz, top_pix=top_pix
            )
            n.save(mean_img_path, mean_img)
            n.save(metrics_path, metrics)

            file_idx += 1
        n.save(offset_path, all_offsets)

        log_cb("After full batch saving:", level=3, log_mem_usage=True)
