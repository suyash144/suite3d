import os
import numpy as np
from copy import copy
import threading
import gc

from ..job import Job
from ..io import s3dio
from ..utils import default_log


class Batches:
    filename: str = "offsets"
    dirname: str = "batch"

    def __init__(
        self,
        job: Job,
        makedirs: bool = True,
    ):
        self.job = job
        self.tifs = job.tifs
        self.batch_size = job.params["tif_batch_size"]
        self.makedirs = makedirs
        self._job_iter_dir = job.dirs["iters"]
        self._job_reg_data_dir = job.dirs["registered_fused_data"]
        self._batches = None
        self._batch_files = None

    @property
    def batches(self):
        if self._batches is None:
            self._compute_init_batches()
        return self._batches

    @property
    def n_batches(self):
        return len(self.batches)

    @property
    def batch_dirs(self):
        if self._batch_files is None:
            self._compute_init_batch_files()
        return self._batch_files["batch_dirs"]

    @property
    def reg_data_paths(self):
        if self._batch_files is None:
            self._compute_init_batch_files()
        return self._batch_files["reg_data_paths"]

    def _compute_init_batches(self):
        n_tifs_to_analyze = self.job.params.get("n_tifs_to_analyze")
        if n_tifs_to_analyze is not None and n_tifs_to_analyze > 0:
            self.tifs = self.tifs[:n_tifs_to_analyze]
        n_tifs = len(self.tifs)
        n_batches = int(np.ceil(n_tifs / self.batch_size))
        self._batches = []
        for i in range(n_batches):
            self.batches.append(
                self.tifs[i * self.batch_size : (i + 1) * self.batch_size]
            )

    def _compute_init_batch_files(self):
        reg_data_paths = []
        batch_dirs = []
        for batch_idx in range(self.n_batches):
            if self._job_reg_data_dir is not None:
                reg_data_filename = self.filename + "%04d.npy" % batch_idx
                reg_data_path = os.path.join(self._job_reg_data_dir, reg_data_filename)
                reg_data_paths.append(reg_data_path)
            if self.makedirs:
                assert self._job_iter_dir is not None
                batch_dir = os.path.join(
                    self._job_iter_dir, self.dirname + "%04d" % batch_idx
                )
                os.makedirs(batch_dir, exist_ok=True)
                batch_dirs.append(batch_dir)

        self._batch_files = {"batch_dirs": batch_dirs, "reg_data_paths": reg_data_paths}


class ThreadedBatchLoader:
    """
    Central class for loading imaging data batches with threading.

    This class provides an iterable for loading batches of imaging data
    with intelligent threading, where
    """

    def __init__(
        self,
        jobio: s3dio,
        batches: list[list[str]],
        log_cb: callable = default_log,
    ):
        self.jobio = jobio
        self.batches = batches
        self.log_cb = log_cb
        self.n_batches = len(batches)
        self.io_thread = None
        self._loaded_data = None

    def _loader_thread(self, batch_idx):
        self._log(f"[Thread] Loading batch {batch_idx}")
        self._log(f"   [Thread] Before load {batch_idx}", log_mem=True)
        self._loaded_data = self.jobio.load_data(self.batches[batch_idx])
        self._log(f"[Thread] Finished loading batch {batch_idx}", log_mem=True)

    def _start_thread(self, batch_idx):
        self.io_thread = threading.Thread(target=self._loader_thread, args=(batch_idx,))
        self.io_thread.start()

    def _join_thread(self):
        if self.io_thread is not None:
            self.io_thread.join()
            self._log("Memory after thread joined", level=3, log_mem=True)
            self.io_thread = None

    def __iter__(self):
        self._start_thread(0)
        for batch_idx in range(self.n_batches):
            self._join_thread()
            data = copy(self._loaded_data)
            self._log("Memory after movie copied from thread", level=3, log_mem=True)
            gc.collect()
            self._log("Memory after thread memory cleared", level=3, log_mem=True)

            # Start next thread if there's more batches
            if batch_idx + 1 < self.n_batches:
                self._start_thread(batch_idx + 1)

            # Yield current batch
            yield data

    def _log(self, msg, level=5, log_mem=False):
        self.log_cb(msg, level, log_mem_usage=log_mem)
