import numpy as np
import gc
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings

class Suite3DProcessor:
    """
    Process suite3d neural recording session data and extract cell patches.
    """
    
    def __init__(self, data_dir: str, output_dir: str, box_size: Tuple[int, int, int] = (5, 20, 20)):
        """
        Initialize the processor.
        
        Args:
            data_dir: Directory containing session folders
            output_dir: Directory to save processed data
            box_size: (nbz, nby, nbx) - size of patches to extract around each cell
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.nbz, self.nby, self.nbx = box_size
        self.nchannel = 3  # mean_img, correlation map and footprint

        # Track processed sessions
        self.processed_sessions = []
        
    def load_session_data(self, session_path: Path) -> Dict:
        """
        Load all relevant data files from a session folder.
        
        Args:
            session_path: Path to session folder
            
        Returns:
            Dictionary containing loaded data
        """
        session_data = {}
        
        # Load fluorescence data
        f_path = session_path / "F.npy"
        if f_path.exists():
            session_data['F'] = np.load(f_path)
        
        # Load neuropil data
        fneu_path = session_path / "Fneu.npy"
        if fneu_path.exists():
            session_data['Fneu'] = np.load(fneu_path)
            
        # Load spikes data
        spks_path = session_path / "spks.npy"
        if spks_path.exists():
            session_data['spks'] = np.load(spks_path)
        
        # Load info dictionary
        info_path = session_path / "info.npy"
        if info_path.exists():
            session_data['info'] = np.load(info_path, allow_pickle=True).item()
        
        # Load cell classification
        iscell_path = session_path / "iscell.npy"
        if iscell_path.exists():
            session_data['iscell'] = np.load(iscell_path)
            
        # Load cell statistics
        stats_path = session_path / "stats.npy"
        if stats_path.exists():
            session_data['stats'] = np.load(stats_path, allow_pickle=True)
            
        return session_data
    
    def get_correlation_map(self, info: Dict) -> Optional[np.ndarray]:
        """
        Get correlation map from info dictionary (handle different naming).
        
        Args:
            info: Info dictionary from info.npy
            
        Returns:
            Correlation map array or None if not found
        """
        if 'corrmap' in info:
            return info['corrmap']
        elif 'vmap_raw' in info:
            return info['vmap_raw']
        elif 'vmap' in info:
            return info['vmap']
        else:
            warnings.warn("None of 'corrmap', 'vmap_raw' or 'vmap' found in info dictionary")
            return None
    
    def extract_cell_patch(self, volume: np.ndarray, center_coord: Tuple[int, int, int]) -> np.ndarray:
        """
        Extract a patch around a cell center from a 3D volume.
        
        Args:
            volume: 3D volume (nz, ny, nx)
            center_coord: (z, y, x) center coordinates
            
        Returns:
            Extracted patch of size (nbz, nby, nbx)
        """
        nz, ny, nx = volume.shape
        cz, cy, cx = center_coord
        
        # Calculate patch boundaries
        z_start = max(0, cz - self.nbz // 2)
        z_end = min(nz, z_start + self.nbz)
        z_start = max(0, z_end - self.nbz)  # Adjust if we're at edge
        
        y_start = max(0, cy - self.nby // 2)
        y_end = min(ny, y_start + self.nby)
        y_start = max(0, y_end - self.nby)
        
        x_start = max(0, cx - self.nbx // 2)
        x_end = min(nx, x_start + self.nbx)
        x_start = max(0, x_end - self.nbx)
        
        # Extract patch
        patch = volume[z_start:z_end, y_start:y_end, x_start:x_end]
        
        # Handle edge cases - pad if necessary
        if patch.shape != (self.nbz, self.nby, self.nbx):
            padded_patch = np.zeros((self.nbz, self.nby, self.nbx), dtype=patch.dtype)
            
            # Calculate padding
            pad_z = (self.nbz - patch.shape[0]) // 2
            pad_y = (self.nby - patch.shape[1]) // 2
            pad_x = (self.nbx - patch.shape[2]) // 2
            
            padded_patch[
                pad_z:pad_z + patch.shape[0],
                pad_y:pad_y + patch.shape[1],
                pad_x:pad_x + patch.shape[2]
            ] = patch
            
            return padded_patch
        
        return patch
    
    def create_cell_footprint(self, coords: List[np.ndarray], lam: np.ndarray, 
                             med_coord: Tuple[int, int, int]) -> np.ndarray:
        """
        Create cell footprint matrix from coordinates and weights relative to med.
        
        Args:
            coords: List of 3 arrays [z_coords, y_coords, x_coords] of pixel coordinates
            lam: Array of weights for each pixel
            med_coord: (z, y, x) center coordinates
            
        Returns:
            Footprint matrix of size (nbz, nby, nbx) with weights at relative positions
        """
        # Initialize empty footprint
        footprint = np.zeros((self.nbz, self.nby, self.nbx), dtype=np.float32)
        
        # Check if coords and lam are valid
        if len(coords) != 3:
            warnings.warn("coords should contain exactly 3 arrays (z, y, x)")
            return footprint
        
        z_coords, y_coords, x_coords = coords
        
        # Check if all arrays have same length
        if not (len(z_coords) == len(y_coords) == len(x_coords) == len(lam)):
            warnings.warn("coords arrays and lam must have same length")
            return footprint
        
        if len(lam) == 0:
            warnings.warn("No weights found, returning empty footprint")
            return footprint
        
        # Convert to relative coordinates (relative to med)
        med_z, med_y, med_x = med_coord
        
        # Calculate patch center
        center_z = self.nbz // 2
        center_y = self.nby // 2
        center_x = self.nbx // 2
        
        # Convert absolute coords to relative coords within patch
        rel_z = z_coords - med_z + center_z
        rel_y = y_coords - med_y + center_y
        rel_x = x_coords - med_x + center_x
        
        # Filter coordinates that fall within patch boundaries
        valid_mask = (
            (rel_z >= 0) & (rel_z < self.nbz) &
            (rel_y >= 0) & (rel_y < self.nby) &
            (rel_x >= 0) & (rel_x < self.nbx)
        )
        
        if np.sum(valid_mask) == 0:
            warnings.warn("No cell pixels fall within the patch boundaries")
            return footprint
        
        # Get valid coordinates and weights
        valid_z = rel_z[valid_mask].astype(int)
        valid_y = rel_y[valid_mask].astype(int)
        valid_x = rel_x[valid_mask].astype(int)
        valid_weights = lam[valid_mask]
        
        # Place weights at corresponding positions
        footprint[valid_z, valid_y, valid_x] = valid_weights
        
        return footprint

    def process_and_save_session(self, session_path: Path) -> Dict:
        """
        Process a single session and immediately save to disk.
        
        Args:
            session_path: Path to session folder
            
        Returns:
            Session info dictionary (patches are saved to disk, not returned)
        """
        print(f"Processing session: {session_path.name}")
        
        # Load session data
        session_data = self.load_session_data(session_path)
        
        # Check required data exists
        required_keys = ['info', 'stats', 'iscell']
        for key in required_keys:
            if key not in session_data:
                raise ValueError(f"Required file {key}.npy not found in {session_path}")
        
        info = session_data['info']
        stats = session_data['stats']
        iscell = session_data['iscell']
        
        # Get mean image and correlation map
        if 'mean_img' not in info:
            raise ValueError("mean_img not found in info dictionary")
        
        mean_img = info['mean_img']
        corrmap = self.get_correlation_map(info)
        
        if corrmap is None:
            # If no correlation map, use max_img as second channel or duplicate mean_img
            if 'max_img' in info:
                corrmap = info['max_img']
                print("Using max_img as second channel (corrmap/vmap not found)")
            else:
                corrmap = mean_img.copy()
                print("Using mean_img as second channel (corrmap/vmap not found)")
        
        # Filter for actual cells
        cell_indices = np.where(iscell[:, 0] == 1)[0]
        n_cells = len(cell_indices)
        
        print(f"Found {n_cells} cells out of {len(stats)} ROIs")
        
        if n_cells == 0:
            print("Warning: No cells found in this session, skipping")
            return None
        
        # Initialize output array
        cell_patches = np.zeros((n_cells, self.nchannel, self.nbz, self.nby, self.nbx), 
                               dtype=np.float32)
        
        # Extract patches for each cell
        for i, cell_idx in enumerate(cell_indices):
            cell_stat = stats[cell_idx]
            
            # Get cell center coordinates (med field)
            if 'med' not in cell_stat:
                warnings.warn(f"Cell {cell_idx} missing 'med' field, skipping")
                continue
                
            med_coord = cell_stat['med']  # Should be (z, y, x)
            
            # Extract patches from channels
            mean_patch = self.extract_cell_patch(mean_img, med_coord)
            corr_patch = self.extract_cell_patch(corrmap, med_coord)
            if 'coords' in cell_stat and 'lam' in cell_stat:
                coords = cell_stat['coords']
                lam = cell_stat['lam']
                footprint_patch = self.create_cell_footprint(coords, lam, med_coord)
            else:
                warnings.warn(f"Cell {cell_idx} missing 'coords' or 'lam' field, using zero footprint")
                footprint_patch = np.zeros((self.nbz, self.nby, self.nbx), dtype=np.float32)
            
            cell_patches[i, 0] = mean_patch
            cell_patches[i, 1] = corr_patch
            cell_patches[i, 2] = footprint_patch
        
        # Prepare session info
        session_info = {
            'session_name': session_path.name,
            'n_cells': n_cells,
            'n_rois_total': len(stats),
            'image_shape': mean_img.shape,
            'patch_shape': (self.nbz, self.nby, self.nbx),
            'cell_indices': cell_indices.tolist(),
            'has_corrmap': 'corrmap' in info,
            'has_vmap': 'vmap_raw' in info
        }
        
        # Save immediately to free memory
        session_name = session_path.name
        patches_file = self.output_dir / f"{session_name}_patches.npy"
        info_file = self.output_dir / f"{session_name}_info.npy"
        
        print(f"Saving patches to {patches_file}")
        np.save(patches_file, cell_patches)
        np.save(info_file, session_info)
        
        # Force garbage collection to free memory
        del cell_patches, mean_img, corrmap, session_data
        gc.collect()
        
        return session_info
    
    def process_all_sessions(self) -> List[Dict]:
        """
        Process all sessions in the data directory, saving each one immediately.
        
        Returns:
            List of session_info dictionaries (patches are saved to disk)
        """
        if not self.data_dir.exists():
            raise ValueError(f"Data directory {self.data_dir} does not exist")
        
        # Find all session folders
        session_paths = []
        for path in self.data_dir.iterdir():
            if path.is_dir() and (path / "info.npy").exists():
                session_paths.append(path)
        
        if not session_paths:
            raise ValueError(f"No valid session folders found in {self.data_dir}")
        
        print(f"Found {len(session_paths)} sessions to process")
        print(f"Output directory: {self.output_dir}")
        
        all_session_info = []
        
        for i, session_path in enumerate(sorted(session_paths), 1):
            try:
                print(f"\n[{i}/{len(session_paths)}] ", end="")
                session_info = self.process_and_save_session(session_path)
                
                if session_info is not None:
                    all_session_info.append(session_info)
                    self.processed_sessions.append(session_path.name)
                    print(f"✓ Session {session_path.name} completed")
                else:
                    print(f"⚠ Session {session_path.name} skipped (no cells)")
                    
            except Exception as e:
                print(f"✗ Error processing {session_path.name}: {e}")
                continue
        
        return all_session_info
    
    def create_combined_dataset(self, session_info_list: List[Dict]) -> None:
        """
        Create a combined dataset from all processed sessions by loading saved files.
        This is memory-efficient as it loads one session at a time.
        
        Args:
            session_info_list: List of session info dictionaries
        """
        if not session_info_list:
            print("No sessions to combine")
            return
        
        print(f"\nCreating combined dataset from {len(session_info_list)} sessions...")
        
        # Calculate total number of cells
        total_cells = sum(info['n_cells'] for info in session_info_list)
        print(f"Total cells across all sessions: {total_cells}")
        
        if total_cells == 0:
            print("No cells to combine")
            return
        
        # Initialize combined array
        print("Initializing combined array...")
        combined_patches = np.zeros((total_cells, self.nchannel, self.nbz, self.nby, self.nbx), 
                                   dtype=np.float32)
        
        # Load and concatenate each session
        current_idx = 0
        for session_info in session_info_list:
            session_name = session_info['session_name']
            n_cells = session_info['n_cells']
            
            if n_cells == 0:
                continue
                
            print(f"Loading {session_name}: {n_cells} cells")
            
            # Load session patches
            patches_file = self.output_dir / f"{session_name}_patches.npy"
            if patches_file.exists():
                session_patches = np.load(patches_file)
                
                # Copy to combined array
                combined_patches[current_idx:current_idx + n_cells] = session_patches
                current_idx += n_cells
                
                # Free memory immediately
                del session_patches
                gc.collect()
            else:
                print(f"Warning: {patches_file} not found")
        
        # Save combined dataset
        combined_patches_file = self.output_dir / "all_sessions_patches.npy"
        print(f"Saving combined patches to {combined_patches_file}")
        np.save(combined_patches_file, combined_patches)
        
        # Create combined info
        combined_info = {
            'total_cells': total_cells,
            'n_sessions': len(session_info_list),
            'session_names': [info['session_name'] for info in session_info_list],
            'patch_shape': (self.nbz, self.nby, self.nbx),
            'nchannel': self.nchannel,
            'session_cell_counts': [info['n_cells'] for info in session_info_list]
        }
        
        combined_info_file = self.output_dir / "all_sessions_info.npy"
        np.save(combined_info_file, combined_info)
        
        # Clean up
        del combined_patches
        gc.collect()
        
        print(f"✓ Combined dataset saved with {total_cells} cells")
    
    def run_full_pipeline(self, create_combined: bool = True) -> None:
        """
        Run the complete processing pipeline.
        
        Args:
            create_combined: Whether to create combined dataset after individual processing
        """
        print("=" * 60)
        print("Suite3D Processing Pipeline")
        print("=" * 60)
        print(f"Data directory: {self.data_dir}")
        print(f"Output directory: {self.output_dir}")
        print(f"Box size (z,y,x): {(self.nbz, self.nby, self.nbx)}")
        
        # Process all sessions
        session_info_list = self.process_all_sessions()
                
        # Create combined dataset if requested
        if create_combined and session_info_list:
            self.create_combined_dataset(session_info_list)
        
        # Final summary
        total_cells = sum(info['n_cells'] for info in session_info_list)
        print("\n" + "=" * 60)
        print("PROCESSING COMPLETE")
        print("=" * 60)
        print(f"Sessions processed: {len(session_info_list)}")
        print(f"Total cells extracted: {total_cells}")
        print(f"Output directory: {self.output_dir}")
        print(f"Files saved:")
        for session_info in session_info_list:
            session_name = session_info['session_name']
            print(f"  - {session_name}_patches.npy ({session_info['n_cells']} cells)")
            print(f"  - {session_name}_info.npy")
        if create_combined and total_cells > 0:
            print(f"  - all_sessions_patches.npy ({total_cells} cells)")
            print(f"  - all_sessions_info.npy")
        print("=" * 60)


def main():

    data_directory = r"\\znas.cortexlab.net\Lab\Share\Ali\for-suyash"
    output_directory = r"\\znas.cortexlab.net\Lab\Share\Ali\for-suyash\output"

    processor = Suite3DProcessor(
        data_dir=data_directory,
        output_dir=output_directory,
        box_size=(5, 20, 20)  # nbz=5, nby=20, nbx=20
    )
    
    # Process all sessions
    try:
        processor.run_full_pipeline(create_combined=True)
        
    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()


    all_sessions_patches = np.load(r"\\znas.cortexlab.net\Lab\Share\Ali\for-suyash\output\all_sessions_patches.npy")

    print(all_sessions_patches.shape)

    all_sessions_info = np.load(r"\\znas.cortexlab.net\Lab\Share\Ali\for-suyash\output\all_sessions_info.npy", allow_pickle=True).item()
    print(all_sessions_info['total_cells'])
    # print(all_sessions_info['n_sessions'])
    