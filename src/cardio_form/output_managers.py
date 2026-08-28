# in cardio_form/output_managers.py

import os

class OutputManager:
    """
    Manages output paths for the pipeline with a flat, prefixed naming scheme.
    
    This class takes a single output directory and a prefix, and generates
    full filenames by appending a descriptive suffix. It does not create any
    subdirectories.
    """
    def __init__(self, output_dir: str, output_prefix: str):
        """
        Initializes the manager.

        Args:
            output_dir (str): The directory where all files will be saved.
            output_prefix (str): The prefix for all generated filenames.
        """
        self.output_dir = output_dir
        self.prefix = output_prefix
        os.makedirs(self.output_dir, exist_ok=True)
        
        # This config maps a logical key to a descriptive suffix.
        # This is the single source of truth for all output filenames.
        self._config = {
            # --- Final Reconstruction Outputs ---
            'prediction': '_whole_heart_segmentation.nii.gz',

            # --- Intermediate Segmentation Outputs ---
            'seg_sax': '_2D_seg_sax.nii.gz',
            'seg_lax_2ch': '_2D_seg_lax_2ch.nii.gz',
            'seg_lax_4ch': '_2D_seg_lax_4ch.nii.gz',

            # --- Intermediate Reconstruction Outputs ---
            'sparse_volume': '_intermediate_sparse_volume.nii.gz',
            'sax_bp': '_intermediate_qc_sax_backprojected.nii.gz',
            'ch2_bp': '_intermediate_qc_ch2_backprojected.nii.gz',
            'ch4_bp': '_intermediate_qc_ch4_backprojected.nii.gz',

            # -- LA Reconstruction 3D specific outputs ---
            # Inputs remapped into the LA network's label space (QC artefacts).
            'la_input_2ch': '_intermediate_la_3d_input_2ch.nii.gz',
            'la_input_4ch': '_intermediate_la_3d_input_4ch.nii.gz',
            'la_prediction': '_la_3d_segmentation.nii.gz',
            'la_sparse_volume': '_la_3d_sparse_volume.nii.gz',
            'la_ch2_bp': '_intermediate_la_3d_qc_ch2_backprojected.nii.gz',
            'la_ch4_bp': '_intermediate_la_3d_qc_ch4_backprojected.nii.gz',

            # --- Left-heart completion output ---
            'left_complete': '_left_complete_segmentation.nii.gz',
        }

    def get_path(self, key: str) -> str:
        """
        Constructs the full, absolute path for a given output key.

        Args:
            key (str): The key for the desired output (e.g., 'prediction').

        Returns:
            The absolute path for the output file.
        """
        if key not in self._config:
            raise KeyError(f"Output key '{key}' not found in OutputManager configuration.")
        
        suffix = self._config[key]
        filename = self.prefix + suffix
        return os.path.join(self.output_dir, filename)

    def get_all_paths(self) -> dict:
        """Returns a dictionary of all configured paths for a given set of keys."""
        # Note: This method is less useful now, as each function will only request the keys it needs.
        # It's kept for potential future use.
        return {key: self.get_path(key) for key in self._config}