import os 
import nibabel as nib


class ReconstructOutputManager:
    """
    A simple and robust class to manage the output paths for the pipeline.
    """
    def __init__(self, base_output_dir: str, subject_id: str):
        """
        Initializes the manager with a base directory and a subject ID.

        Args:
            base_output_dir (str): The root directory for all outputs.
            subject_id (str): The unique identifier for the case being processed.
        """
        self.base_dir = os.path.join(base_output_dir, subject_id)
        
        # --- Configuration ---
        # This private dictionary defines our entire output structure.
        # It maps a simple key to a tuple: (subdirectory, filename_template)
        self._config = {
            'prediction': ('', '{subject_id}_whole_heart_segmentation.nii.gz'),
            'sparse_volume': ('intermediate', '{subject_id}_sparse_volume.nii.gz'),
            'sax_bp': ('quality_control', '{subject_id}_qc_sax_backprojected.nii.gz'),
            'ch2_bp': ('quality_control', '{subject_id}_qc_ch2_backprojected.nii.gz'),
            'ch4_bp': ('quality_control', '{subject_id}_qc_ch4_backprojected.nii.gz'),
        }

    def get_path(self, key: str) -> str:
        """
        Gets the full, absolute path for a given output key.

        This method automatically creates the necessary subdirectory just before
        returning the path.

        Args:
            key (str): The key for the desired output (e.g., 'prediction').

        Returns:
            The absolute path for the output file.
        """
        if key not in self._config:
            raise KeyError(f"Output key '{key}' not found in ReconstructOutputManager configuration.")

        subdirectory, filename_template = self._config[key]
        
        # Format the filename with the subject_id
        filename = filename_template.format(subject_id=os.path.basename(self.base_dir))
        
        # Construct the full path
        full_path = os.path.join(self.base_dir, subdirectory, filename)
        
        # Just-in-time directory creation
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        
        return full_path

    def get_all_paths(self) -> dict:
        """Returns a dictionary of all configured paths."""
        return {key: self.get_path(key) for key in self._config}