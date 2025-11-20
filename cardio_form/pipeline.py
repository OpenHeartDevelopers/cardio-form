# In cardio_form/pipeline.py

import os
import torch


from cardio_form.utils import configure_logging
logger = configure_logging('CardioFormPipeline')

# Each of these modules contains complex, low-level logic.
import cardio_form.reconstruct_3d as reconstruct_3d
import cardio_form.segment_2d as segment_2d  

from cardio_form.models  import default_model_manager

CHOICES_VIEW_TYPE = ['sax', 'lax_2ch', 'lax_4ch']

class CardioForm:
    """
    A high-level API for the Cardiac MRI processing pipeline.
    This class uses LAZY LOADING for all models to improve efficiency.
    """
    def __init__(self, device: str = 'auto'):
        """
        Initializes the pipeline. THIS IS A VERY FAST, LIGHTWEIGHT OPERATION.
        No models are loaded at this stage.

        Args:
            device (str): The device to run models on ('auto', 'cpu', 'cuda').
        """
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        logger.info(f"CardioForm pipeline configured for device: '{self.device}'")
        
        # --- State variables ---
        # We store the model OBJECTS here, but initialize them to None.
        # The preceding underscore indicates they are "private" to this class.
        self._recon_model = None # 3D whole heart reconstruction model
        self._sax_seg_model = None # 2D SAX model For the future
        self._la_recon_model = None # 3D LA 3D segmentation 
        self._lax_2ch_seg_model = None # For the future
        self._lax_4ch_seg_model = None # For the future
        # etc.

    @property
    def recon_model(self):
        """
        Lazy-loads the 3D reconstruction model.

        The first time this property is accessed, it will load the model.
        On all subsequent calls, it will instantly return the already-loaded model.
        """
        if self._recon_model is None:
            logger.info("Loading reconstruction model for the first time...")
            model_path = default_model_manager.get_model_path('reconstruction_3d')
            self._recon_model = reconstruct_3d.load_model(model_path, self.device)
        return self._recon_model

    @property 
    def la_recon_model(self) : 
        if self._la_recon_model is None: 
            logger.info("Loading Model for the first time... ") 
            # model_path = default_model_manager.get_model_path('ID')
            # self._la_recon_model = segment_2d.load_model(model_path, self.device)
            pass 
        self._la_recon_model

    @property
    def sax_model_path(self) -> str: # <-- RENAMED
        """Lazy-gets the path to the SAX segmentation model CHECKPOINT FILE."""
        if self._sax_seg_model is None:
            logger.info("Locating SAX segmentation model...")
            self._sax_seg_model = default_model_manager.get_model_path('segment_sax')
        return self._sax_seg_model

    @property
    def lax2ch_model_path(self) -> str: # <-- RENAMED
        """Lazy-gets the path to the LAX 2CH segmentation model CHECKPOINT FILE."""
        if self._lax_2ch_seg_model is None:
            logger.info("Locating LAX 2CH segmentation model...")
            self._lax_2ch_seg_model = default_model_manager.get_model_path('segment_lax_2ch')
        return self._lax_2ch_seg_model

    @property
    def lax4ch_model_path(self) -> str: # <-- RENAMED
        """Lazy-gets the path to the LAX 4CH segmentation model CHECKPOINT FILE."""
        if self._lax_4ch_seg_model is None:
            logger.info("Locating LAX 4CH segmentation model...")
            self._lax_4ch_seg_model = default_model_manager.get_model_path('segment_lax_4ch')
        return self._lax_4ch_seg_model

    # --- Public Methods ---

    def reconstruct(
        self, 
        sax_path: str, 
        ch2_file_path: str, 
        ch4_file_path: str, 
        output_dir: str, 
        subject_id: str = None
    ) -> dict:
        """Runs ONLY the 3D reconstruction step of the pipeline."""
        if not subject_id:
            subject_id = os.path.basename(sax_path).split('.')[0]
            subject_id = f'WH_{subject_id}'
            subject_id = subject_id.replace('SAX_', '')
            logger.info(f"Subject ID not provided. Inferred as: '{subject_id}'")

        logger.info(f"--- Starting 3D Reconstruction for {subject_id} ---")

        # The first time this method is called, self.recon_model will trigger
        # the loading logic. The second time, it will be instant.
        reconstruction_outputs = reconstruct_3d.run_3d_reconstruction(
            sax_file=sax_path,
            ch2_file=ch2_file_path,
            ch4_file=ch4_file_path,
            output_dir=output_dir,
            subject_id=subject_id,
            model=self.recon_model, # Accessing the property, not the _variable
            device_str=self.device
        )
        
        logger.info(f"Reconstruction for {subject_id} complete.")
        return reconstruction_outputs

    def segment(self, input_path: str, output_dir: str, view_type: str, subject_id: str = None) -> str:
        """
        Runs 2D segmentation for a specific view (SAX, LAX 2CH, or LAX 4CH).

        Args:
            input_path (str): Path to the input NIfTI file.
            output_dir (str): Root directory to save the output segmentation.
            view_type (str): The type of view. Must be one of ['sax', 'lax_2ch', 'lax_4ch'].
            subject_id (str, optional): Name for the case. Inferred if not provided.

        Returns:
            The absolute path to the generated segmentation file.
        """
        if view_type not in CHOICES_VIEW_TYPE:
            raise ValueError(f"Invalid view_type: '{view_type}'. Must be one of [{'|'.join(CHOICES_VIEW_TYPE)}].")
        
        if not subject_id:
            subject_id = os.path.basename(input_path).split('.')[0]

        logger.info(f"--- Starting {view_type.upper()} Segmentation for {subject_id} ---")

        # 1. Select the correct model directory using our properties
        model_path_map = {
            'sax': self.sax_model_path,
            'lax_2ch': self.lax2ch_model_path,
            'lax_4ch': self.lax4ch_model_path
        }
        model_path = model_path_map[view_type]

        # 2. Define a clean, predictable output path
        output_filename = f"{subject_id}_seg_{view_type}.nii.gz"
        subject_output_dir = os.path.join(output_dir, subject_id)
        os.makedirs(subject_output_dir, exist_ok=True)
        output_path = os.path.join(subject_output_dir, output_filename)

        # 3. Call the engine to do the work
        segment_2d.run_segmentation(
            input_path=input_path,
            output_path=output_path,
            model_path=model_path, # Pass the full file path
            device=self.device
        )
        return output_path
