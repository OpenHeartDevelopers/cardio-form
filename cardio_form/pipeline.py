# In cardio_form/pipeline.py

import os
import torch

# --- We import our "engines" ---
# Each of these modules contains complex, low-level logic.
import cardio_form.reconstruct_3d as reconstruct_3d
# from . import segment_2d  # We will add this module later

from cardio_form.models  import default_model_manager

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
        
        print(f"CardioForm pipeline configured for device: '{self.device}'")
        
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
            print("Loading reconstruction model for the first time...")
            model_path = default_model_manager.get_model_path('reconstruction_3d')
            self._recon_model = reconstruct_3d.load_model(model_path, self.device)
        return self._recon_model

    @property
    def sax_seg_model(self):
        """Lazy-loads the SAX segmentation model. (EXAMPLE FOR THE FUTURE)"""
        if self._sax_seg_model is None:
            print("Loading SAX segmentation model for the first time...")
            # model_path = default_model_manager.get_model_path('segment_sax')
            # self._sax_seg_model = segment_2d.load_model(model_path, self.device)
            pass # Placeholder for now
        return self._sax_seg_model
    
    @property 
    def la_recon_model(self) : 
        if self._la_recon_model is None: 
            print("Loading Model for the first time... ") 
            # model_path = default_model_manager.get_model_path('ID')
            # self._la_recon_model = segment_2d.load_model(model_path, self.device)
            pass 
        self._la_recon_model

    @property 
    def lax_2ch_seg_model(self) : 
        if self._lax_2ch_seg_model is None: 
            print("Loading Model for the first time... ") 
            # model_path = default_model_manager.get_model_path('ID')
            # self._lax_2ch_seg_model = segment_2d.load_model(model_path, self.device)
            pass 
        self._lax_2ch_seg_model

    @property 
    def lax_4ch_seg_model(self) : 
        if self._lax_4ch_seg_model is None: 
            print("Loading Model for the first time... ") 
            # model_path = default_model_manager.get_model_path('ID')
            # self._lax_4ch_seg_model = segment_2d.load_model(model_path, self.device)
            pass 
        self._lax_4ch_seg_model


    # --- Public Methods ---

    def reconstruct(
        self, 
        sax_path: str, 
        lax2ch_path: str, 
        lax4ch_path: str, 
        output_dir: str, 
        subject_id: str = None
    ) -> dict:
        """Runs ONLY the 3D reconstruction step of the pipeline."""
        if not subject_id:
            subject_id = os.path.basename(sax_path).split('.')[0]
            print(f"Subject ID not provided. Inferred as: '{subject_id}'")

        print(f"--- Starting 3D Reconstruction for {subject_id} ---")

        # The first time this method is called, self.recon_model will trigger
        # the loading logic. The second time, it will be instant.
        reconstruction_outputs = reconstruct_3d.run_3d_reconstruction(
            sax_file=sax_path,
            ch2_file=lax2ch_path,
            ch4_file=lax4ch_path,
            output_dir=output_dir,
            subject_id=subject_id,
            model=self.recon_model, # Accessing the property, not the _variable
            device_str=self.device
        )
        
        print(f"Reconstruction for {subject_id} complete.")
        return reconstruction_outputs

    # Example of a future method
    def segment_sax(self, raw_sax_mri_path: str, output_dir: str, subject_id: str):
        """Runs ONLY the SAX segmentation step. (EXAMPLE FOR THE FUTURE)"""
        print(f"--- Starting SAX Segmentation for {subject_id} ---")
        # This would trigger the loading of the SAX model on its first run.
        # result = segment_2d.run_segmentation(
        #     input_path=raw_sax_mri_path,
        #     output_dir=output_dir,
        #     subject_id=subject_id,
        #     model=self.sax_seg_model, # Accessing the sax_seg_model property
        #     device_str=self.device
        # )
        # return result
        pass