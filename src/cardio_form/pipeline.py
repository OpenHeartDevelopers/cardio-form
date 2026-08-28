# In cardio_form/pipeline.py

import os

from cardio_form.utils import configure_logging
logger = configure_logging('CardioFormPipeline')

# Each of these modules contains complex, low-level logic and pulls in torch.
# Importing this module is therefore expensive; the CLI defers importing
# cardio_form.pipeline until a job actually runs.
import cardio_form.reconstruct_3d as reconstruct_3d
import cardio_form.reconstruct_la_3d as reconstruct_la
import cardio_form.segment_2d as segment_2d  

from cardio_form.models  import default_model_manager

# Re-exported from config so `from cardio_form.pipeline import CHOICES_VIEW_TYPE`
# keeps working; config.py is the single definition.
from cardio_form.config import CHOICES_VIEW_TYPE

class CardioForm:
    """
    A high-level API for the Cardiac MRI processing pipeline.
    This class uses LAZY LOADING for all models to improve efficiency.
    """
    def __init__(self, device: str = 'cpu'):
        """
        Initializes the pipeline. THIS IS A VERY FAST, LIGHTWEIGHT OPERATION.
        No models are loaded at this stage.

        Args:
            device (str): The device to run models on ('auto', 'cpu', 'cuda').
        """
        if device == 'auto':
            import torch
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
            logger.info("Loading LA reconstruction model for the first time... ") 
            model_path = default_model_manager.get_model_path('la_reconstruction_3d')
            self._la_recon_model = reconstruct_la.load_la_model(model_path, self.device)
        return self._la_recon_model

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
        output_prefix: str = None,
        quality_control: bool = False,
    ) -> dict:
        """Runs ONLY the 3D reconstruction step of the pipeline.

        Set ``quality_control`` to also write the sparse volume and the three
        back-projections. Off by default: nothing downstream reads them.
        """
        
        logger.info(f"--- Starting 3D Reconstruction for {output_prefix} ---")

        # The first time this method is called, self.recon_model will trigger
        # the loading logic. The second time, it will be instant.
        reconstruction_outputs = reconstruct_3d.run_3d_reconstruction(
            sax_file=sax_path,
            ch2_file=ch2_file_path,
            ch4_file=ch4_file_path,
            output_dir=output_dir,
            output_prefix=output_prefix,
            model=self.recon_model, # Accessing the property, not the _variable
            device_str=self.device,
            compute_qc=quality_control
        )
        
        logger.info(f"Reconstruction for {output_prefix} complete.")
        return reconstruction_outputs

    def segment(self, input_path: str, output_dir: str, view_type: str, output_prefix: str) -> str:
        """
        Runs 2D segmentation for a specific view (SAX, LAX 2CH, or LAX 4CH).

        Args:
            input_path (str): Path to the input NIfTI file.
            output_dir (str): Root directory to save the output segmentation.
            view_type (str): The type of view. Must be one of ['sax', 'lax_2ch', 'lax_4ch'].
            output_prefix (str, optional): Name for the case. Inferred if not provided.

        Returns:
            The absolute path to the generated segmentation file.
        """
        if view_type not in CHOICES_VIEW_TYPE:
            raise ValueError(f"Invalid view_type: '{view_type}'. Must be one of [{'|'.join(CHOICES_VIEW_TYPE)}].")
        
        logger.info(f"--- Starting {view_type.upper()} Segmentation for {output_prefix} ---")

        # 1. Select the correct model directory using our properties
        model_path_map = {
            'sax': self.sax_model_path,
            'lax_2ch': self.lax2ch_model_path,
            'lax_4ch': self.lax4ch_model_path
        }
        model_path = model_path_map[view_type]

        output_path = segment_2d.run_segmentation(
            input_path=input_path,
            output_dir=output_dir,
            output_prefix=output_prefix,
            view_type=view_type,
            model_path=model_path, # Pass the full file path
            device=self.device
        )
        return output_path
    
    def reconstruct_la_3d(self, ch2_file: str, ch4_file: str, output_dir: str, output_prefix: str,
                          quality_control: bool = False) -> dict : 
        """ 
        Runs ONLY the LA 3D reconstruction step of the pipeline. 

        Set ``quality_control`` to also write the sparse volume and both
        back-projections.
        """ 
        logger.info(f"--- Starting LA 3D Reconstruction for {output_prefix} ---") 

        la_reconstruction_outputs = reconstruct_la.run_la_reconstruction(
            model=self.la_recon_model,
            ch2_file=ch2_file,
            ch4_file=ch4_file, 
            output_dir=output_dir,
            output_prefix=output_prefix,
            device_str=self.device,
            compute_qc=quality_control
        ) 

        logger.info(f"LA 3D Reconstruction for {output_prefix} complete.") 
        return la_reconstruction_outputs

    def left_complete(
        self,
        la_file: str,
        whs_file: str,
        output_dir: str,
        output_prefix: str,
        selected_groups=None,
    ) -> str:
        """
        Enhance a whole-heart segmentation with the LA network's left-side output.

        The LA volume is resampled onto the whole-heart grid and written only
        where the whole-heart map is background, so existing structure is never
        modified. Loads no model; pure geometry.

        Args:
            la_file (str): LA 3D segmentation NIfTI.
            whs_file (str): Whole-heart segmentation NIfTI to enhance.
            output_dir (str): Directory for the output file.
            output_prefix (str): Prefix for the output filename.
            selected_groups (list): Group names from the left label space to
                merge. ``None`` merges every structure with a merge target.

        Returns:
            str: Path to the written segmentation.
        """
        from cardio_form import geometry
        from cardio_form import io as cf_io
        from cardio_form.cli.left_complete import build_merge_mapping
        from cardio_form.output_managers import OutputManager

        logger.info(f"--- Starting left-heart completion for {output_prefix} ---")

        outputs = OutputManager(output_dir=output_dir, output_prefix=output_prefix)
        mapping = build_merge_mapping(selected_groups)

        whs_data, whs_affine, whs_header = cf_io.load_label_map(whs_file)
        la_nifti = cf_io.load_nifti(la_file)

        la_on_grid = geometry.resample_label_map_to(la_nifti, whs_data.shape, whs_affine)
        completed = geometry.fill_into_background(whs_data, la_on_grid, mapping)

        output_path = outputs.get_path('left_complete')
        cf_io.save_nifti(completed, whs_affine, whs_header, output_path)

        logger.info(f"Left-heart completion for {output_prefix} complete.")
        return output_path

    def run_full_pipeline(
        self,
        sax_path: str,
        ch2_path: str,
        ch4_path: str,
        output_dir: str,
        output_prefix: str,
        quality_control: bool = False,
    ):
        """
        Runs the full end-to-end pipeline for a single subject.

        Accepts explicit paths to the three required CINE NIfTI files.
        File discovery (e.g. glob or YAML-driven resolution) is the caller's
        responsibility.

        Args:
            sax_path (str): Path to the SAX CINE NIfTI file.
            ch2_path (str): Path to the LAX 2-chamber CINE NIfTI file.
            ch4_path (str): Path to the LAX 4-chamber CINE NIfTI file.
            output_dir (str): Root directory for all pipeline outputs.
            output_prefix (str): Unique ID / prefix for all output filenames.
        """

        logger.info(f"\n===== Starting Full Pipeline for prefix: {output_prefix} =====")
        logger.info(f"  - All outputs will be saved in: {output_dir}")

        logger.info(f"Ensuring output directory exists: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)

        # --- 2. Run 2D Segmentation for each view ---
        logger.info("\n--- Step 1: Running 2D Segmentation ---")

        sax_seg_path = self.segment(
            input_path=sax_path,
            output_dir=output_dir,
            output_prefix=output_prefix,
            view_type='sax',
        )

        lax2ch_seg_path = self.segment(
            input_path=ch2_path,
            output_dir=output_dir,
            output_prefix=output_prefix,
            view_type='lax_2ch',
        )

        lax4ch_seg_path = self.segment(
            input_path=ch4_path,
            output_dir=output_dir,
            output_prefix=output_prefix,
            view_type='lax_4ch',
        )

        # --- 3. Run 3D Reconstruction ---
        logger.info("\n--- Step 2: Running 3D Reconstruction ---")
        
        self.reconstruct(
            sax_path=sax_seg_path,
            ch2_file_path=lax2ch_seg_path,
            ch4_file_path=lax4ch_seg_path,
            output_dir=output_dir, # The main output goes in the top-level subject folder
            output_prefix=output_prefix,
            quality_control=quality_control
        )

        logger.info(f"\n===== Full Pipeline for {output_prefix} Complete! =====")
