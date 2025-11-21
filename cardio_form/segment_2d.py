# in cardio_form/segment_2d.py
import os
import torch
import numpy as np
import SimpleITK as sitk
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO

from cardio_form.utils import configure_logging
logger = configure_logging('Segment2D')

def run_segmentation(
    input_path: str,
    output_dir: str, 
    output_prefix: str,
    view_type: str,
    model_path: str,  # <-- CHANGE #1: We now accept the full file path
    device: str = 'cuda'
):
    """
    Runs nnUNetv2 segmentation on a single NIfTI file.

    Args:
        input_path (str): Absolute path to the input NIfTI file.
        output_dir (str): Directory to save the output segmentation.
        output_prefix (str): Prefix for the output filename.
        view_type (str): The view being segmented ('sax', 'lax_2ch', 'lax_4ch'). Only used for the output name
        model_path (str): Absolute path to the trained nnU-Net checkpoint FILE
                          (e.g., '/path/to/weights/sax_segment_checkpoint_final.pth').
        device (str): Device to run inference on ('cuda' or 'cpu').
    """
    # --- YOUR SUGGESTION ---
    # Dynamically determine the model directory and checkpoint name from the path
    model_dir = os.path.dirname(os.path.dirname(model_path)) # nnU-Net needs the parent of the 'fold_all' folder
    checkpoint_name = os.path.basename(model_path)
    
    logger.info(f"--- Starting nnU-Net Segmentation ---")
    logger.info(f"  Input: {input_path}")
    logger.info(f"  Model Directory: {model_dir}")
    logger.info(f"  Checkpoint: {checkpoint_name}")

    # 1. Instantiate the nnUNetPredictor with standard settings
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=False,  # Mirroring is disabled as per original scripts
        device=torch.device(device),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True
    )

    # 2. Initialize the predictor with the specific model folder
    predictor.initialize_from_trained_model_folder(
        model_dir,
        use_folds=('all',), # Note: This might need to be dynamic too in the future, but is fine for now
        checkpoint_name=checkpoint_name,
    )

    # 3. Read the input image using nnU-Net's own reader
    img, props = SimpleITKIO().read_images([input_path])
    
    # 4. Run prediction
    logger.info("  Running prediction...")
    pred_array = predictor.predict_single_npy_array(img, props, None, None, False)
    
    # 5. Save the output image, preserving original geometry
    # The output from predict_single_npy_array is a numpy array. We need to
    # convert it back to a SimpleITK image, restoring its geometric properties.
    pred_image = sitk.GetImageFromArray(pred_array.astype(np.uint8))
    pred_image.SetDirection(props['sitk_stuff']['direction'])
    pred_image.SetOrigin(props['sitk_stuff']['origin'])
    pred_image.SetSpacing(props['sitk_stuff']['spacing'])

    # --- Construct the output path ---
    # The filename is now created consistently inside the function.
    output_filename = f"{output_prefix}_2D_seg_{view_type}.nii.gz"
    final_output_path = os.path.join(output_dir, output_filename)

    sitk.WriteImage(pred_image, final_output_path, useCompression=True)
    
    logger.info(f"  Segmentation complete. Output saved to: {final_output_path}")
    return final_output_path # Return the path that was actually created