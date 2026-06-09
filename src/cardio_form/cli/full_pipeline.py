import os
import sys
import argparse
import glob as _glob

from cardio_form.pipeline import CardioForm
from cardio_form.utils import configure_logging
logger = configure_logging(__name__)

_SAX_PATTERN   = '*CINE_image_SAX*.nii.gz'
_CH2_PATTERN   = '*CINE_image_CH2*.nii.gz'
_CH4_PATTERN   = '*CINE_image_CH4*.nii.gz'


def _find_single(directory: str, pattern: str) -> str:
    matches = _glob.glob(os.path.join(directory, pattern))
    if not matches:
        raise FileNotFoundError(
            f"No file matching '{pattern}' found in {directory}"
        )
    if len(matches) > 1:
        raise ValueError(
            f"Multiple files matching '{pattern}' found in {directory}: {matches}"
        )
    return matches[0]


def run_full_pipeline_job(input_dir, output_dir, output_prefix, device='cpu'):
    """
    Pure Python function to run the full pipeline.
    Accepts standard types, not argparse objects.
    """
    logger.info("--- Initializing CardioForm Pipeline ---")

    sax_path  = _find_single(input_dir, _SAX_PATTERN)
    ch2_path  = _find_single(input_dir, _CH2_PATTERN)
    ch4_path  = _find_single(input_dir, _CH4_PATTERN)
    logger.info("Found all required input CINE images.")

    try:
        pipeline = CardioForm(device=device)

        pipeline.run_full_pipeline(
            sax_path=sax_path,
            ch2_path=ch2_path,
            ch4_path=ch4_path,
            output_dir=output_dir,
            output_prefix=output_prefix,
        )
        logger.info("\n--- Full pipeline job finished successfully! ---")
        return True

    except Exception as e:
        logger.error(f"FATAL: Failed to run full pipeline. Error: {e}")
        raise e

def main(args):
    """
    CLI Wrapper. Only handles Argument Parsing.
    """
    try:
        run_full_pipeline_job(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            output_prefix=args.output_prefix,
            device=args.device
        )
    except Exception:
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the full CardioForm pipeline: 2D Segmentation -> 3D Reconstruction.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("--input-dir", required=True,  help="Path to the input directory containing the raw CINE MRI images.")
    parser.add_argument("--output-dir", required=True,  help="Path to the root directory where all outputs will be saved.")
    parser.add_argument("-p", "--output-prefix", required=True, help="Prefix for all output filenames (inferred if not provided).")
    parser.add_argument("--device", default="cpu", choices=['auto', 'cpu', 'cuda'], help="Device to run the models on.")
    
    args = parser.parse_args()
    main(args) 
