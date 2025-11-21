# in scripts/run_full_pipeline.py

import os
import sys
import argparse
import glob

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cardio_form.pipeline import CardioForm
from cardio_form.utils import configure_logging
logger = configure_logging(__name__)

def main(args):

    # --- Initialize the Pipeline ---
    logger.info("--- Initializing CardioForm Pipeline ---")
    pipeline = CardioForm(device=args.device)

    # --- Determine the Output Prefix ---
    # If the user doesn't provide a prefix, we create a sensible default
    # from the input directory's name.
    if args.output_prefix:
        output_prefix = args.output_prefix
    else:
        output_prefix = os.path.basename(os.path.normpath(args.input_dir))
        logger.info(f"Output prefix not provided. Inferred as: '{output_prefix}'")

    # --- Run the Full Pipeline ---
    pipeline.run_full_pipeline(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        output_prefix=output_prefix # Pass the new prefix
    )

    
    logger.info("--- Full pipeline finished successfully! ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the full CardioForm pipeline: 2D Segmentation -> 3D Reconstruction.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("--input-dir", required=True,  help="Path to the input directory containing the raw CINE MRI images.")
    parser.add_argument("--output-dir", required=True,  help="Path to the root directory where all outputs will be saved.")
    parser.add_argument("-p", "--output-prefix", required=False, help="Prefix for all output filenames (inferred if not provided).")
    parser.add_argument("--device", default="cpu", choices=['auto', 'cpu', 'cuda'], help="Device to run the models on.")
    
    args = parser.parse_args()
    main(args) 
