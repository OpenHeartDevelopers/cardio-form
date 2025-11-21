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

def main():
    parser = argparse.ArgumentParser(
        description="Run the full CardioForm pipeline: 2D Segmentation -> 3D Reconstruction.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("--input-dir", required=True, 
                        help="Path to the input directory containing the raw CINE MRI images.")
    parser.add_argument("--output-dir", required=True, 
                        help="Path to the root directory where all outputs will be saved.")
    parser.add_argument("--subject-id", 
                        help="Optional name for the case. If not provided, it's inferred from the input directory name.")
    parser.add_argument("--device", default="cpu", choices=['auto', 'cpu', 'cuda'],
                        help="Device to run the models on.")
    
    args = parser.parse_args()

    # --- Initialize the Pipeline ---
    logger.info("--- Initializing CardioForm Pipeline ---")
    pipeline = CardioForm(device=args.device)

    # --- Run the Full Pipeline ---
    pipeline.run_full_pipeline(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        subject_id=args.subject_id
    )
    
    logger.info("--- Full pipeline finished successfully! ---")

if __name__ == "__main__":
    main() 
