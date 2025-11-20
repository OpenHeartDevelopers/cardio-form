# in scripts/run_segmentation.py

import os
import sys
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cardio_form.pipeline import CardioForm, CHOICES_VIEW_TYPE
from cardio_form.utils import configure_logging
logger = configure_logging('ScriptSegment2D')

def main():
    parser = argparse.ArgumentParser(
        description="Run 2D segmentation on a cardiac MRI NIfTI file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("--input", required=True, help="Path to the input NIfTI file.")
    parser.add_argument("--output-dir", required=True, help="Root directory to save the output segmentation.")
    parser.add_argument("--view-type", required=True, choices=CHOICES_VIEW_TYPE, 
                        help="The type of cardiac view to segment.")
    parser.add_argument("--subject-id", help="Optional name for the case. Inferred from input filename if not provided.")
    parser.add_argument("--device", default="auto", choices=['auto', 'cpu', 'cuda'], help="Device to run the model on.")
    
    args = parser.parse_args()

    logger.info("--- Initializing CardioForm Pipeline ---")
    pipeline = CardioForm(device=args.device)

    # Call the high-level segment method from our pipeline class
    pipeline.segment(
        input_path=args.input,
        output_dir=args.output_dir,
        view_type=args.view_type,
        subject_id=args.subject_id
    )
    
    logger.info("\n--- Script finished successfully! ---")

if __name__ == "__main__":
    main()