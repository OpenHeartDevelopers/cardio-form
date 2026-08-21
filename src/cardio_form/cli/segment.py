import sys
import argparse

from cardio_form.config import CHOICES_VIEW_TYPE
from cardio_form.utils import configure_logging
logger = configure_logging('ScriptSegment2D')

def run_segmentation_job(input_path, output_dir, view_type, output_prefix, device='cpu'):  
    """
    Pure Python function to run the segmentation. 
    Accepts standard types, not argparse objects.
    """
    # Imported here, not at module scope: pulls in torch/nnunetv2 (~3.5s).
    from cardio_form.pipeline import CardioForm
    logger.info("--- Initializing CardioForm Pipeline ---")
    
    try:
        pipeline = CardioForm(device=device)
        
        # Call the high-level method from our pipeline class.
        pipeline.segment(
            input_path=input_path,
            output_dir=output_dir,
            view_type=view_type,
            output_prefix=output_prefix
        )
        logger.info("\n--- Segmentation job finished successfully! ---")
        return True
        
    except Exception as e:
        logger.error(f"FATAL: Failed to run segmentation. Error: {e}")
        raise e

def main(args):
    """
    CLI Wrapper. Only handles Argument Parsing.
    """
    try:
        run_segmentation_job(
            input_path=args.input,
            output_dir=args.output_dir,
            view_type=args.view_type,
            output_prefix=args.output_prefix,
            device=args.device
        )
    except Exception:
        sys.exit(1)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run 2D segmentation on a cardiac MRI NIfTI file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("--input", required=True, help="Path to the input NIfTI file.")
    parser.add_argument("--output-dir", required=True, help="Root directory to save the output segmentation.")
    parser.add_argument("--view-type", required=True, choices=CHOICES_VIEW_TYPE, help="The type of cardiac view to segment.")
    parser.add_argument("-p", "--output-prefix", required=True, help="Prefix for the output filename (e.g., 'subject_001_cine').")
    parser.add_argument("--device", default="auto", choices=['auto', 'cpu', 'cuda'], help="Device to run the model on.")
    
    args = parser.parse_args()

    main(args)