import sys
import argparse

from cardio_form.utils import configure_logging
logger = configure_logging('ScriptReconstructLA3D')

def run_la_reconstruction_job(ch2_file, ch4_file, output_dir, output_prefix, device='cpu'):
    """
    Pure Python function to run the LA 3D reconstruction. 
    Accepts standard types, not argparse objects.
    """
    # Imported here, not at module scope: pulls in torch/nnunetv2 (~3.5s).
    from cardio_form.pipeline import CardioForm
    logger.info("--- Initializing CardioForm Pipeline ---")
    
    try:
        pipeline = CardioForm(device=device)
        
        # Call the high-level method from our pipeline class.
        pipeline.reconstruct_la_3d(
            ch2_file=ch2_file,
            ch4_file=ch4_file,
            output_dir=output_dir,
            output_prefix=output_prefix
        )
        logger.info("\n--- LA 3D Reconstruction job finished successfully! ---")
        return True
        
    except Exception as e:
        logger.error(f"FATAL: Failed to run LA 3D reconstruction. Error: {e}")
        raise e

def main(args):
    """
    CLI Wrapper. Only handles Argument Parsing.
    """
    try:
        run_la_reconstruction_job(
            ch2_file=args.ch2_file,
            ch4_file=args.ch4_file,
            output_dir=args.output_dir,
            output_prefix=args.output_prefix,
            device=args.device
        )
    except Exception:
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run LA 3D reconstruction from 2D cardiac MRI segmentations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter # Shows default values in help
    )
    
    # --- Input Arguments ---
    parser.add_argument("-ch2", "--ch2-file", required=True, help="Path to the LAX 2-chamber segmentation NIfTI file.")
    parser.add_argument("-ch4", "--ch4-file", required=True, help="Path to the LAX 4-chamber segmentation NIfTI file.")
    
    # --- Output Arguments ---
    parser.add_argument("-o", "--output-dir", required=True, help="Directory to save all output files.")
    parser.add_argument("-p", "--output-prefix", required=True, help="Prefix for all output filenames (e.g., 'subject_001_cine').")
    
    # --- Configuration Arguments ---
    parser.add_argument("--model-version", default="default", help="Version of the LA reconstruction model to use (from models.yaml).")
    parser.add_argument("--device", default="cpu", choices=['cpu', 'cuda'], help="Device to run the model on.")
    
    args = parser.parse_args()
    main(args)