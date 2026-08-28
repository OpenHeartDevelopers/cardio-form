import sys
import argparse

from cardio_form.utils import configure_logging

# Create a module-level logger
logger = configure_logging('ScriptReconstruct3D')

def run_reconstruction_job(sax_path, ch2_path, ch4_path, output_dir, output_prefix, device='cpu',
                           quality_control=False):
    """
    Pure Python function to run the reconstruction. 
    Accepts standard types, not argparse objects.
    """
    # Imported here, not at module scope: pulls in torch/nnunetv2 (~3.5s).
    from cardio_form.pipeline import CardioForm
    logger.info("--- Initializing CardioForm Pipeline ---")
    
    try:
        pipeline = CardioForm(device=device)
        
        # Call the high-level method from our pipeline class.
        pipeline.reconstruct(
            sax_path=sax_path,
            ch2_file_path=ch2_path,
            ch4_file_path=ch4_path,
            output_dir=output_dir,
            output_prefix=output_prefix,
            quality_control=quality_control
        )
        logger.info("\n--- Reconstruction job finished successfully! ---")
        return True
        
    except Exception as e:
        logger.error(f"FATAL: Failed to run reconstruction. Error: {e}")
        # In a larger pipeline, you might want to raise the error up to the caller
        raise e

def main(args):
    """
    CLI Wrapper. Only handles Argument Parsing.
    """
    try:
        run_reconstruction_job(
            sax_path=args.sax_file,
            ch2_path=args.ch2_file,
            ch4_path=args.ch4_file,
            output_dir=args.output_dir,
            output_prefix=args.output_prefix,
            device=args.device,
            quality_control=args.quality_control
        )
    except Exception:
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run 3D reconstruction from 2D cardiac MRI segmentations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # --- Input Arguments ---
    parser.add_argument("-sax", "--sax-file", required=True, help="Path to SAX NIfTI.")
    parser.add_argument("-ch2", "--ch2-file", required=True, help="Path to 2CH NIfTI.")
    parser.add_argument("-ch4", "--ch4-file", required=True, help="Path to 4CH NIfTI.")
    
    # --- Output Arguments ---
    parser.add_argument("-o", "--output-dir", required=True, help="Output directory.")
    parser.add_argument("-p", "--output-prefix", required=True, help="Output filename prefix.")
    
    # --- Configuration Arguments ---
    parser.add_argument("--device", default="cpu", choices=['cpu', 'cuda'], help="Device to use.")
    
    parser.add_argument("-qc", "--quality-control", action="store_true", help="Also write diagnostic artefacts (sparse volume, back-projections).")

    args = parser.parse_args()
    main(args)
