# in scripts/run_reconstruction.py

import os
import sys
import argparse

# This adds the project root to the Python path, allowing us to import `cardio_form`
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cardio_form.pipeline import CardioForm

def main(args):
    """
    Command-line interface for running the 3D reconstruction step of the CardioForm pipeline.
    """

    print("--- Initializing CardioForm Pipeline ---")
    try:
        # Instantiate the main pipeline class, passing the model version and device.
        # This will lazy-load the model when `reconstruct` is called.
        pipeline = CardioForm(
            device=args.device, 
            recon_model_version=args.model_version
        )
    except Exception as e:
        print(f"FATAL: Failed to initialize CardioForm pipeline. Error: {e}")
        return

    # --- Run the Reconstruction ---
    # Call the high-level method from our pipeline class.
    pipeline.reconstruct(
        sax_path=args.sax_file,
        ch2_file_path=args.ch2_file,
        ch4_file_path=args.ch4_file,
        output_dir=args.output_dir,
        subject_id=args.subject_id
    )
    
    print("\n--- Script finished successfully! ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run 3D reconstruction from 2D cardiac MRI segmentations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter # Shows default values in help
    )
    
    # --- Input Arguments ---
    parser.add_argument("--sax-file", required=True, help="Path to the SAX segmentation NIfTI file.")
    parser.add_argument("--ch2-file", required=True, help="Path to the LAX 2-chamber segmentation NIfTI file.")
    parser.add_argument("--ch4-file", required=True, help="Path to the LAX 4-chamber segmentation NIfTI file.")
    
    # --- Output Arguments ---
    parser.add_argument("--output-dir", required=True, help="Directory to save all output files.")
    parser.add_argument("--subject-id", help="Optional name for the case. If not provided, it's inferred from the SAX filename.")
    
    # --- Configuration Arguments ---
    parser.add_argument("--model-version", default="default", help="Version of the reconstruction model to use (from models.yaml).")
    parser.add_argument("--device", default="auto", choices=['auto', 'cpu', 'cuda'], help="Device to run the model on.")
    
    args = parser.parse_args()
    main(args)