import os
import sys
import argparse
import glob as _glob

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


def _resolve_view(view_name, explicit_path, input_dir, pattern, flag_hint):
    """Resolve one input view: explicit path wins, else glob from input_dir.

    Raises a clear error if neither source can supply the file.
    """
    if explicit_path:
        return explicit_path
    if input_dir:
        return _find_single(input_dir, pattern)
    raise FileNotFoundError(
        f"No source for the {view_name} image: pass {flag_hint} "
        f"or provide --input-dir containing a '{pattern}' file."
    )


def run_full_pipeline_job(output_dir, output_prefix, device='cpu',
                          input_dir=None, sax_path=None, ch2_path=None, ch4_path=None, quality_control=False):
    """
    Pure Python function to run the full pipeline.
    Accepts standard types, not argparse objects.

    Each view is resolved independently: an explicit *_path argument takes
    precedence; otherwise the file is discovered by glob inside input_dir.
    """
    # Imported here, not at module scope: pulls in torch/nnunetv2 (~3.5s).
    from cardio_form.pipeline import CardioForm
    logger.info("--- Initializing CardioForm Pipeline ---")

    sax_path  = _resolve_view("SAX",  sax_path, input_dir, _SAX_PATTERN, "-sax/--sax-file")
    ch2_path  = _resolve_view("2CH",  ch2_path, input_dir, _CH2_PATTERN, "-ch2/--ch2-file")
    ch4_path  = _resolve_view("4CH",  ch4_path, input_dir, _CH4_PATTERN, "-ch4/--ch4-file")
    logger.info("Found all required input CINE images.")

    try:
        pipeline = CardioForm(device=device)

        pipeline.run_full_pipeline(
            sax_path=sax_path,
            ch2_path=ch2_path,
            ch4_path=ch4_path,
            output_dir=output_dir,
            output_prefix=output_prefix,
            quality_control=quality_control,
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
        description="Run the full CardioForm pipeline: 2D Segmentation -> 3D Reconstruction.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("--input-dir", required=False, default=None, help="Directory of raw CINE MRI images; views are auto-discovered by glob. Optional if the explicit view flags are given.")
    parser.add_argument("-sax", "--sax-file", default=None, help="Path to SAX NIfTI (overrides discovery from --input-dir).")
    parser.add_argument("-ch2", "--ch2-file", default=None, help="Path to 2CH NIfTI (overrides discovery from --input-dir).")
    parser.add_argument("-ch4", "--ch4-file", default=None, help="Path to 4CH NIfTI (overrides discovery from --input-dir).")
    parser.add_argument("--output-dir", required=True,  help="Path to the root directory where all outputs will be saved.")
    parser.add_argument("-p", "--output-prefix", required=True, help="Prefix for all output filenames (inferred if not provided).")
    parser.add_argument("--device", default="cpu", choices=['auto', 'cpu', 'cuda'], help="Device to run the models on.")
    
    parser.add_argument("-qc", "--quality-control", action="store_true", help="Also write diagnostic artefacts (sparse volume, back-projections).")

    args = parser.parse_args()
    main(args) 
