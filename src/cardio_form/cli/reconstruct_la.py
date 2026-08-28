import sys
import argparse

from cardio_form.utils import configure_logging
logger = configure_logging('ScriptReconstructLA3D')

# The 2D LAX segmentation models and the LA 3D network use different label
# spaces. These map the former onto the latter, giving 1=LA, 2=LV, 3=RA, 4=RV.
# Ported from the upstream pipeline's LA_2CH/LA_4CH_label_mapping.py.
LA_MAP_2CH = {1: 2, 2: 0, 3: 1}
LA_MAP_4CH = {1: 2, 2: 0, 3: 4, 4: 1, 5: 3}

# Label values the 2D LAX models can emit. Anything else is not a LAX
# segmentation in the expected space.
RAW_LAX_LABELS_2CH = {0, 1, 2, 3}
RAW_LAX_LABELS_4CH = {0, 1, 2, 3, 4, 5}


def _remap_lax_input(input_path, output_path, mapping, allowed_labels, view):
    """Remap one LAX segmentation into the LA network's label space.

    Returns the path of the written file.
    """
    import numpy as np

    from cardio_form import geometry
    from cardio_form import io as cf_io

    data, affine, header = cf_io.load_label_map(input_path)

    present = set(np.unique(data).astype(int).tolist())
    unexpected = present - allowed_labels
    if unexpected:
        raise ValueError(
            f"{view} input '{input_path}' contains label(s) {sorted(unexpected)}, "
            f"which are outside the expected 2D LAX label space {sorted(allowed_labels)}. "
            f"Pass the raw output of `cardioform segment --view-type lax_{view.lower()}`; "
            f"an already-remapped or unrelated file will not reconstruct correctly."
        )

    logger.info(f"Remapping {view} input into the LA network label space.")
    remapped = geometry.remap_labels(data, mapping)
    cf_io.save_nifti(remapped, affine, header, output_path)
    return output_path


def run_la_reconstruction_job(ch2_file, ch4_file, output_dir, output_prefix, device='cpu',
                              quality_control=False):
    """
    Pure Python function to run the LA 3D reconstruction. 
    Accepts standard types, not argparse objects.

    The LAX inputs must be remapped into the LA network's label space before
    inference. ``run_la_reconstruction`` takes paths, so the remapped volumes
    have to exist on disk; when ``quality_control`` is off they go to a
    temporary directory and are removed afterwards.
    """
    # Imported here, not at module scope: pulls in torch/nnunetv2 (~3.5s).
    import contextlib
    import os
    import tempfile

    from cardio_form.pipeline import CardioForm
    from cardio_form.output_managers import OutputManager
    logger.info("--- Initializing CardioForm Pipeline ---")
    
    try:
        outputs = OutputManager(output_dir=output_dir, output_prefix=output_prefix)

        # Keep the remapped inputs only when they were asked for.
        with contextlib.ExitStack() as stack:
            if quality_control:
                ch2_target = outputs.get_path('la_input_2ch')
                ch4_target = outputs.get_path('la_input_4ch')
            else:
                scratch = stack.enter_context(tempfile.TemporaryDirectory(prefix='cardioform_la_'))
                ch2_target = os.path.join(scratch, 'la_input_2ch.nii.gz')
                ch4_target = os.path.join(scratch, 'la_input_4ch.nii.gz')

            ch2_mapped = _remap_lax_input(
                ch2_file, ch2_target, LA_MAP_2CH, RAW_LAX_LABELS_2CH, '2CH',
            )
            ch4_mapped = _remap_lax_input(
                ch4_file, ch4_target, LA_MAP_4CH, RAW_LAX_LABELS_4CH, '4CH',
            )

            pipeline = CardioForm(device=device)

            # Call the high-level method from our pipeline class.
            pipeline.reconstruct_la_3d(
                ch2_file=ch2_mapped,
                ch4_file=ch4_mapped,
                output_dir=output_dir,
                output_prefix=output_prefix,
                quality_control=quality_control,
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
            device=args.device,
            quality_control=args.quality_control
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
    parser.add_argument("-qc", "--quality-control", action="store_true", help="Also write diagnostic artefacts (sparse volume, back-projections, remapped inputs).")
    
    args = parser.parse_args()
    main(args)