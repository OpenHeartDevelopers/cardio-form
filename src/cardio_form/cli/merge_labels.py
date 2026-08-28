import sys
import argparse

from cardio_form.config import DEFAULT_LABEL_SPACE, LABEL_SPACES, label_space_path
from cardio_form.utils import configure_logging
logger = configure_logging('ScriptMergeLabels')

def run_merge_labels_job(input_path, output_path, user_labels_to_keep, value_after_merge=None, label_space=DEFAULT_LABEL_SPACE):
    # Imported here, not at module scope: geometry pulls in scipy (~0.4s)
    # and io pulls in nibabel (~0.3s).
    from cardio_form import geometry
    from cardio_form import io as cf_io
    from pycemrg.data import LabelManager
    label_manager = LabelManager(label_space_path(label_space))
    logger.info(f"User requested to merge labels: {user_labels_to_keep}")
    try:
        # Use the label manager to translate the user's flexible input
        # into a clean, sorted list of integer labels.
        labels_to_merge = label_manager.get_values_from_names(user_labels_to_keep)
        logger.info(f"Translated to integer labels: {labels_to_merge}")

    except KeyError as e:
        logger.error(f"FATAL ERROR: {e}")
        return
    except FileNotFoundError as e:
        logger.error(f"FATAL ERROR: Could not find labels.yaml. {e}")
        return

    try:
        # Orchestrate: load -> pure transform -> save.
        data, affine, header = cf_io.load_label_map(input_path)
        merged = geometry.merge_labels(data, labels_to_merge, value_after_merge)
        cf_io.save_nifti(merged, affine, header, output_path)
        logger.info(f"Merged segmentation saved to: {output_path}")
        logger.info("\n--- Merging complete! ---")
    except FileNotFoundError as e:
        logger.error(f"FATAL ERROR: Input file not found. {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")


def main(args):
    """
    CLI Wrapper. Only handles Argument Parsing.
    """
    try:
        run_merge_labels_job(
            input_path=args.input,
            output_path=args.output,
            user_labels_to_keep=args.labels,
            value_after_merge=args.value_after_merge,
            label_space=args.label_space
        )
    except Exception:
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merge a NIfTI segmentation file to keep only specified labels. "
                    "Labels can be specified by name, integer value, or group name (from labels.yaml).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("-i", "--input", required=True, 
                        help="Path to the input segmentation NIfTI file.")
    parser.add_argument("-o", "--output", required=True, 
                        help="Path for the new, Mergeed output NIfTI file.")
    parser.add_argument("-l", "--labels", required=True, nargs='+', type=str,
                        help="A space-separated list of labels to keep. "
                             "Can be names (LV_myo), groups (ventricles), or numbers (5). "
                             "Example: --labels ventricles LA_bp 7")
    parser.add_argument("-v", "--value-after-merge", type=int, default=None,
                        help="The label value to assign to the merged labels ")
    
    parser.add_argument("--label-space", default=DEFAULT_LABEL_SPACE, choices=sorted(LABEL_SPACES),
                        help="Which label space the input uses.")

    args = parser.parse_args()
    main(args)