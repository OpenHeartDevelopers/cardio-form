# in scripts/utils/filter_labels.py

import os
import sys
import argparse

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from cardio_form import geometry
from cardio_form.labels import default_label_manager # Import the default manager instance
from cardio_form.utils import configure_logging
logger = configure_logging('ScriptFilterLabels')

def main(args):

    logger.info(f"User requested to keep labels: {args.labels}")
    try:
        # Use the label manager to translate the user's flexible input
        # into a clean, sorted list of integer labels.
        labels_to_keep = default_label_manager.get_values_from_names(args.labels)
        logger.info(f"Translated to integer labels: {labels_to_keep}")

    except KeyError as e:
        logger.error(f"FATAL ERROR: {e}")
        return
    except FileNotFoundError as e:
        logger.error(f"FATAL ERROR: Could not find labels.yaml. {e}")
        return

    try:
        # Call the geometry engine function with the clean integer list
        geometry.filter_labels(
            input_path=args.input,
            output_path=args.output,
            labels_to_keep=labels_to_keep
        )
        logger.info("\n--- Filtering complete! ---")
    except FileNotFoundError as e:
        logger.error(f"FATAL ERROR: Input file not found. {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Filter a NIfTI segmentation file to keep only specified labels. "
                    "Labels can be specified by name, integer value, or group name (from labels.yaml).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("-i", "--input", required=True, 
                        help="Path to the input segmentation NIfTI file.")
    parser.add_argument("-o", "--output", required=True, 
                        help="Path for the new, filtered output NIfTI file.")
    parser.add_argument("-l", "--labels", required=True, nargs='+', type=str,
                        help="A space-separated list of labels to keep. "
                             "Can be names (LV_myo), groups (ventricles), or numbers (5). "
                             "Example: --labels ventricles LA_bp 7")
    
    args = parser.parse_args()
    main(args)