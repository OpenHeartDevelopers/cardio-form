import sys
import argparse

from cardio_form.config import DEFAULT_LABEL_SPACE, LABEL_SPACES, label_space_path
from cardio_form.utils import configure_logging
logger = configure_logging('ScriptRelabel')


def _resolve_label(token: str, label_manager) -> int:
    """Resolve a single label token (integer string or label name) to an int."""
    token = token.strip()
    if token.isdigit():
        return int(token)
    return label_manager.get_value(token)


def _parse_mapping(pairs: list, label_manager) -> dict:
    """Parse ``['OLD:NEW', ...]`` into a ``{int: int}`` mapping.

    Each side may be an integer or a label name from the active label space.
    Groups are not accepted here because remapping is one-to-one.
    """
    mapping = {}
    for pair in pairs:
        if ':' not in pair:
            raise ValueError(f"Invalid mapping '{pair}'. Expected OLD:NEW (e.g. 4:1 or MYO_septum:LV_myo).")
        old_str, new_str = pair.split(':', 1)
        mapping[_resolve_label(old_str, label_manager)] = _resolve_label(new_str, label_manager)
    return mapping


def run_relabel_job(input_path, output_path, mapping_pairs, label_space=DEFAULT_LABEL_SPACE):
    """
    Pure Python function to remap labels in a segmentation file.
    Accepts standard types, not argparse objects.
    """
    # Imported here, not at module scope: geometry pulls in scipy (~0.4s)
    # and io pulls in nibabel (~0.3s).
    from cardio_form import geometry
    from cardio_form import io as cf_io
    from pycemrg.data import LabelManager

    label_manager = LabelManager(label_space_path(label_space))
    logger.info(f"User requested label remapping: {mapping_pairs} (space: {label_space})")
    try:
        # Translate the user's flexible OLD:NEW pairs into a clean integer mapping.
        mapping = _parse_mapping(mapping_pairs, label_manager)
        logger.info(f"Resolved integer mapping: {mapping}")
    except KeyError as e:
        logger.error(f"FATAL ERROR: {e}")
        return
    except ValueError as e:
        logger.error(f"FATAL ERROR: {e}")
        return

    try:
        # Orchestrate: load -> pure transform -> save.
        data, affine, header = cf_io.load_label_map(input_path)
        remapped = geometry.remap_labels(data, mapping)
        cf_io.save_nifti(remapped, affine, header, output_path)
        logger.info(f"Remapped segmentation saved to: {output_path}")
        logger.info("\n--- Relabeling complete! ---")
    except FileNotFoundError as e:
        logger.error(f"FATAL ERROR: Input file not found. {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")


def main(args):
    """
    CLI Wrapper. Only handles Argument Parsing.
    """
    try:
        run_relabel_job(
            input_path=args.input,
            output_path=args.output,
            mapping_pairs=args.map,
            label_space=args.label_space,
        )
    except Exception:
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Remap labels in a NIfTI segmentation file using OLD:NEW pairs. "
                    "Each side may be an integer or a label name from labels.yaml.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("-i", "--input", required=True,
                        help="Path to the input segmentation NIfTI file.")
    parser.add_argument("-o", "--output", required=True,
                        help="Path for the new, remapped output NIfTI file.")
    parser.add_argument("-m", "--map", required=True, nargs='+', type=str,
                        help="A space-separated list of OLD:NEW pairs. "
                             "Each side can be a name (LV_myo) or a number (5). "
                             "Example: --map MYO_septum:LV_myo 7:0")

    parser.add_argument("--label-space", default=DEFAULT_LABEL_SPACE, choices=sorted(LABEL_SPACES),
                        help="Which label space the input uses.")

    args = parser.parse_args()
    main(args)
