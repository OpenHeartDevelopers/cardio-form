import sys
import argparse

from cardio_form.utils import configure_logging
logger = configure_logging('ScriptLeftComplete')

# Selection flags -> group names in left_labels.yaml.
SELECTION_GROUPS = {
    'include_la': 'la',
    'include_lv': 'lv',
    'include_veins': 'veins',
}


def build_merge_mapping(selected_groups=None):
    """Build the ``{left label: whole-heart label}`` integer mapping.

    Reads left_labels.yaml for the LA network's output space and its
    ``merge_into`` table, and labels.yaml for the whole-heart space.

    Args:
        selected_groups (list): Group names from left_labels.yaml to keep.
            ``None`` or empty keeps every label that has a merge target.

    Returns:
        dict: ``{left_label_int: whole_heart_label_int}``.
    """
    import yaml
    from pycemrg.data import LabelManager

    from cardio_form.config import config_path
    from cardio_form.labels import default_label_manager

    left_manifest = config_path('left_labels.yaml')
    left_manager = LabelManager(left_manifest)
    with open(left_manifest, 'r') as f:
        merge_into = yaml.safe_load(f).get('merge_into', {})

    if selected_groups:
        keep = set(left_manager.get_values_from_names(selected_groups))
    else:
        keep = None

    mapping = {}
    for left_name, whole_heart_name in merge_into.items():
        left_value = left_manager.get_value(left_name)
        if keep is not None and left_value not in keep:
            continue
        mapping[left_value] = default_label_manager.get_value(whole_heart_name)

    if not mapping:
        raise ValueError(
            "The selection flags left nothing to merge. "
            "Drop the --include-* flags to merge every structure."
        )

    return mapping


def run_left_complete_job(
    whs_file,
    output_dir,
    output_prefix,
    la_file=None,
    ch2_file=None,
    ch4_file=None,
    device='cpu',
    selected_groups=None,
):
    """
    Enhance a whole-heart segmentation with the LA network's left-side output.

    The LA volume is resampled onto the whole-heart grid and written only where
    the whole-heart map is background, so existing structure is never modified.

    Either ``la_file`` or both ``ch2_file`` and ``ch4_file`` must be given;
    ``la_file`` takes precedence.
    """
    # Imported here, not at module scope: pulls in the pipeline stack.
    from cardio_form.pipeline import CardioForm
    from cardio_form.output_managers import OutputManager

    try:
        outputs = OutputManager(output_dir=output_dir, output_prefix=output_prefix)

        if la_file is None:
            if not (ch2_file and ch4_file):
                raise ValueError(
                    "Provide -la/--la-file, or both -ch2 and -ch4 to reconstruct it first."
                )
            logger.info("No -la/--la-file given; running LA 3D reconstruction first.")
            from cardio_form.cli.reconstruct_la import run_la_reconstruction_job
            run_la_reconstruction_job(
                ch2_file=ch2_file,
                ch4_file=ch4_file,
                output_dir=output_dir,
                output_prefix=output_prefix,
                device=device,
            )
            la_file = outputs.get_path('la_prediction')
        else:
            logger.info(f"Using supplied LA reconstruction: {la_file}")


        output_path = CardioForm(device=device).left_complete(
            la_file=la_file,
            whs_file=whs_file,
            output_dir=output_dir,
            output_prefix=output_prefix,
            selected_groups=selected_groups,
        )

        logger.info(f"Left-complete segmentation saved to: {output_path}")
        logger.info("\n--- Left-heart completion finished successfully! ---")
        return output_path

    except Exception as e:
        logger.error(f"FATAL: Failed to run left-heart completion. Error: {e}")
        raise e


def main(args):
    """
    CLI Wrapper. Only handles Argument Parsing.
    """
    selected = [group for flag, group in SELECTION_GROUPS.items()
                if getattr(args, flag, False)]
    try:
        run_left_complete_job(
            whs_file=args.whs_file,
            output_dir=args.output_dir,
            output_prefix=args.output_prefix,
            la_file=args.la_file,
            ch2_file=args.ch2_file,
            ch4_file=args.ch4_file,
            device=args.device,
            selected_groups=selected,
        )
    except Exception:
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Enhance a whole-heart segmentation with the LA network's left-side output.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("-la", "--la-file", default=None, help="LA 3D segmentation NIfTI (overrides -ch2/-ch4).")
    parser.add_argument("-ch2", "--ch2-file", default=None, help="LAX 2-chamber segmentation NIfTI.")
    parser.add_argument("-ch4", "--ch4-file", default=None, help="LAX 4-chamber segmentation NIfTI.")
    parser.add_argument("-whs", "--whs-file", required=True, help="Whole-heart segmentation NIfTI to enhance.")
    parser.add_argument("-o", "--output-dir", required=True, help="Directory to save all output files.")
    parser.add_argument("-p", "--output-prefix", required=True, help="Prefix for all output filenames.")
    parser.add_argument("--device", default="cpu", choices=['cpu', 'cuda'], help="Device, used only on the -ch2/-ch4 path.")
    parser.add_argument("--include-la", dest="include_la", action="store_true", help="Merge the LA body.")
    parser.add_argument("--include-lv", dest="include_lv", action="store_true", help="Merge the LV completion.")
    parser.add_argument("--include-veins", dest="include_veins", action="store_true", help="Merge the pulmonary vein classes.")

    args = parser.parse_args()
    main(args)
