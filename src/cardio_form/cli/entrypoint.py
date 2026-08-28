import argparse

from cardio_form.config import CHOICES_VIEW_TYPE, DEFAULT_LABEL_SPACE, LABEL_SPACES
from cardio_form.utils import configure_logging
logger = configure_logging('Docker')

# The job functions are imported lazily inside run(), at the point of dispatch.
# Importing them here would pull torch and nnunetv2 into every invocation,
# including `cardioform --help`.

# Sub-command aliases -> canonical mode name. Must stay in sync with the
# `aliases=` lists passed to add_parser() in main().
MODE_ALIASES = {
    '3d': 'reconstruct',
    'la_3d': 'reconstruct_la',
    'full': 'full_pipeline',
    'left-complete': 'left_complete',
}

def run(args):
    """
    The Main Switchboard.
    It unpacks the 'args' object and calls the specific job functions.
    """
    # argparse sets dest='mode' to the alias the user actually typed, not the
    # canonical parser name, so resolve it back before dispatching.
    mode = MODE_ALIASES.get(args.mode, args.mode)
    logger.info(f'Attempting mode: {mode}')
    
    if mode == 'reconstruct':
        from cardio_form.cli.reconstruct_3d import run_reconstruction_job
        run_reconstruction_job(
            sax_path=args.sax_file,
            ch2_path=args.ch2_file,
            ch4_path=args.ch4_file,
            output_dir=args.output_dir,
            output_prefix=args.output_prefix,
            device=args.device
        )
    elif mode == 'segment':
        from cardio_form.cli.segment import run_segmentation_job
        run_segmentation_job(
            input_path=args.input,
            output_dir=args.output_dir,
            view_type=args.view_type,
            output_prefix=args.output_prefix,
            device=args.device
        )
    elif mode == 'full_pipeline':
        from cardio_form.cli.full_pipeline import run_full_pipeline_job
        run_full_pipeline_job(
            input_dir=args.input_dir,
            sax_path=args.sax_file,
            ch2_path=args.ch2_file,
            ch4_path=args.ch4_file,
            output_dir=args.output_dir,
            output_prefix=args.output_prefix,
            device=args.device
        )

    elif mode == 'reconstruct_la':
        from cardio_form.cli.reconstruct_la import run_la_reconstruction_job
        run_la_reconstruction_job(
            ch2_file=args.ch2_file,
            ch4_file=args.ch4_file,
            output_dir=args.output_dir,
            output_prefix=args.output_prefix,
            device=args.device
        )
    elif mode == 'left_complete':
        from cardio_form.cli.left_complete import run_left_complete_job, SELECTION_GROUPS
        selected = [group for flag, group in SELECTION_GROUPS.items()
                    if getattr(args, flag, False)]
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
    elif mode == 'labels':
        if args.action == 'filter':
            from cardio_form.cli.filter_labels import run_filter_labels_job
            if not args.labels:
                raise SystemExit("Error: 'labels filter' requires -l/--labels")
            run_filter_labels_job(
                input_path=args.input,
                output_path=args.output,
                user_labels_to_keep=args.labels,
                label_space=args.label_space
            )
        elif args.action == 'merge':
            from cardio_form.cli.merge_labels import run_merge_labels_job
            if not args.labels:
                raise SystemExit("Error: 'labels merge' requires -l/--labels")
            run_merge_labels_job(
                input_path=args.input,
                output_path=args.output,
                user_labels_to_keep=args.labels,
                value_after_merge=args.value_after_merge,
                label_space=args.label_space
            )
        elif args.action == 'relabel':
            from cardio_form.cli.relabel import run_relabel_job
            if not args.map:
                raise SystemExit("Error: 'labels relabel' requires -m/--map")
            run_relabel_job(
                input_path=args.input,
                output_path=args.output,
                mapping_pairs=args.map,
                label_space=args.label_space
            )
    else:
        raise SystemExit(f"Error: unhandled mode '{args.mode}'.")



def main():
    ml_parent_parser = argparse.ArgumentParser(add_help=False)
    
    # Common Output Args
    ml_parent_parser.add_argument("-o", "--output-dir", required=True, help="Root directory to save outputs.")
    ml_parent_parser.add_argument("-p", "--output-prefix", required=True, help="Prefix for output filenames.")
    
    # Common Config Args
    ml_parent_parser.add_argument("--device", default="cpu", choices=['auto', 'cpu', 'cuda'], help="Device to run the model on.")
    
    parser = argparse.ArgumentParser( description="Docker Entrypoint for CardioForm", formatter_class=argparse.ArgumentDefaultsHelpFormatter )

    subparsers = parser.add_subparsers(dest='mode', required=True)
    
    # --- Parser for the reconstruction mode ---
    reconstruct_parser = subparsers.add_parser('reconstruct', parents=[ml_parent_parser], aliases=['3d'], help='Run 3D reconstruction from 2D segmentations.')
    reconstruct_parser.add_argument("-sax", "--sax-file", required=True, help="Path to SAX NIfTI.")
    reconstruct_parser.add_argument("-ch2", "--ch2-file", required=True, help="Path to 2CH NIfTI.")
    reconstruct_parser.add_argument("-ch4", "--ch4-file", required=True, help="Path to 4CH NIfTI.")

    # --- Parser for the segmentation mode ---
    segment_parser = subparsers.add_parser('segment', parents=[ml_parent_parser], help='Run 2D segmentation on a cardiac MRI NIfTI file.')
    segment_parser.add_argument("--input", required=True, help="Path to the input NIfTI file.")
    segment_parser.add_argument("--view-type", required=True, choices=CHOICES_VIEW_TYPE, help="The type of cardiac view to segment.")

    # --- Parser for full_pipeline mode ---
    full_pipeline_parser = subparsers.add_parser('full_pipeline', parents=[ml_parent_parser], aliases=['full'], help='Run the full CardioForm pipeline: 2D Segmentation -> 3D Reconstruction.')
    full_pipeline_parser.add_argument("--input-dir", required=False, default=None, help="Directory of raw CINE MRI images; views are auto-discovered by glob. Optional if the explicit view flags are given.")
    full_pipeline_parser.add_argument("-sax", "--sax-file", default=None, help="Path to SAX NIfTI (overrides discovery from --input-dir).")
    full_pipeline_parser.add_argument("-ch2", "--ch2-file", default=None, help="Path to 2CH NIfTI (overrides discovery from --input-dir).")
    full_pipeline_parser.add_argument("-ch4", "--ch4-file", default=None, help="Path to 4CH NIfTI (overrides discovery from --input-dir).")

    # --- Parser for LA reconstruction mode ---
    la_reconstruct_parser = subparsers.add_parser('reconstruct_la', aliases=['la_3d'], help='Run LA 3D reconstruction from 2D segmentations.')
    la_reconstruct_parser.add_argument("-ch2", "--ch2-file", required=True, help="Path to the LAX 2-chamber segmentation NIfTI file.")
    la_reconstruct_parser.add_argument("-ch4", "--ch4-file", required=True, help="Path to the LAX 4-chamber segmentation NIfTI file.")

    la_reconstruct_parser.add_argument("-o", "--output-dir", required=True, help="Directory to save all output files.")
    la_reconstruct_parser.add_argument("-p", "--output-prefix", required=True, help="Prefix for all output filenames (e.g., 'subject_001_cine').")
    
    la_reconstruct_parser.add_argument("--device", default="cpu", choices=['cpu', 'cuda'], help="Device to run the model on.")

    # --- Parser for left-heart completion mode ---
    left_complete_parser = subparsers.add_parser('left_complete', aliases=['left-complete'], help="Enhance a whole-heart segmentation with the LA network's left-side output.")
    left_complete_parser.add_argument("-la", "--la-file", default=None, help="LA 3D segmentation NIfTI (overrides -ch2/-ch4).")
    left_complete_parser.add_argument("-ch2", "--ch2-file", default=None, help="LAX 2-chamber segmentation NIfTI; used when -la/--la-file is absent.")
    left_complete_parser.add_argument("-ch4", "--ch4-file", default=None, help="LAX 4-chamber segmentation NIfTI; used when -la/--la-file is absent.")
    left_complete_parser.add_argument("-whs", "--whs-file", required=True, help="Whole-heart segmentation NIfTI to enhance.")
    left_complete_parser.add_argument("-o", "--output-dir", required=True, help="Directory to save all output files.")
    left_complete_parser.add_argument("-p", "--output-prefix", required=True, help="Prefix for all output filenames.")
    left_complete_parser.add_argument("--device", default="cpu", choices=['cpu', 'cuda'], help="Device, used only on the -ch2/-ch4 path.")
    left_complete_parser.add_argument("--include-la", dest="include_la", action="store_true", help="Merge the LA body.")
    left_complete_parser.add_argument("--include-lv", dest="include_lv", action="store_true", help="Merge the LV completion.")
    left_complete_parser.add_argument("--include-veins", dest="include_veins", action="store_true", help="Merge the pulmonary vein classes.")

    # --- Parser for label utilities mode ---
    labels_parser = subparsers.add_parser('labels', help='Utilities for editing labels in segmentation files (filter / merge / relabel).')
    labels_parser.add_argument("action", choices=['filter', 'merge', 'relabel'], help="The label utility action to perform.")
    labels_parser.add_argument("-i", "--input", required=True,  help="Path to the input segmentation NIfTI file.")
    labels_parser.add_argument("-o", "--output", required=True,  help="Path for the new output NIfTI file.")
    labels_parser.add_argument("-l", "--labels", required=False, nargs='+', type=str, help="(filter/merge) A space-separated list of labels. "
                             "Can be names (LV_myo), groups (ventricles), or numbers (5). "
                             "Example: --labels ventricles LA_bp 7")
    labels_parser.add_argument("--label-space", default=DEFAULT_LABEL_SPACE, choices=sorted(LABEL_SPACES),
                             help="Which label space the input uses. The 2D segmentation outputs "
                                  "do NOT use the whole-heart convention.")
    labels_parser.add_argument("-v", "--value-after-merge", required=False, type=int, default=None, help="(merge) The label value to assign to the merged labels.")
    labels_parser.add_argument("-m", "--map", required=False, nargs='+', type=str, help="(relabel) Space-separated OLD:NEW pairs. "
                             "Each side can be a name (MYO_septum) or a number. "
                             "Example: --map MYO_septum:LV_myo 7:0")
    
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
