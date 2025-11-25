import os
import argparse 

from cardio_form.pipeline import CHOICES_VIEW_TYPE
from cardio_form.utils import configure_logging
logger = configure_logging('Docker')

# ML model methods
from scripts.run_reconstruct_3d import run_reconstruction_job
from scripts.run_segmentation import run_segmentation_job
from scripts.run_full_pipeline import run_full_pipeline_job
from scripts.run_la_reconstruction import run_la_reconstruction_job

# Other utilities
from scripts.utils.filter_labels import run_filter_labels_job
from scripts.utils.merge_labels import run_merge_labels_job

def main(args):
    """
    The Main Switchboard.
    It unpacks the 'args' object and calls the specific job functions.
    """
    mode = args.mode
    
    if mode == 'reconstruct':
        run_reconstruction_job(
            sax_path=args.sax_file,
            ch2_path=args.ch2_file,
            ch4_path=args.ch4_file,
            output_dir=args.output_dir,
            output_prefix=args.output_prefix,
            device=args.device
        )
    elif mode == 'segment':
        run_segmentation_job(
            input_path=args.input,
            output_dir=args.output_dir,
            view_type=args.view_type,
            output_prefix=args.output_prefix,
            device=args.device
        )
    elif mode == 'full_pipeline':
        run_full_pipeline_job(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            output_prefix=args.output_prefix,
            device=args.device
        )

    elif mode == 'reconstruct_la':
        run_la_reconstruction_job(
            ch2_file=args.ch2_file,
            ch4_file=args.ch4_file,
            output_dir=args.output_dir,
            output_prefix=args.output_prefix,
            device=args.device
        )
    elif mode == 'labels': 
        if args.action == 'filter':
            run_filter_labels_job(
                input_path=args.input,
                output_path=args.output,
                user_labels_to_keep=args.labels
            )
        elif args.action == 'merge':
            run_merge_labels_job(
                input_path=args.input,
                output_path=args.output,
                user_labels_to_keep=args.labels,
                value_after_merge=args.value_after_merge
            )
    


if __name__ == "__main__":

    ml_parent_parser = argparse.ArgumentParser(add_help=False)
    
    # Common Output Args
    ml_parent_parser.add_argument("-o", "--output-dir", required=True, help="Root directory to save outputs.")
    ml_parent_parser.add_argument("-p", "--output-prefix", required=True, help="Prefix for output filenames.")
    
    # Common Config Args
    ml_parent_parser.add_argument("--device", default="cpu", choices=['auto', 'cpu', 'cuda'], help="Device to run the model on.")
    
    parser = argparse.ArgumentParser( description="Docker Entrypoint for CardioForm", formatter_class=argparse.ArgumentDefaultsHelpFormatter )

    subparsers = parser.add_subparsers(dest='mode', required=True)
    
    # --- Parser for the reconstruction mode ---
    reconstruct_parser = subparsers.add_parser('reconstruct', parents=[ml_parent_parser], aliases=['reconstrunct_3d', '3d'], help='Run 3D reconstruction from 2D segmentations.')
    reconstruct_parser.add_argument("-sax", "--sax-file", required=True, help="Path to SAX NIfTI.")
    reconstruct_parser.add_argument("-ch2", "--ch2-file", required=True, help="Path to 2CH NIfTI.")
    reconstruct_parser.add_argument("-ch4", "--ch4-file", required=True, help="Path to 4CH NIfTI.")

    # --- Parser for the segmentation mode ---
    segment_parser = subparsers.add_parser('segment', parents=[ml_parent_parser], help='Run 2D segmentation on a cardiac MRI NIfTI file.')
    segment_parser.add_argument("--input", required=True, help="Path to the input NIfTI file.")
    segment_parser.add_argument("--view-type", required=True, choices=CHOICES_VIEW_TYPE, help="The type of cardiac view to segment.")

    # --- Parser for full_pipeline mode ---
    full_pipeline_parser = subparsers.add_parser('full_pipeline', parents=[ml_parent_parser], aliases=['full'], help='Run the full CardioForm pipeline: 2D Segmentation -> 3D Reconstruction.')
    full_pipeline_parser.add_argument("--input-dir", required=True,  help="Path to the input directory containing the raw CINE MRI images.")

    # --- Parser for LA reconstruction mode ---
    la_reconstruct_parser = subparsers.add_parser('reconstruct_la', aliases=['la_3d'], help='Run LA 3D reconstruction from 2D segmentations.')
    la_reconstruct_parser.add_argument("-ch2", "--ch2-file", required=True, help="Path to the LAX 2-chamber segmentation NIfTI file.")
    la_reconstruct_parser.add_argument("-ch4", "--ch4-file", required=True, help="Path to the LAX 4-chamber segmentation NIfTI file.")

    la_reconstruct_parser.add_argument("-o", "--output-dir", required=True, help="Directory to save all output files.")
    la_reconstruct_parser.add_argument("-p", "--output-prefix", required=True, help="Prefix for all output filenames (e.g., 'subject_001_cine').")
    
    la_reconstruct_parser.add_argument("--device", default="cpu", choices=['cpu', 'cuda'], help="Device to run the model on.")

    # --- Parser for label utilities mode ---
    labels_parser = subparsers.add_parser('labels', help='Utilities for filtering labels in segmentation files.')
    labels_parser.add_argument("action", choices=['filter', 'merge'], help="The label utility action to perform.")
    labels_parser.add_argument("-i", "--input", required=True,  help="Path to the input segmentation NIfTI file.")
    labels_parser.add_argument("-o", "--output", required=True,  help="Path for the new, filtered output NIfTI file.")
    labels_parser.add_argument("-l", "--labels", required=True, nargs='+', type=str, help="A space-separated list of labels to keep. "
                             "Can be names (LV_myo), groups (ventricles), or numbers (5). "
                             "Example: --labels ventricles LA_bp 7")
    labels_parser.add_argument("-v", "--value-after-merge", required=False, type=int, default=None, help="The label value to assign to the merged labels ")
    
    args = parser.parse_args()
    main(args)    