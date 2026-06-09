#!/usr/bin/env python
"""Extract a single time frame from a 4D CINE NIfTI as a clean 2D/3D volume.

The CardioForm 2D segmentation models expect a single-channel image. Raw CINE
series are 4D (``X, Y, Z, T``); feeding the whole stack to nnUNet makes it read
the time axis as channels and crashes. This helper pulls one phase out so it can
be segmented.

Run after `pip install -e .` (uses the installed ``cardio_form`` package).

Examples:
    python helper_scripts/extract_frame.py -i 2CH_FIESTA_BH.nii.gz
    python helper_scripts/extract_frame.py -i cine.nii.gz -f 7 -o cine_ed.nii.gz
"""

import argparse
import sys

from cardio_form import io as cf_io


def extract_frame(input_path: str, output_path: str, frame: int) -> str:
    """Write frame ``frame`` of a 4D NIfTI to ``output_path``. Returns the path."""
    img = cf_io.load_nifti(input_path)

    if img.ndim != 4:
        raise ValueError(
            f"Input is {img.ndim}D with shape {img.shape}; expected a 4D CINE "
            f"(X, Y, Z, T). Nothing to extract."
        )

    n_frames = img.shape[3]
    if not 0 <= frame < n_frames:
        raise IndexError(
            f"Frame {frame} is out of range; this series has {n_frames} frames "
            f"(valid: 0..{n_frames - 1})."
        )

    data = img.dataobj[..., frame]  # (X, Y, Z) — avoids loading the full 4D array
    cf_io.save_nifti(data, img.affine, img.header, output_path)
    print(f"Wrote frame {frame} of {n_frames} -> {output_path} (shape {data.shape})")
    return output_path


def _default_output(input_path: str, frame: int) -> str:
    """Insert ``_frame{N}`` before the .nii / .nii.gz extension."""
    for ext in (".nii.gz", ".nii"):
        if input_path.endswith(ext):
            return f"{input_path[: -len(ext)]}_frame{frame}{ext}"
    raise ValueError(f"Input does not look like a NIfTI file: {input_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract a single time frame from a 4D CINE NIfTI."
    )
    parser.add_argument("-i", "--input", required=True, help="Path to the 4D CINE NIfTI.")
    parser.add_argument(
        "-f", "--frame", type=int, default=0,
        help="Frame (cardiac phase) index to extract. Default: 0.",
    )
    parser.add_argument(
        "-o", "--output", default=None,
        help="Output path. Defaults to '<input>_frame<N>.nii.gz' next to the input.",
    )
    args = parser.parse_args()

    output_path = args.output or _default_output(args.input, args.frame)

    try:
        extract_frame(args.input, output_path, args.frame)
    except (ValueError, IndexError, FileNotFoundError) as e:
        sys.exit(f"Error: {e}")


if __name__ == "__main__":
    main()
