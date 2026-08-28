#!/usr/bin/env python
"""Check SAX <-> LAX orientation consistency for a CardioForm case.

Two complementary checks:

  Mode A (geometry):   --sax / --ch2 / --ch4
      Reports per-view affine geometry (axcodes, determinant/handedness, voxel
      spacing, slice-stacking direction) and the angle between the SAX stacking
      axis and each LAX plane. Catches header-level problems: a left-handed
      affine, a SAX<->LAX handedness mismatch, or views that are not in the
      expected SAX/LAX arrangement.

      Limitation: a SAX whose (data, affine) pair is *self-consistent* but whose
      slice order is reversed relative to the LAX will still look clean here.
      Mode B tests that directly.

  Mode B (content):    --sax-seg / --sax-bp
      Compares the SAX segmentation (`_2D_seg_sax.nii.gz`) against the SAX
      backprojection (`_intermediate_qc_sax_backprojected.nii.gz`).

      NOTE: the backprojection is only written when the reconstruction was run
      with `-qc` / `--quality-control`. Re-run `cardioform reconstruct -qc ...`
      if the file is missing. Both share
      the same grid, but the backprojection's content comes from the
      LAX-anchored 3D model. If `seg[..., k]` matches `bp[..., N-1-k]` better
      than `bp[..., k]`, the SAX apex<->base slice order is reversed relative to
      the LAX -- the reported symptom.

Run after `pip install -e .` (uses the installed ``cardio_form`` package).

Examples:
    # Mode A only (the three raw inputs):
    python helper_scripts/check_orientation.py \
        --sax sax.nii.gz --ch2 2ch.nii.gz --ch4 4ch.nii.gz

    # Mode B only (pipeline outputs):
    python helper_scripts/check_orientation.py \
        --sax-seg out/sub01_2D_seg_sax.nii.gz \
        --sax-bp  out/sub01_intermediate_qc_sax_backprojected.nii.gz

    # Both:
    python helper_scripts/check_orientation.py \
        --sax sax.nii.gz --ch2 2ch.nii.gz --ch4 4ch.nii.gz \
        --sax-seg out/sub01_2D_seg_sax.nii.gz \
        --sax-bp  out/sub01_intermediate_qc_sax_backprojected.nii.gz

Exit code is non-zero if any check raises a flag or returns a non-consistent
verdict, so it can be used in scripts.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass

import numpy as np
import nibabel as nib

from cardio_form import io as cf_io


# --- tuning constants (named, not magic) ---
PLANE_ANGLE_TOL_DEG = 30.0  # SAX stacking axis should lie ~in the LAX plane (expect ~0 deg)
MIN_OVERLAP = 0.10          # below this, slice-ordering agreement is inconclusive


# ============================================================================
# Logic layer (pure, stateless: operates on affines / arrays, returns contracts)
# ============================================================================

@dataclass(frozen=True)
class ViewGeometry:
    """Geometric description of one acquisition, derived from its affine."""
    name: str
    shape: tuple
    axcodes: tuple
    spacing: tuple
    slice_dir: tuple   # normalized 3rd affine column (through-plane direction)
    determinant: float

    @property
    def handedness(self) -> str:
        return "left" if self.determinant < 0 else "right"


@dataclass(frozen=True)
class SaxLaxRelation:
    """Relationship between the SAX stacking axis and one LAX plane."""
    lax_name: str
    sax_axis_to_lax_plane_deg: float

    @property
    def flagged(self) -> bool:
        return self.sax_axis_to_lax_plane_deg > PLANE_ANGLE_TOL_DEG


@dataclass(frozen=True)
class SliceOrderingCheck:
    """Forward vs reversed slice-agreement between SAX seg and backprojection."""
    n_slices: int
    forward_score: float
    reversed_score: float

    @property
    def verdict(self) -> str:
        if max(self.forward_score, self.reversed_score) < MIN_OVERLAP:
            return "inconclusive"
        return "consistent" if self.forward_score >= self.reversed_score else "reversed"


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def _acute_angle_deg(u: np.ndarray, v: np.ndarray) -> float:
    """Unsigned acute angle (0..90 deg) between two directions."""
    c = abs(float(np.dot(_unit(u), _unit(v))))
    return float(np.degrees(np.arccos(np.clip(c, 0.0, 1.0))))


def describe_view(name: str, affine: np.ndarray, shape: tuple) -> ViewGeometry:
    """Summarise a view's geometry from its affine and shape. Pure transform."""
    rot_scale = np.asarray(affine)[:3, :3]
    spacing = tuple(round(float(np.linalg.norm(rot_scale[:, i])), 3) for i in range(3))
    slice_dir = tuple(round(float(x), 3) for x in _unit(rot_scale[:, 2]))
    determinant = float(np.linalg.det(rot_scale))
    axcodes = nib.aff2axcodes(affine)
    return ViewGeometry(
        name=name,
        shape=tuple(int(s) for s in shape[:3]),
        axcodes=axcodes,
        spacing=spacing,
        slice_dir=slice_dir,
        determinant=determinant,
    )


def relate_sax_to_lax(sax: ViewGeometry, lax: ViewGeometry) -> SaxLaxRelation:
    """Angle between the SAX stacking axis and the LAX plane (0 deg = axis in plane)."""
    angle_between_normals = _acute_angle_deg(np.array(sax.slice_dir), np.array(lax.slice_dir))
    sax_axis_to_lax_plane = 90.0 - angle_between_normals
    return SaxLaxRelation(lax_name=lax.name, sax_axis_to_lax_plane_deg=round(sax_axis_to_lax_plane, 1))


def _foreground_iou(a: np.ndarray, b: np.ndarray):
    """IoU of the foreground (label > 0) masks, or None if both are empty."""
    fa, fb = a > 0, b > 0
    union = int(np.logical_or(fa, fb).sum())
    if union == 0:
        return None
    return float(np.logical_and(fa, fb).sum()) / float(union)


def compare_slice_ordering(seg: np.ndarray, bp: np.ndarray) -> SliceOrderingCheck:
    """Compare SAX seg vs backprojection under forward and reversed slice pairing."""
    if seg.shape != bp.shape:
        raise ValueError(
            f"SAX seg shape {seg.shape} != backprojection shape {bp.shape}; "
            f"they must share the same grid (same case, same SAX geometry)."
        )
    n = seg.shape[-1]
    forward, reverse = [], []
    for k in range(n):
        f = _foreground_iou(seg[..., k], bp[..., k])
        r = _foreground_iou(seg[..., k], bp[..., n - 1 - k])
        if f is not None:
            forward.append(f)
        if r is not None:
            reverse.append(r)
    forward_score = float(np.mean(forward)) if forward else 0.0
    reversed_score = float(np.mean(reverse)) if reverse else 0.0
    return SliceOrderingCheck(
        n_slices=n,
        forward_score=round(forward_score, 3),
        reversed_score=round(reversed_score, 3),
    )


# ============================================================================
# Orchestration layer (file I/O, printing, exit code)
# ============================================================================

def _run_geometry(sax_path: str, ch2_path: str, ch4_path: str):
    sax_img = cf_io.load_nifti(sax_path)
    ch2_img = cf_io.load_nifti(ch2_path)
    ch4_img = cf_io.load_nifti(ch4_path)
    sax = describe_view("sax", sax_img.affine, sax_img.shape)
    ch2 = describe_view("2ch", ch2_img.affine, ch2_img.shape)
    ch4 = describe_view("4ch", ch4_img.affine, ch4_img.shape)
    views = [sax, ch2, ch4]
    relations = [relate_sax_to_lax(sax, ch2), relate_sax_to_lax(sax, ch4)]
    return views, relations


def _print_geometry(views, relations) -> bool:
    """Print the mode A report. Returns True if anything was flagged."""
    flagged = False
    print("=== Per-view geometry (mode A) ===")
    for v in views:
        print(
            f"  {v.name:4s} shape={v.shape}  axcodes={''.join(v.axcodes)}  "
            f"det={v.determinant:+.3f} ({v.handedness}-handed)  "
            f"spacing={v.spacing}  slice_dir={v.slice_dir}"
        )

    print("\n=== Cross-view checks ===")
    sax = views[0]
    for v in views:
        if v.determinant < 0:
            print(f"  [FLAG] {v.name} affine is left-handed (negative determinant) "
                  f"-- mirrors reconstruct_3d.py:217 handedness warning.")
            flagged = True
    for v in views[1:]:
        if (v.determinant < 0) != (sax.determinant < 0):
            print(f"  [FLAG] handedness mismatch: sax ({sax.handedness}) vs {v.name} ({v.handedness}).")
            flagged = True
    for rel in relations:
        status = "FLAG" if rel.flagged else "ok"
        print(f"  [{status}] SAX stacking axis is {rel.sax_axis_to_lax_plane_deg:.1f} deg "
              f"from the {rel.lax_name} plane (expect ~0 deg; <{PLANE_ANGLE_TOL_DEG:.0f} ok).")
        flagged = flagged or rel.flagged

    if len({v.axcodes for v in views}) > 1:
        print("  [note] the three views do not share orientation axcodes. This is normal for "
              "oblique cardiac acquisitions, but if one series was reoriented to canonical "
              "(e.g. RAS) without the others, that can cause a SAX<->LAX mismatch.")

    print("\n  NOTE: a self-consistent but slice-reversed SAX (data order reversed vs its own\n"
          "  affine) can still look clean above. Use --sax-seg/--sax-bp (mode B) to test that\n"
          "  directly.")
    return flagged


def _run_content(sax_seg_path: str, sax_bp_path: str) -> SliceOrderingCheck:
    seg, _, _ = cf_io.load_label_map(sax_seg_path)
    bp, _, _ = cf_io.load_label_map(sax_bp_path)
    return compare_slice_ordering(seg, bp)


def _print_content(check: SliceOrderingCheck) -> bool:
    """Print the mode B report. Returns True if the verdict is not 'consistent'."""
    print("\n=== SAX slice-ordering verification (mode B: seg vs backprojection) ===")
    print(f"  slices = {check.n_slices}")
    print(f"  forward  agreement (seg[k]   vs bp[k])     = {check.forward_score:.3f}")
    print(f"  reversed agreement (seg[k]   vs bp[N-1-k]) = {check.reversed_score:.3f}")

    verdict = check.verdict
    if verdict == "consistent":
        print("  [VERDICT] CONSISTENT -- SAX slice order agrees with the LAX-anchored model.")
        return False
    if verdict == "reversed":
        print("  [VERDICT] REVERSED -- SAX apex<->base slice order is flipped vs the LAX model.\n"
              "            This is the reported symptom: the SAX input's slice ordering is\n"
              "            reversed relative to its affine. Fix upstream (DICOM->NIfTI / reorient).")
        return True
    print("  [VERDICT] INCONCLUSIVE -- too little foreground overlap to decide.\n"
          "            Check that seg and backprojection are from the same case.")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Check SAX<->LAX orientation consistency for a CardioForm case.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--sax", help="SAX input/image NIfTI (mode A).")
    parser.add_argument("--ch2", help="2CH input/image NIfTI (mode A).")
    parser.add_argument("--ch4", help="4CH input/image NIfTI (mode A).")
    parser.add_argument("--sax-seg", dest="sax_seg",
                        help="SAX segmentation '_2D_seg_sax.nii.gz' (mode B).")
    parser.add_argument("--sax-bp", dest="sax_bp",
                        help="SAX backprojection '_intermediate_qc_sax_backprojected.nii.gz' (mode B). Only written when reconstruction ran with -qc.")
    args = parser.parse_args()

    mode_a = all([args.sax, args.ch2, args.ch4])
    mode_b = all([args.sax_seg, args.sax_bp])
    if not (mode_a or mode_b):
        parser.error(
            "nothing to do. Provide all of --sax/--ch2/--ch4 (mode A) and/or both "
            "--sax-seg/--sax-bp (mode B)."
        )

    flagged = False
    try:
        if mode_a:
            views, relations = _run_geometry(args.sax, args.ch2, args.ch4)
            flagged = _print_geometry(views, relations) or flagged
        if mode_b:
            check = _run_content(args.sax_seg, args.sax_bp)
            flagged = _print_content(check) or flagged
    except (ValueError, IndexError, FileNotFoundError) as e:
        sys.exit(f"Error: {e}")

    sys.exit(1 if flagged else 0)


if __name__ == "__main__":
    main()
