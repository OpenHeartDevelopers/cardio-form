"""NIfTI input/output helpers for CardioForm.

Centralises file loading and saving so the geometry layer can stay pure
(operating on arrays and affines, never on file paths).
"""

import nibabel as nib
import numpy as np


def load_nifti(path) -> nib.Nifti1Image:
    """Load a NIfTI file and return the nibabel image object."""
    return nib.load(path)


def load_data_and_affine(path) -> tuple:
    """Load a NIfTI file, returning ``(data_array, affine)``."""
    img = nib.load(path)
    return img.get_fdata(), img.affine


def load_label_map(path) -> tuple:
    """Load a segmentation as an integer label map.

    Returns ``(data, affine, header)`` where ``data`` is ``int16``. Float voxel
    values are rounded to the nearest integer first, since segmentations are
    sometimes stored as floats.
    """
    img = nib.load(path)
    data = np.round(img.get_fdata()).astype(np.int16)
    return data, img.affine, img.header


def save_nifti(data, affine, header, path) -> None:
    """Save an array as a NIfTI file, preserving the given affine and header."""
    nib.save(nib.Nifti1Image(data, affine, header), path)
