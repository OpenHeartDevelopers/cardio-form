"""CardioForm: whole-heart segmentation and 3D reconstruction from cardiac CINE MRI."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("cardio-form")
except PackageNotFoundError:  # package not installed (e.g. running from a raw checkout)
    __version__ = "0.0.0+unknown"
