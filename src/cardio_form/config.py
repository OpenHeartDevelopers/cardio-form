"""Resolution of CardioForm configuration files (label spaces / models.yaml).

The YAML manifests ship inside the package, at ``src/cardio_form/config_data/``,
so they are available to a plain ``pip install`` as well as to an editable
checkout and the Docker image. The location can be overridden explicitly with
the ``CARDIOFORM_CONFIG_DIR`` environment variable, following the project's
"explicit environment injection" rule.

``dev_root()`` is separate and deliberately different: it points at the repo
root, which is what the ``file://`` development weights in ``models.yaml`` are
written relative to.
"""

import os
from pathlib import Path

# .../src/cardio_form/config.py -> the packaged manifests sit beside this file.
_DEFAULT_CONFIG_DIR = Path(__file__).resolve().parent / "config_data"

# .../src/cardio_form/config.py -> parents[2] == project root (one above src/).
# Only meaningful in an editable checkout; used for the file:// dev weights.
_DEV_ROOT = Path(__file__).resolve().parents[2]

# Named label spaces -> manifest filename. Every stage of the pipeline uses a
# different set of integers, so a label name is only meaningful against a space.
LABEL_SPACES = {
    "whole_heart": "whole_heart_labels.yaml",  # 3D whole-heart network OUTPUT
    "sax": "sax_labels.yaml",                  # nnUNet 2D SAX output
    "lax_2ch": "lax_2ch_labels.yaml",          # nnUNet 2D LAX 2CH output
    "lax_4ch": "lax_4ch_labels.yaml",          # nnUNet 2D LAX 4CH output
    "sparse": "sparse_labels.yaml",            # sparse volume fed to 3D recon
    "left": "left_labels.yaml",                # LA network output
}

DEFAULT_LABEL_SPACE = "whole_heart"


def config_dir() -> Path:
    """Directory containing the YAML manifests.

    Honours ``CARDIOFORM_CONFIG_DIR`` if set, otherwise the packaged
    ``config_data`` directory.
    """
    env = os.environ.get("CARDIOFORM_CONFIG_DIR")
    return Path(env) if env else _DEFAULT_CONFIG_DIR


def config_path(filename: str) -> Path:
    """Absolute path to a config file (e.g. ``config_path('models.yaml')``)."""
    return config_dir() / filename


def dev_root() -> Path:
    """Project root, for resolving ``file://`` development paths in models.yaml.

    Honours ``CARDIOFORM_DEV_ROOT`` if set. Only relevant to an editable
    checkout with a local ``weights/`` directory; installed users resolve models
    from URLs instead.
    """
    env = os.environ.get("CARDIOFORM_DEV_ROOT")
    return Path(env) if env else _DEV_ROOT


def label_space_path(space: str = DEFAULT_LABEL_SPACE) -> Path:
    """Absolute path to the manifest for a named label space."""
    if space not in LABEL_SPACES:
        raise KeyError(
            f"Unknown label space '{space}'. Available: {sorted(LABEL_SPACES)}"
        )
    return config_path(LABEL_SPACES[space])


# Valid cardiac view identifiers. Defined here rather than in pipeline.py so the
# CLI can build its argparse choices without importing the heavy model stack.
CHOICES_VIEW_TYPE = ['sax', 'lax_2ch', 'lax_4ch']
