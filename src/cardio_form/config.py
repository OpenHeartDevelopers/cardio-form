"""Resolution of CardioForm configuration files (labels.yaml / models.yaml).

The package now lives under ``src/cardio_form/``, so the YAML manifests sit two
levels above this file (the project root). The location can be overridden
explicitly with the ``CARDIOFORM_CONFIG_DIR`` environment variable, following
the project's "explicit environment injection" rule.

In the Docker image the package is at ``/app/src/cardio_form`` and the manifests
are copied to ``/app``, so the same ``parents[2]`` resolution holds.
"""

import os
from pathlib import Path

# .../src/cardio_form/config.py -> parents[2] == project root (one above src/)
_DEFAULT_CONFIG_DIR = Path(__file__).resolve().parents[2]


def config_dir() -> Path:
    """Directory containing labels.yaml / models.yaml.

    Honours ``CARDIOFORM_CONFIG_DIR`` if set, otherwise defaults to the project
    root inferred from this file's location.
    """
    env = os.environ.get("CARDIOFORM_CONFIG_DIR")
    return Path(env) if env else _DEFAULT_CONFIG_DIR


def config_path(filename: str) -> Path:
    """Absolute path to a config file (e.g. ``config_path('labels.yaml')``)."""
    return config_dir() / filename
