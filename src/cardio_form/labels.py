"""Anatomical label management for CardioForm.

The label logic now lives in the shared ``pycemrg`` library
(``pycemrg.data.LabelManager``); this module keeps the historical
``cardio_form.labels`` import path stable and wires up the default manager
against the project's ``labels.yaml``.

``labels.yaml`` retains the same schema pycemrg expects (top-level ``labels``
and ``groups`` keys, with recursive groups), so the swap is behaviour-preserving.
"""

from pycemrg.data import LabelManager

from cardio_form.config import config_path

# pycemrg's LabelManager requires an explicit manifest path. Resolve labels.yaml
# at the project root (override with CARDIOFORM_CONFIG_DIR).
default_label_manager = LabelManager(config_path("labels.yaml"))

__all__ = ["LabelManager", "default_label_manager"]
