"""Anatomical label management for CardioForm.

The label logic now lives in the shared ``pycemrg`` library
(``pycemrg.data.LabelManager``); this module keeps the historical
``cardio_form.labels`` import path stable and wires up the default manager
against the whole-heart label space.

Each label-space manifest uses the schema pycemrg expects (top-level ``labels``
and ``groups`` keys, with recursive groups), so the swap is behaviour-preserving.
"""

from pycemrg.data import LabelManager

from cardio_form.config import label_space_path

# pycemrg's LabelManager requires an explicit manifest path. The default
# manager reads the whole-heart output space; see config.LABEL_SPACES.
default_label_manager = LabelManager(label_space_path("whole_heart"))

__all__ = ["LabelManager", "default_label_manager"]
