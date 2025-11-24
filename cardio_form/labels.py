# in cardio_form/labels.py

import yaml
from pathlib import Path

class LabelManager:
    """
    Manages anatomical label definitions from a YAML configuration file.

    This class reads a label manifest (e.g., labels.yaml) and provides
    utilities to translate between human-readable names (like 'LV_myo'),
    groups (like 'ventricles'), and their corresponding integer values.
    """
    def __init__(self, config_path=None):
        """
        Initializes the LabelManager by loading the configuration file.

        Args:
            config_path (str, optional): Path to the labels YAML file. If None,
                                         it defaults to 'labels.yaml' in the
                                         project root.
        """
        if config_path is None:
            # Assume this file is in cardio_form/ and labels.yaml is in the parent dir
            config_path = Path(__file__).parent.parent / 'labels.yaml'

        if not Path(config_path).exists():
            raise FileNotFoundError(f"Label configuration file not found at: {config_path}")

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        self._labels = config.get('labels', {})
        self._groups = config.get('groups', {})
        
        # Create a reverse mapping for convenience (e.g., {1: 'LV_myo'})
        self._value_to_name = {v: k for k, v in self._labels.items()}

    def get_value(self, name: str) -> int:
        """
        Translates a single label name (e.g., 'LV_myo') to its integer value.
        """
        if name not in self._labels:
            raise KeyError(f"Label name '{name}' not found.")
        return self._labels[name]
    
    def get_name(self, value: int) -> str:
        """
        Translates an integer value (e.g., 1) to its human-readable name.
        """
        if value not in self._value_to_name:
            raise KeyError(f"Label value '{value}' not found.")
        return self._value_to_name[value]

    def get_values_from_names(self, names: list[str]) -> list[int]:
        """
        Translates a list of strings into a sorted, unique list of integer label values.

        The input list can contain:
        - Individual label names (e.g., 'LV_myo')
        - Group names (e.g., 'ventricles'), which will be expanded
        - Numbers as strings (e.g., '5')

        Examples:
            get_values_from_names(['LV_myo', 'RV_bp']) -> [1, 3]
            get_values_from_names(['ventricles', 'LA_bp']) -> [1, 2, 3, 5]
            get_values_from_names(['1', '3', 'LA_bp']) -> [1, 3, 5]
        """
        final_values = set()
        for name in names:
            if name.isdigit():
                final_values.add(int(name))
            elif name in self._labels:
                final_values.add(self._labels[name])
            elif name in self._groups:
                # Recursively expand the group to get its integer values
                group_names = self._groups[name]
                group_values = self.get_values_from_names(group_names)
                final_values.update(group_values)
            else:
                # Provide helpful error message with available options
                available_keys = list(self._labels.keys()) + list(self._groups.keys())
                raise KeyError(
                    f"Label or group name '{name}' not found. "
                    f"Available keys are: {available_keys}"
                )
        
        return sorted(list(final_values))

# For convenience, create a default instance that can be imported elsewhere
default_label_manager = LabelManager()