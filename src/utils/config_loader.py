"""
Configuration loader utility for YAML-based hierarchical configuration management.
Supports environment variable overrides and nested config merging.
"""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, Union
import copy


class IncludeLoader(yaml.SafeLoader):
    """Custom YAML loader with !include constructor."""

    def __init__(self, stream):
        self._root = Path(stream.name).parent
        super().__init__(stream)


def include_constructor(loader, node):
    """Construct a YAML node by including another file."""
    filename = loader.construct_scalar(node)
    filepath = loader._root / filename

    if not filepath.exists():
        raise FileNotFoundError(f"Included file not found: {filepath}")

    with open(filepath, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


# Register the !include constructor
yaml.add_constructor('!include', include_constructor, IncludeLoader)


class ConfigLoader:
    """Load and manage hierarchical YAML configurations."""

    def __init__(self, config_path: Union[str, Path]):
        """
        Initialize the configuration loader.

        Args:
            config_path: Path to the main configuration file
        """
        self.config_path = Path(config_path)
        self.config_dir = self.config_path.parent
        self.config = None

    def load(self) -> Dict[str, Any]:
        """
        Load configuration from YAML file with support for includes.

        Returns:
            Dictionary containing the full configuration
        """
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = yaml.load(f, Loader=IncludeLoader)

        # Apply environment variable overrides
        config = self._apply_env_overrides(config)

        self.config = config
        return config

    def _load_yaml(self, path: Path) -> Dict[str, Any]:
        """Load a YAML file."""
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def _apply_env_overrides(self, config: Dict[str, Any], prefix: str = "MMLDD") -> Dict[str, Any]:
        """
        Apply environment variable overrides to configuration.

        Environment variables should be named: MMLDD_SECTION_KEY=value
        Example: MMLDD_TRAINING_BATCH_SIZE=64

        Args:
            config: Configuration dictionary
            prefix: Environment variable prefix

        Returns:
            Configuration with environment overrides applied
        """
        config_copy = copy.deepcopy(config)

        for env_var, value in os.environ.items():
            if env_var.startswith(prefix + "_"):
                # Remove prefix and split by underscore
                keys = env_var[len(prefix)+1:].lower().split('_')

                # Navigate to the nested key and update
                current = config_copy
                for key in keys[:-1]:
                    if key in current and isinstance(current[key], dict):
                        current = current[key]
                    else:
                        break
                else:
                    # Set the value (attempt type conversion)
                    final_key = keys[-1]
                    if final_key in current:
                        current[final_key] = self._convert_type(value, type(current[final_key]))

        return config_copy

    def _convert_type(self, value: str, target_type: type) -> Any:
        """Convert string value to target type."""
        if target_type == bool:
            return value.lower() in ('true', '1', 'yes', 'on')
        elif target_type == int:
            return int(value)
        elif target_type == float:
            return float(value)
        elif target_type == list:
            return value.split(',')
        else:
            return value

    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get a configuration value using dot notation.

        Args:
            key_path: Dot-separated path to the config value (e.g., 'training.batch_size')
            default: Default value if key not found

        Returns:
            Configuration value or default

        Example:
            >>> config_loader.get('training.batch_size')
            32
        """
        if self.config is None:
            raise RuntimeError("Configuration not loaded. Call load() first.")

        keys = key_path.split('.')
        current = self.config

        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default

        return current

    def update(self, key_path: str, value: Any) -> None:
        """
        Update a configuration value using dot notation.

        Args:
            key_path: Dot-separated path to the config value
            value: New value to set
        """
        if self.config is None:
            raise RuntimeError("Configuration not loaded. Call load() first.")

        keys = key_path.split('.')
        current = self.config

        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]

        current[keys[-1]] = value

    def merge(self, other_config: Dict[str, Any]) -> None:
        """
        Merge another configuration dictionary into the current config.

        Args:
            other_config: Configuration dictionary to merge
        """
        if self.config is None:
            self.config = other_config
        else:
            self.config = self._deep_merge(self.config, other_config)

    def _deep_merge(self, base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively merge two dictionaries."""
        result = copy.deepcopy(base)

        for key, value in update.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = copy.deepcopy(value)

        return result

    def save(self, output_path: Union[str, Path]) -> None:
        """
        Save the current configuration to a YAML file.

        Args:
            output_path: Path to save the configuration
        """
        if self.config is None:
            raise RuntimeError("Configuration not loaded. Call load() first.")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)

    def to_dict(self) -> Dict[str, Any]:
        """Return the configuration as a dictionary."""
        if self.config is None:
            raise RuntimeError("Configuration not loaded. Call load() first.")
        return copy.deepcopy(self.config)


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Convenience function to load a configuration file.

    Args:
        config_path: Path to the configuration file

    Returns:
        Configuration dictionary
    """
    loader = ConfigLoader(config_path)
    return loader.load()


def get_default_config_path() -> Path:
    """
    Get the default configuration file path.

    Returns:
        Path to the default config.yaml file
    """
    project_root = Path(__file__).parent.parent.parent
    return project_root / "config" / "config.yaml"


if __name__ == "__main__":
    # Test the config loader
    config_path = get_default_config_path()

    if config_path.exists():
        loader = ConfigLoader(config_path)
        config = loader.load()

        print("Configuration loaded successfully!")
        print(f"Project name: {loader.get('project.name')}")
        print(f"Batch size: {loader.get('training.batch_size')}")
        print(f"Number of classes: {loader.get('dataset.num_classes')}")
    else:
        print(f"Config file not found at: {config_path}")
