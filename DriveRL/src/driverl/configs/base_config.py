"""Base class for configuration objects."""

from dataclasses import dataclass, fields
from typing import Any, ClassVar, Dict, List, Type, TypeVar

import yaml

from driverl.utils.logging import logger
from driverl.utils.misc import RecursiveLoader, apply_extends

T = TypeVar("T", bound="BaseConfig")


@dataclass
class BaseConfig:
    """Base class for configuration objects."""

    _extra_key_warnings: ClassVar[List[str]] = []

    @classmethod
    def collect_extra_key_warnings(cls) -> list[str]:
        """Return and clear accumulated extra-key warnings."""
        warnings = cls._extra_key_warnings.copy()
        cls._extra_key_warnings.clear()
        return warnings

    @staticmethod
    def _coerce_nested_value(field_type: Any, value: Any, *, allow_extra: bool) -> Any:
        """Recursively parse nested config dictionaries when possible."""
        if not isinstance(value, dict):
            return value
        if isinstance(field_type, type) and issubclass(field_type, BaseConfig):
            return field_type.from_dict(value, allow_extra=allow_extra)
        return value

    @classmethod
    def from_dict(
        cls: Type[T], data: Dict[str, Any], *, allow_extra: bool = False
    ) -> T:
        """Create an instance from a dictionary.

        Args:
            data: Dictionary containing configuration data
            allow_extra: If True, ignore keys that are not defined in the dataclass

        Returns:
            Instance of the configuration class

        Raises:
            ValueError: If dictionary contains keys not defined in the dataclass
        """
        # Get the dataclass fields keyed by name
        field_map = {field.name: field for field in fields(cls)}
        field_names = set(field_map)

        # Check for unexpected keys
        unexpected_keys = set(data.keys()) - field_names
        if unexpected_keys:
            if not allow_extra:
                raise ValueError(f"Unexpected configuration keys: {unexpected_keys}")
            msg = f"{cls.__name__}: ignoring unexpected keys {sorted(unexpected_keys)}"
            logger.warning(msg)
            cls._extra_key_warnings.append(msg)

        # Create instance with the data, recursively parsing nested configs
        filtered_data = {
            key: cls._coerce_nested_value(
                field_map[key].type, value, allow_extra=allow_extra
            )
            for key, value in data.items()
            if key in field_names
        }
        return cls(**filtered_data)

    @classmethod
    def from_yaml_file(cls: Type[T], file_path: str, *, allow_extra: bool = False) -> T:
        """Load configuration from a YAML file.

        Args:
            file_path: Path to the YAML configuration file
            allow_extra: If True, ignore keys that are not defined in the dataclass

        Returns:
            Instance of the configuration class
        """

        with open(file_path, "r") as f:
            data = yaml.load(f, Loader=RecursiveLoader)
        data = apply_extends(data)
        return cls.from_dict(data, allow_extra=allow_extra)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary.

        Returns:
            Dictionary representation of the configuration
        """
        return {field.name: getattr(self, field.name) for field in fields(self)}

    def update(self, config_dict: dict):
        """
        Updates the attributes of the configuration object from a dictionary.

        Args:
            config_dict: A dictionary containing the configuration values.

        Raises:
            ValueError: If the config_dict contains keys that are not
                defined in the dataclass.
        """
        for key, value in config_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown configuration key: {key}")

        return self
