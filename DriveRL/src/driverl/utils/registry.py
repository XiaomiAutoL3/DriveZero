import inspect
from typing import Callable

from driverl.utils.logging import logger

__all__ = ["Registry"]


class Registry(object):
    """
    Registry used throughout the project.
    Mock from MMDET: https://mmdetection.readthedocs.io/en/v2.2.0/tutorials/new_modules.html
    """

    def __init__(self, name):
        """Initialize a registry with name."""
        self._name = name
        self._module_dict = dict()

    def __repr__(self):
        format_str = self.__class__.__name__ + "(name={}, items={})".format(
            self._name, list(self._module_dict.keys())
        )
        return format_str

    @property
    def name(self):
        """Name of the registry."""
        return self._name

    @property
    def module_keys(self):
        """List of registered modules."""
        return list(self._module_dict.keys())

    @property
    def module_dict(self):
        """Dictionary of registered modules."""
        return self._module_dict

    def get(self, key, check=True) -> Callable:
        """Get a registered module."""
        if check and key not in self._module_dict:
            raise KeyError(
                f"{key} is not registered in {self.name}, currently registered modules are {self.module_keys}"
            )
        return self._module_dict.get(key, None)

    def _register_module(self, module_class):
        """Register a module.

        Args:
            module (:obj:`nn.Module`): Module to be registered.
        """
        if not inspect.isclass(module_class):
            raise TypeError(
                f"module must be a class, but got {format(type(module_class))}"
            )
        module_name = module_class.__name__
        if module_name in self._module_dict:
            logger.warning(f"{module_name} is already registered in {self.name}.")
            return
        self._module_dict[module_name] = module_class

    def register_module(self, cls):
        """Register a module."""
        self._register_module(cls)
        return cls

    @staticmethod
    def get_model_instance_name(model_name, base_class_name=""):
        model_instance_name = model_name
        if model_name.islower():
            model_instance_name = "".join(
                [_.capitalize() for _ in model_name.split("_")]
            )
        model_instance_name += base_class_name
        return model_instance_name
