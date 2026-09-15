"""
This module defines the base class for neural networks used in learning agents.
"""

import numpy as np
import torch
import torch.nn as nn

from driverl.utils.registry import Registry

NETWORK_REGISTER = Registry("network")


class BaseNetwork(nn.Module):
    """Base class for all neural networks."""

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, *args, **kwargs):
        raise NotImplementedError("Each network must implement its own forward pass.")

    @staticmethod
    def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
        """CleanRL's default layer initialization"""
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
        return layer

    @staticmethod
    def network_factory(network_name: str, **kwargs) -> "BaseNetwork":
        """Factory method for networks."""
        network_name = Registry.get_model_instance_name(network_name)
        supported_network_names = NETWORK_REGISTER.module_keys
        assert network_name in supported_network_names, (
            f"Currently only support {supported_network_names}, but got {network_name}."
        )
        return NETWORK_REGISTER.get(network_name)(**kwargs)
