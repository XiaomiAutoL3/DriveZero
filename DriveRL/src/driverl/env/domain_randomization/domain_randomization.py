from dataclasses import dataclass, field
from typing import Callable

import torch

from driverl.env.domain_randomization.config import (
    DomainRandomizationConfig,
    RewardFeatureConfig,
)


@dataclass
class RandomizedFeatureDict:
    """Container for sampled domain-randomization tensors."""

    config: DomainRandomizationConfig
    values: dict[str, torch.Tensor] = field(default_factory=dict)

    def has(self, key: str) -> bool:
        return key in self.values

    def get(self, key: str, calculate: bool = False) -> torch.Tensor:
        value = self.values.get(key)
        if value is None:
            raise KeyError(f"Feature '{key}' not found in RandomizedFeatureDict.")

        if not self.config.enabled and calculate:
            feature_config = self.config.to_dict()[key]
            if feature_config.calculator_default is not None:
                value = torch.full_like(value, feature_config.calculator_default)

        return value

    def get_enabled_features(self) -> torch.Tensor | None:
        """Get all valid features."""
        if not self.config.enabled_feature_config_list:
            return None
        return torch.stack(
            [
                self.values[feature_name]
                for feature_name, feature in self.config.to_dict().items()
                if feature.enabled and feature.as_feature
            ],
            dim=-1,
        )


def _uniform_sampler(
    batch_size: int, max_agents: int, config: RewardFeatureConfig, device: torch.device
) -> torch.Tensor:
    """Sample uniform random values in [min, max]."""
    min_val = config.min
    max_val = config.max
    return (
        torch.rand((batch_size, max_agents), device=device) * (max_val - min_val)
        + min_val
    )


FEATURE_SAMPLERS: dict[str, Callable[..., torch.Tensor]] = {
    # "frame_delay": _sample_frame_delay,
}


def sample_randomized_features(
    batch_size: int,
    max_agents: int,
    domain_randomization_config: DomainRandomizationConfig,
    device: torch.device,
) -> RandomizedFeatureDict:
    """Sample domain-randomization tensors for one rollout."""
    cfg = (
        DomainRandomizationConfig.from_dict(domain_randomization_config)
        if isinstance(domain_randomization_config, dict)
        else domain_randomization_config
    )

    tensors: dict[str, torch.Tensor] = {}

    for feature_name, feature in cfg.to_dict().items():
        if cfg.enabled and feature.enabled:
            sampler = FEATURE_SAMPLERS.get(feature_name, _uniform_sampler)
            tensors[feature_name] = sampler(batch_size, max_agents, feature, device)
        else:
            tensors[feature_name] = torch.full(
                (batch_size, max_agents), feature.default, device=device
            )
    return RandomizedFeatureDict(cfg, tensors)


__all__ = [
    "RandomizedFeatureDict",
    "sample_randomized_features",
]
