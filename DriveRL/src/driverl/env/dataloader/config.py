"""Minimal data settings required by inference configuration parsing."""

from dataclasses import dataclass, field
from typing import Optional

from driverl.configs.base_config import BaseConfig


@dataclass
class DataLoaderConfig(BaseConfig):
    dataset_type: str = "legacy"
    dataset_files: dict[str, str] = field(default_factory=dict)
    batch_size: int = 1
    num_workers: int = 0
    drop_last: bool = False
    shuffle: Optional[bool] = False
    target_sample_rate_hz: Optional[float] = 5.0
    load_occupancy_grid: bool = False
    load_traffic_light_features: bool = True
    num_goal_positions: int = 1

    def __post_init__(self) -> None:
        if self.target_sample_rate_hz is not None and self.target_sample_rate_hz <= 0:
            raise ValueError("target_sample_rate_hz must be positive")
