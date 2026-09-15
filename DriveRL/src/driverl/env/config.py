"""Inference configuration shared by the policy loader and TTS runtime."""

from dataclasses import dataclass, field

import torch

from driverl.configs.base_config import BaseConfig
from driverl.env.dataloader.config import DataLoaderConfig
from driverl.env.engine.config import EngineConfig
from driverl.utils.logging import logger


@dataclass
class EnvConfig(BaseConfig):
    scenario_control_type: str = "non_reactive"
    device: str = "cuda"
    enable_occupancy_grid: bool = False
    action_type: str = "continuous"
    dynamics_model: str = "nuplan_bicycle_model"
    num_steps: int = 110
    goal_count_probs: list[float] = field(default_factory=lambda: [0.5, 0.5])
    dataloader_config: DataLoaderConfig = field(default_factory=DataLoaderConfig)
    engine_config: EngineConfig = field(default_factory=EngineConfig)

    @property
    def num_goal_positions(self) -> int:
        return len(self.goal_count_probs)

    def __post_init__(self) -> None:
        if not self.goal_count_probs or sum(self.goal_count_probs) <= 0:
            raise ValueError("goal_count_probs must sum to a positive value")
        if self.device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA is unavailable; using CPU for DriveRL inference")
            self.device = "cpu"
        if isinstance(self.dataloader_config, dict):
            self.dataloader_config = DataLoaderConfig.from_dict(
                self.dataloader_config, allow_extra=True
            )
        if isinstance(self.engine_config, dict):
            self.engine_config = EngineConfig.from_dict(
                self.engine_config, allow_extra=True
            )
        self.dataloader_config.num_goal_positions = self.num_goal_positions
        self.dataloader_config.load_occupancy_grid = self.enable_occupancy_grid
