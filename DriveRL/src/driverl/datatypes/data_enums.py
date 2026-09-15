"""Data enums used by the DriveRL runtime."""

from enum import Enum

import torch


class LaneType(Enum):
    CURB = 0, "CURB"
    LANE_CENTER_LINE = 1, "LANE_CENTER_LINE"
    SOLID_LINE = 2, "SOLID_LINE"
    DASHED_LINE = 3, "DASHED_LINE"
    ROAD_CENTER_LINE = 4, "ROAD_CENTER_LINE"
    ROAD_BOUNDARY = 5, "ROAD_BOUNDARY"
    STOP_LINE = 6, "STOP_LINE"
    DOUBLE_SOLID_LINE = 7, "DOUBLE_SOLID_LINE"
    DOUBLE_DASHED_LINE = 8, "DOUBLE_DASHED_LINE"
    DASHED_SOLID_LINE = 9, "DASHED_SOLID_LINE"
    SOLID_DASHED_LINE = 10, "SOLID_DASHED_LINE"
    CROSSWALK = 11, "CROSSWALK"

    def __init__(self, code: int, label: str):
        self.code = code
        self.label = label

    def __int__(self) -> int:
        return self.code

    def __str__(self) -> str:
        return self.label


class AgentControlType(Enum):
    LOG_REPLAY = 0
    CONTROLLED = 1


class AgentControlManager:
    """Track the policy-controlled agents in a rollout batch."""

    def __init__(
        self,
        num_envs: int,
        max_agents: int,
        device: torch.device,
    ):
        self.num_envs = num_envs
        self.max_agents = max_agents
        self.device = device
        self._control_types = torch.full(
            (num_envs, max_agents),
            AgentControlType.LOG_REPLAY.value,
            dtype=torch.int8,
            device=device,
        )

    @property
    def log_replay_mask(self) -> torch.Tensor:
        return self._control_types == AgentControlType.LOG_REPLAY.value

    @property
    def controlled_mask(self) -> torch.Tensor:
        return self._control_types == AgentControlType.CONTROLLED.value

    def set_controlled(self, mask: torch.Tensor) -> None:
        self._control_types[self.controlled_mask] = AgentControlType.LOG_REPLAY.value
        self._control_types[mask] = AgentControlType.CONTROLLED.value
