"""
This module defines the configuration for agents.
"""

from dataclasses import dataclass

import torch

from driverl.configs.base_config import BaseConfig
from driverl.utils.logging import logger


@dataclass
class AgentConfig(BaseConfig):
    """Configuration for agents."""

    agent_name: str = "vanilla_net_agent"
    network_name: str = "vanilla_net"
    compile: bool = False
    device: str = "cuda"  # Options: "cpu", "cuda"
    enable_occupancy_grid: bool = True
    dynamics_model: str = "bicycle_model"  # Options: ""bicycle_model", "delta_local"
    embed_dim: int = 128
    num_heads: int = 4
    max_agents: int = 0
    min_vehicle_agents: int = 32
    min_pedestrian_agents: int = 32
    max_lanes: int = 0
    max_lane_centers: int = 0
    history_steps: int = 1
    future_steps: int = 0
    enable_agent_frame_masking: bool = False
    history_frame_gap: int = 1
    mode: str = "train"
    enable_value_decomposition: bool = False
    # Number of (x, y) goals consumed by goal-aware agents. The default of 1
    # keeps VanillaNet's original single-goal architecture.
    num_goal_positions: int = 1
    # Preserve goal-slot order instead of mean-pooling goals as an unordered set.
    enable_ordered_goal_encoding: bool = False
    # Filter radii around ego. ``agent_filter_radius`` applies to agent
    # features (and lateral lane reach). ``lane_forward_range`` and
    # ``lane_backward_range`` define an asymmetric longitudinal lane window in
    # the ego frame. Defaulting the lane values to 200 keeps parity with the
    # original circular radius filter; override in yaml to shrink either side.
    agent_filter_radius: float = 200.0
    lane_forward_range: float = 200.0
    lane_backward_range: float = 200.0

    # Optional VanillaNet capacity knobs. Defaults reproduce the original
    # architecture exactly, so old configs and checkpoints keep working.
    # Preserve original LaneType codes instead of the legacy 0/1/2 remapping.
    use_raw_lane_types: bool = False
    # Include steering-rate in VanillaNet kinematics / visualizer state.
    use_steering_rate: bool = True
    # Keep ego history frames in VanillaNet agent features.
    use_ego_history: bool = False
    # Total number of ego→agent cross-attention layers (>=1). 1 = legacy.
    num_agent_attention_layers: int = 1
    # Replace masked-amax global lane token with ego→lane cross-attention.
    enable_lane_attention: bool = False
    num_lane_attention_layers: int = 1
    # None means "same as num_heads".
    num_heads_lane: int | None = None
    # Append current-frame traffic-light one-hot features to lane features.
    enable_traffic_light_features: bool = False

    def __post_init__(self):
        """Validate configuration values."""

        if self.future_steps < 0:
            raise ValueError("future_steps must be non-negative.")

        # Check CUDA availability and warn if needed
        if self.device == "cuda" and not torch.cuda.is_available():
            logger.warning(
                "CUDA device specified but not available. "
                "Automatically switching to CPU device."
            )
            self.device = "cpu"
