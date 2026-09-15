"""Configuration for DriveRL nuPlan simulation adapters."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from driverl.env.domain_randomization.config import DomainRandomizationConfig

_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_DRIVERL_NUPLAN_CONFIG_PATH = (
    str(_REPOSITORY_ROOT / "release" / "configs" / "driverl_teacher.yaml")
)
DEFAULT_DRIVERL_NUPLAN_CHECKPOINT_PATH = (
    str(_REPOSITORY_ROOT / "release" / "checkpoints" / "checkpoint_2400.pt")
)


@dataclass
class DriveRLNuPlanControllerConfig:
    """Runtime settings for executing DriveRL actions in nuPlan simulation."""

    device: str = "cpu"
    dynamics_model: str = "nuplan_bicycle_model"
    frame_time_interval: float = 0.1
    enable_dynamics_noise: bool = False
    dynamics_noise_scale: float = 1.0
    min_jerk_long: float = -5.0
    max_jerk_long: float = 3.0
    positive_jerk_limit_when_nonnegative_acc: float = 1.0
    min_jerk_lat: float = -1.0
    max_jerk_lat: float = 1.0
    min_a_long: float = -6.5
    max_a_long: float = 1.5
    min_a_lat: float = -3.0
    max_a_lat: float = 3.0
    control_delay_frame: int = 0
    control_delay_smoothing: float = 0.0
    lateral_mode_transition_speed: float = 5.0
    lateral_mode_transition_width: float = 0.3
    lateral_action_space_shape_gamma: float = 2.0
    max_tire_angle_rate_lower_bound: float = 0.03
    max_tire_angle_rate_upper_bound: float = 0.21
    max_tire_angle_rate_gain: float = 0.973
    num_jerk_lat_actions: int = 64
    num_jerk_long_actions: int = 24
    nuplan_bicycle_accel_time_constant: float = 0.2
    nuplan_bicycle_steering_angle_time_constant: float = 0.05
    nuplan_bicycle_max_steering_angle: float = 1.0471975511965976
    nuplan_bicycle_max_acceleration: float = 3.0
    nuplan_bicycle_max_steering_rate: float = 0.5
    nuplan_bicycle_wheelbase: float = 3.089
    nuplan_bicycle_rear_axle_to_center: float = 1.461


@dataclass
class DriveRLNuPlanFeatureBuilderConfig:
    """Shape/device settings for runtime nuPlan to DriveRL feature conversion."""

    device: str = "cpu"
    history_steps: int = 5
    history_sample_interval: float = 0.2
    max_agents: int = 128
    min_vehicle_agents: int = 64
    num_goal_positions: int = 2
    route_goal_horizon_s: float = 12.0
    route_goal_min_speed_mps: float = 5.0
    route_goal_pair_mode: str = "legacy"
    max_lanes_centers: int = 1024
    max_lanes_other: int = 1024
    max_route_points: int = 100
    enable_occupancy_grid: bool = False
    default_agent_length: float = 4.5
    default_agent_width: float = 2.0
    map_radius_m: float = 50.0
    use_nuplan_current_ego_map_query: bool = True
    route_map_query_points: int = 15
    route_map_query_spacing_m: float = 15.0
    route_map_query_distance_m: float = 160.0
    domain_randomization: DomainRandomizationConfig | None = None

    @property
    def route_roadblock_correction(self) -> bool:
        """Route correction is mandatory in the public inference contract."""
        return True


@dataclass
class DriveRLNuPlanPlannerConfig:
    """Runtime settings for the nuPlan planner adapter."""

    config_path: str = DEFAULT_DRIVERL_NUPLAN_CONFIG_PATH
    checkpoint_path: str = DEFAULT_DRIVERL_NUPLAN_CHECKPOINT_PATH
    device: str = "cuda"
    sampling_method: str = "argmax"
    trajectory_num_poses: int = 80
    trajectory_interval_length: float = 0.1
    trajectory_mode: str = "action"
    route_goal_horizon_s: float = 12.0
    route_goal_min_speed_mps: float = 5.0
    route_goal_pair_mode: str = "legacy"
    lqr_action_match_heading_span: float = 0.8
    lqr_action_match_lateral_span: float = 3.0
    lqr_action_match_heading_samples: int = 9
    lqr_action_match_lateral_samples: int = 5
    lqr_discretization_time: float = 0.1
    lqr_tracking_horizon: int = 10
    nuplan_bicycle_accel_time_constant: float = 0.2
    nuplan_bicycle_max_acceleration: float = 3.0
    nuplan_bicycle_max_steering_rate: float = 0.5
    policy_interval_s: float = 0.0
    tts_enabled: bool = False
    tts_num_candidates: int = 8
    tts_seed: int = 42
    strict_checkpoint: bool = True
    compile_agent: bool = False
    min_jerk_long: float = -5.0
    max_jerk_long: float = 3.0
    positive_jerk_limit_when_nonnegative_acc: float = 1.0
    min_jerk_lat: float = -1.0
    max_jerk_lat: float = 1.0

    @property
    def route_roadblock_correction(self) -> bool:
        """Route correction is mandatory in the public inference contract."""
        return True
