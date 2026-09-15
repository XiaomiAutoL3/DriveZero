from dataclasses import dataclass, field

from driverl.configs.base_config import BaseConfig
from driverl.env.domain_randomization.config import (
    DomainRandomizationConfig,
)


@dataclass
class EngineConfig(BaseConfig):
    """Static dynamics and reward settings used by TTS."""

    done_after_reaching_goal: bool = False
    enable_visible_mask: bool = False
    enable_narrow_road_right_preference_reward: bool = False
    control_delay_frame: int = 0

    collision_reward_weight: float = -1.0
    collision_segment_time: float = 0.2
    collision_distance_threshold: float = 0.0
    collision_speed_buffer_base: float = 0.0
    collision_speed_buffer_max: float = 0.0
    collision_relative_speed_buffer_gain: float = 0.0
    enable_dynamics_noise: bool = False

    min_jerk_long: float = -2.0
    max_jerk_long: float = 1.0
    positive_jerk_limit_when_nonnegative_acc: float = 1.0
    min_jerk_lat: float = -1.0
    max_jerk_lat: float = 1.0
    min_a_long: float = -3.0
    max_a_long: float = 1.5
    min_a_lat: float = -3.0
    max_a_lat: float = 3.0
    nuplan_bicycle_accel_time_constant: float = 0.2
    nuplan_bicycle_steering_angle_time_constant: float = 0.05
    nuplan_bicycle_max_steering_angle: float = 1.0471975511965976
    nuplan_bicycle_max_acceleration: float = 3.0
    nuplan_bicycle_max_steering_rate: float = 0.5
    nuplan_bicycle_wheelbase: float = 3.089
    nuplan_bicycle_rear_axle_to_center: float = 1.461

    # Speed-adaptive lateral acceleration comfort thresholds
    a_lat_comfort_low_speed: float = 5.0  # m/s; below this, threshold stays at minimum
    a_lat_comfort_high_speed: float = (
        30.0  # m/s; above this, threshold stays at maximum
    )
    a_lat_comfort_threshold_low: float = 4.0  # m/s²; comfort threshold at low speed
    a_lat_comfort_threshold_high: float = 6.0  # m/s²; comfort threshold at high speed

    # Speed-adaptive longitudinal deceleration comfort thresholds
    a_long_decel_full_speed: float = (
        2.78  # m/s (~10 km/h); at/above this speed, full min_a_long applies
    )
    a_long_decel_floor: float = -0.8  # m/s²; effective decel threshold at zero speed

    nuplan_comfort_min_lon_accel: float = -4.05
    nuplan_comfort_max_lon_accel: float = 2.40
    nuplan_comfort_max_abs_lat_accel: float = 4.89
    nuplan_comfort_max_abs_lon_jerk: float = 4.13
    nuplan_comfort_max_abs_mag_jerk: float = 8.37
    nuplan_comfort_max_abs_yaw_rate: float = 0.95
    nuplan_comfort_max_abs_yaw_accel: float = 1.93

    num_jerk_lat_actions: int = 12
    num_jerk_long_actions: int = 12

    domain_randomization: DomainRandomizationConfig = field(
        default_factory=DomainRandomizationConfig.default_preset
    )

    def __post_init__(self) -> None:
        """Normalize the nested reward configuration."""
        if isinstance(self.domain_randomization, dict):
            self.domain_randomization = DomainRandomizationConfig.from_dict(
                self.domain_randomization
            )


@dataclass
class EngineRuntimeConfig:
    """Runtime parameters for the engine that are determined at execution time."""

    batch_size: int
    max_agents: int
    frame_time_interval: float
    dynamics_model: str
    device: str
    num_steps: int
    init_steps: int = 0
    enable_occupancy_grid: bool = True
    occ_grid_xmin: float = -60.0
    occ_grid_xmax: float = 200.0
    occ_grid_ymin: float = -30.0
    occ_grid_ymax: float = 30.0
    occ_grid_resolution: float = 0.2
    occ_grid_max_range: float = 100.0
    occ_num_rays: int = 512
    occ_k: int = 1


@dataclass
class EngineMergedConfig(EngineConfig, EngineRuntimeConfig):
    """Combined engine config for convenience in downstream components."""

    @classmethod
    def from_configs(
        cls, config: EngineConfig, runtime_config: EngineRuntimeConfig
    ) -> "EngineMergedConfig":
        data = {**vars(config), **vars(runtime_config)}
        return cls(**data)

    def __post_init__(self) -> None:
        EngineConfig.__post_init__(self)


__all__ = [
    "EngineConfig",
    "EngineRuntimeConfig",
    "EngineMergedConfig",
]
