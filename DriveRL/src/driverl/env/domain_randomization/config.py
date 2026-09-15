from dataclasses import dataclass, field, fields

from driverl.configs.base_config import BaseConfig


@dataclass
class RewardFeatureConfig(BaseConfig):
    min: float = 0.0
    max: float = 0.0
    enabled: bool = False
    default: float = 0.0
    as_feature: bool = True
    calculator_default: float | None = None


@dataclass
class DomainRandomizationConfig(BaseConfig):
    enabled: bool = False
    """Configuration for reward domain randomization."""

    # collision related
    collision_reward_weight: RewardFeatureConfig = field(
        default_factory=RewardFeatureConfig
    )
    collision_segment_time: RewardFeatureConfig = field(
        default_factory=RewardFeatureConfig
    )
    collision_distance_threshold: RewardFeatureConfig = field(
        default_factory=RewardFeatureConfig
    )
    ttc_weight: RewardFeatureConfig = field(default_factory=RewardFeatureConfig)
    ttc_time: RewardFeatureConfig = field(default_factory=RewardFeatureConfig)
    ttc_width_buffer: RewardFeatureConfig = field(default_factory=RewardFeatureConfig)
    ttc_length_buffer: RewardFeatureConfig = field(default_factory=RewardFeatureConfig)
    # road bound and solid lane related
    off_road_weight: RewardFeatureConfig = field(default_factory=RewardFeatureConfig)
    cross_lane_weight: RewardFeatureConfig = field(default_factory=RewardFeatureConfig)
    wrong_way_weight: RewardFeatureConfig = field(default_factory=RewardFeatureConfig)
    # goal related
    goal_reaching_threshold: RewardFeatureConfig = field(
        default_factory=RewardFeatureConfig
    )
    goal_reaching_weight: RewardFeatureConfig = field(
        default_factory=RewardFeatureConfig
    )
    goal_reaching_distance_weight: RewardFeatureConfig = field(
        default_factory=RewardFeatureConfig
    )
    survival_reward_numerator: RewardFeatureConfig = field(
        default_factory=RewardFeatureConfig
    )
    # comfort related
    comfort_weight: RewardFeatureConfig = field(default_factory=RewardFeatureConfig)
    # center line reward configuration
    deviation_distance_weight: RewardFeatureConfig = field(
        default_factory=RewardFeatureConfig
    )
    deviation_angle_limit: RewardFeatureConfig = field(
        default_factory=RewardFeatureConfig
    )
    deviation_angle_weight: RewardFeatureConfig = field(
        default_factory=RewardFeatureConfig
    )
    max_deviation_distance_for_penalty: RewardFeatureConfig = field(
        default_factory=RewardFeatureConfig
    )
    curb_clearance_distance: RewardFeatureConfig = field(
        default_factory=RewardFeatureConfig
    )
    curb_clearance_weight: RewardFeatureConfig = field(
        default_factory=RewardFeatureConfig
    )
    # speed related
    static_speed_weight: RewardFeatureConfig = field(
        default_factory=RewardFeatureConfig
    )
    over_speed_threshold: RewardFeatureConfig = field(
        default_factory=RewardFeatureConfig
    )
    over_speed_scale: RewardFeatureConfig = field(default_factory=RewardFeatureConfig)
    # width buffer
    width_buffer: RewardFeatureConfig = field(default_factory=RewardFeatureConfig)

    # similarity related
    position_weight: RewardFeatureConfig = field(default_factory=RewardFeatureConfig)
    velocity_weight: RewardFeatureConfig = field(default_factory=RewardFeatureConfig)
    heading_weight: RewardFeatureConfig = field(default_factory=RewardFeatureConfig)
    position_threshold: RewardFeatureConfig = field(default_factory=RewardFeatureConfig)
    velocity_threshold: RewardFeatureConfig = field(default_factory=RewardFeatureConfig)
    heading_threshold: RewardFeatureConfig = field(default_factory=RewardFeatureConfig)

    def __post_init__(self) -> None:
        for f in fields(self):
            # Check if the field is of type RewardFeatureConfig
            if f.type is RewardFeatureConfig:
                value = getattr(self, f.name)
                # If the value is a dict, convert it to a RewardFeatureConfig instance
                if isinstance(value, dict):
                    setattr(self, f.name, RewardFeatureConfig.from_dict(value))

    @staticmethod
    def default_preset() -> "DomainRandomizationConfig":
        """A project default preset matching the provided YAML snippet."""
        cfg = DomainRandomizationConfig(enabled=True)

        cfg.collision_reward_weight = RewardFeatureConfig(
            min=-2.0, max=0.0, enabled=True, default=-1.0
        )
        cfg.collision_segment_time = RewardFeatureConfig(
            min=0.15, max=0.35, enabled=True, default=0.2, calculator_default=0.2
        )
        cfg.collision_distance_threshold = RewardFeatureConfig(
            min=0.0, max=0.0, enabled=False, default=0.0
        )

        cfg.ttc_weight = RewardFeatureConfig(
            min=0.0, max=1.0, enabled=True, default=0.5
        )
        cfg.ttc_time = RewardFeatureConfig(
            min=1.0, max=5.0, enabled=True, default=2.5, calculator_default=2.5
        )

        cfg.ttc_width_buffer = RewardFeatureConfig(
            min=0.0, max=0.0, enabled=False, default=0.0, calculator_default=0.0
        )
        cfg.ttc_length_buffer = RewardFeatureConfig(
            min=0.0, max=0.0, enabled=False, default=0.0, calculator_default=0.0
        )

        cfg.off_road_weight = RewardFeatureConfig(
            min=-2.0, max=0.0, enabled=True, default=-1.0
        )
        cfg.cross_lane_weight = RewardFeatureConfig(
            min=0.0, max=1.0, enabled=True, default=0.4
        )
        cfg.wrong_way_weight = RewardFeatureConfig(
            min=0.0, max=1.0, enabled=True, default=0.2
        )

        cfg.goal_reaching_threshold = RewardFeatureConfig(
            min=1.5, max=3.0, enabled=True, default=3.0
        )
        cfg.goal_reaching_weight = RewardFeatureConfig(
            min=0.0, max=1.0, enabled=True, default=0.3
        )
        cfg.goal_reaching_distance_weight = RewardFeatureConfig(
            min=0.0, max=0.1, enabled=False, default=0.0
        )

        cfg.survival_reward_numerator = RewardFeatureConfig(
            min=0.0, max=0.0, enabled=False, default=0.0
        )

        cfg.comfort_weight = RewardFeatureConfig(
            min=0.0, max=1.0, enabled=True, default=0.5
        )

        cfg.deviation_distance_weight = RewardFeatureConfig(
            min=0.75, max=0.75, enabled=False, default=0.75
        )
        cfg.deviation_angle_limit = RewardFeatureConfig(
            min=0.75, max=0.75, enabled=False, default=0.75
        )
        cfg.deviation_angle_weight = RewardFeatureConfig(
            min=0.0, max=0.0, enabled=False, default=0.0
        )
        cfg.max_deviation_distance_for_penalty = RewardFeatureConfig(
            min=4.0, max=4.0, enabled=False, default=4.0
        )
        cfg.curb_clearance_distance = RewardFeatureConfig(
            min=0.0, max=1.0, enabled=False, default=0.7
        )
        cfg.curb_clearance_weight = RewardFeatureConfig(
            min=0.0, max=1.0, enabled=True, default=0.5
        )

        cfg.static_speed_weight = RewardFeatureConfig(
            min=0.2, max=1.0, enabled=True, default=0.6
        )
        cfg.over_speed_threshold = RewardFeatureConfig(
            min=2.23, max=2.23, enabled=False, default=2.23
        )
        cfg.over_speed_scale = RewardFeatureConfig(
            min=0.5, max=1.5, enabled=False, default=1.0
        )

        cfg.width_buffer = RewardFeatureConfig(
            min=0.0, max=0.0, enabled=False, default=0.0, calculator_default=0.0
        )

        cfg.position_weight = RewardFeatureConfig(
            min=1.0, max=1.0, enabled=False, default=1.0
        )
        cfg.velocity_weight = RewardFeatureConfig(
            min=1.0, max=1.0, enabled=False, default=1.0
        )
        cfg.heading_weight = RewardFeatureConfig(
            min=1.0, max=1.0, enabled=False, default=1.0
        )

        cfg.position_threshold = RewardFeatureConfig(
            min=2.0, max=2.0, enabled=False, default=2.0
        )
        cfg.velocity_threshold = RewardFeatureConfig(
            min=2.0, max=2.0, enabled=False, default=2.0
        )
        cfg.heading_threshold = RewardFeatureConfig(
            min=0.5, max=0.5, enabled=False, default=0.5
        )

        return cfg

    @property
    def feature_config_list(self) -> list[RewardFeatureConfig]:
        return [
            getattr(self, f.name)
            for f in fields(self)
            if f.type is RewardFeatureConfig and f.name != "width_buffer"
        ]

    @property
    def enabled_feature_config_list(self) -> list[RewardFeatureConfig]:
        feature_config_list = self.feature_config_list
        return [
            feature
            for feature in feature_config_list
            if feature.enabled and feature.as_feature
        ]

    def to_dict(self) -> dict[str, RewardFeatureConfig]:
        """Converts the DomainRandomizationConfig instance to a dictionary."""
        return {
            f.name: getattr(self, f.name)
            for f in fields(self)
            if f.type is RewardFeatureConfig
        }


__all__ = [
    "DomainRandomizationConfig",
]
