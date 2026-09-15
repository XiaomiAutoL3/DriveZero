import abc
from enum import Enum, unique

from driverl.datatypes.scenario_data import ScenarioData
from driverl.env.engine.config import EngineMergedConfig
from driverl.utils.registry import Registry

__all__ = [
    "BaseRewardCalculator",
    "InfoDimension",
    "REWARD_CALCULATOR_REGISTER",
]


REWARD_CALCULATOR_REGISTER = Registry("reward_calculator")


@unique
class InfoDimension(Enum):
    """Dimensions of the `(num_envs, num_agents, size)` info tensor returned by the engine."""

    #: Collision flag (1 if collision, 0 otherwise).
    COLLISION_FLAG = 0
    #: Goal reached flag (1 if goal reached this step, 0 otherwise).
    GOAL_REACHED_FLAG = 1
    #: Road event: 0 = none, 1 = off-road, 2 = crossed solid lane, 3 = wrong-way, 4 = severe wrong-way.
    OFFROAD_EVENT = 2
    #: TTC (Time-To-Collision): seconds until ego hits another agent (both at constant velocity).
    TTC_ALERT = 3
    #: TTO (Time-To-Offroad): seconds until ego hits a road boundary (curb), constant-curvature motion.
    TTO_ALERT = 4
    #: Lateral deviation from the nearest centerline segment (meters).
    CENTERLINE_DEVIATION = 5
    #: Flag indicating the agent is too close to curb boundary.
    CURB_TOO_CLOSE = 6
    #: Overspeed flag (1 if exceeding the allowable speed, 0 otherwise).
    OVERSPEED_FLAG = 7
    #: Longitudinal acceleration (m/s²).
    LONGITUDINAL_ACCEL = 8
    #: Lateral acceleration (m/s²).
    LATERAL_ACCEL = 9
    #: Longitudinal jerk (m/s³).
    LONGITUDINAL_JERK = 10
    #: Lateral jerk (m/s³).
    LATERAL_JERK = 11
    #: Absolute position error to log trajectory (meters).
    SIMILARITY_POSITION_ERROR = 12
    #: Absolute velocity error to log trajectory (m/s).
    SIMILARITY_VELOCITY_ERROR = 13
    #: Heading error to log trajectory (radians).
    SIMILARITY_HEADING_ERROR = 14
    #: Change in route progress since previous step (meters).
    ROUTE_PROGRESS_DELTA = 15

    #: Longitudinal velocity in the vehicle body frame (m/s).
    V_LONG = 16
    #: Lateral velocity in the vehicle body frame (m/s).
    V_LAT = 17
    #: Current yaw (orientation) of the agent (radians).
    YAW = 18
    #: Current yaw rate (rad/s) of the agent.
    YAW_RATE = 19
    #: Current steering wheel angle command (degrees).
    WHEEL_ANGLE = 20
    #: Rate of change of the steering wheel angle (degrees/s).
    WHEEL_RATE = 21

    COLLISION_REWARD = 22
    OFFROAD_REWARD = 23
    GOAL_REACHED_REWARD = 24
    TTC_REWARD = 25
    TTO_REWARD = 26
    CENTERLINE_REWARD = 27
    COMFORT_REWARD = 28
    SIMILARITY_REWARD = 29
    ROUTE_PROGRESS_REWARD = 30
    OVERSPEED_REWARD = 31
    CURB_CLEARANCE_REWARD = 32
    #: Collision responsibility (0-1 range).
    COLLISION_RESPONSIBILITY = 33
    COLLISION_POINT_RESPONSIBILITY = 34
    COLLISION_LANE_RESPONSIBILITY = 35
    COLLISION_RELATIVE_SPEED_RESPONSIBILITY = 36
    OCC_HIT = 37
    #: TTG (Time-To-General-object): seconds until ego hits a static occupancy obstacle, constant-curvature.
    TTG_ALERT = 38
    TTG_REWARD = 39
    #: Lane-change flag (1 if a lane-change crossing is detected, 0 otherwise).
    LANE_CHANGE_INFO = 40
    #: Cross-lane soft reward score.
    CROSS_LANE_REWARD = 41
    #: Comfort sub-reward: longitudinal jerk.
    COMFORT_JERK_LONG_REWARD = 42
    #: Comfort sub-reward: longitudinal acceleration.
    COMFORT_A_LONG_REWARD = 43
    #: Comfort sub-reward: lateral jerk.
    COMFORT_JERK_LAT_REWARD = 44
    #: Comfort sub-reward: lateral acceleration.
    COMFORT_A_LAT_REWARD = 45
    #: Comfort sub-reward: steering rate.
    COMFORT_STEERING_RATE_REWARD = 46
    #: Time-to-stopped estimation (ego straight, NPC stationary).
    TTS_ALERT = 47
    #: Time-to-stopped reward.
    TTS_REWARD = 48
    COLLISION_RESPONSIBILITY_OTHER = 49
    COLLISION_POINT_RESPONSIBILITY_OTHER = 50
    COLLISION_LANE_RESPONSIBILITY_OTHER = 51
    #: TTC collision responsibility (0-1 range).
    TTC_RESPONSIBILITY = 52
    #: Rate of change of the steering wheel angle rate (degrees/s²).
    WHEEL_ACCEL = 53
    #: Comfort sub-reward: steering acceleration.
    COMFORT_STEERING_ACCEL_REWARD = 54
    #: NAVSIM-compatible on-route red connector intersection flag.
    TRAFFIC_LIGHT_VIOLATION = 55
    #: Hard reward contributed by a traffic-light violation.
    TRAFFIC_LIGHT_REWARD = 56
    #: Index of the first intersecting route connector, or -1 when none.
    TRAFFIC_LIGHT_CONNECTOR_INDEX = 57
    #: nuPlan ID of the first intersecting route connector, or -1 when none.
    TRAFFIC_LIGHT_CONNECTOR_ID = 58

    @classmethod
    def size(cls) -> int:
        """Return the total number of info dimensions."""
        return max(d.value for d in cls) + 1

    @classmethod
    def info_dimensions(cls):
        """Return members in display order."""
        return sorted(cls, key=lambda item: item.value)

    def component_id(self) -> str:
        """Stable identifier derived from the label."""
        return _INFO_COMPONENT_IDS.get(self.name, self.name.lower())

    def value_type(self) -> str:
        """Semantic type of the value ('boolean', 'category', 'numeric')."""
        return _INFO_VALUE_TYPES.get(self.name, "numeric")


_INFO_COMPONENT_IDS = {
    "COLLISION_FLAG": "collision",
    "COLLISION_RESPONSIBILITY": "collision_responsibility",
    "GOAL_REACHED_FLAG": "goal_reached",
    "OFFROAD_EVENT": "off_road_event",
    "TTC_ALERT": "ttc_alert",
    "TTO_ALERT": "tto_alert",
    "TTG_ALERT": "ttg_alert",
    "CENTERLINE_DEVIATION": "centerline_deviation_m",
    "CURB_TOO_CLOSE": "curb_too_close",
    "OVERSPEED_FLAG": "overspeed",
    "LONGITUDINAL_ACCEL": "longitudinal_accel",
    "LATERAL_ACCEL": "lateral_accel",
    "LONGITUDINAL_JERK": "longitudinal_jerk",
    "LATERAL_JERK": "lateral_jerk",
    "SIMILARITY_POSITION_ERROR": "position_error_m",
    "SIMILARITY_VELOCITY_ERROR": "velocity_error_m_per_s",
    "SIMILARITY_HEADING_ERROR": "heading_error_rad",
    "ROUTE_PROGRESS_DELTA": "route_progress_delta_m",
    "V_LONG": "v_long",
    "V_LAT": "v_lat",
    "YAW": "yaw",
    "YAW_RATE": "yaw_rate",
    "WHEEL_ANGLE": "wheel_deg",
    "WHEEL_RATE": "wheel_rate",
    "WHEEL_ACCEL": "wheel_accel",
    "COLLISION_REWARD": "collision_reward",
    "OFFROAD_REWARD": "offroad_reward",
    "GOAL_REACHED_REWARD": "goal_reached_reward",
    "TTC_REWARD": "ttc_reward",
    "TTO_REWARD": "tto_reward",
    "TTG_REWARD": "ttg_reward",
    "CROSS_LANE_REWARD": "cross_lane_reward",
    "CENTERLINE_REWARD": "centerline_reward",
    "COMFORT_REWARD": "comfort_reward",
    "SIMILARITY_REWARD": "similarity_reward",
    "ROUTE_PROGRESS_REWARD": "route_progress_reward",
    "OVERSPEED_REWARD": "overspeed_reward",
    "COLLISION_POINT_RESPONSIBILITY": "collision_point_responsibility",
    "COLLISION_LANE_RESPONSIBILITY": "collision_lane_responsibility",
    "COLLISION_RESPONSIBILITY_OTHER": "collision_responsibility_other",
    "COLLISION_POINT_RESPONSIBILITY_OTHER": ("collision_point_responsibility_other"),
    "COLLISION_LANE_RESPONSIBILITY_OTHER": ("collision_lane_responsibility_other"),
    "COLLISION_RELATIVE_SPEED_RESPONSIBILITY": (
        "collision_relative_speed_responsibility"
    ),
    "OCC_HIT": "occ_hit",
    "LANE_CHANGE_INFO": "lane_change_info",
    "COMFORT_JERK_LONG_REWARD": "comfort_reward_jerk_long",
    "COMFORT_A_LONG_REWARD": "comfort_reward_a_long",
    "COMFORT_JERK_LAT_REWARD": "comfort_reward_jerk_lat",
    "COMFORT_A_LAT_REWARD": "comfort_reward_a_lat",
    "COMFORT_STEERING_RATE_REWARD": "comfort_reward_steering_rate",
    "COMFORT_STEERING_ACCEL_REWARD": "comfort_reward_steering_accel",
    "TTS_ALERT": "tts_alert",
    "TTS_REWARD": "tts_reward",
    "TTC_RESPONSIBILITY": "ttc_responsibility",
    "TRAFFIC_LIGHT_VIOLATION": "traffic_light_violation",
    "TRAFFIC_LIGHT_REWARD": "traffic_light_reward",
    "TRAFFIC_LIGHT_CONNECTOR_INDEX": "traffic_light_connector_index",
    "TRAFFIC_LIGHT_CONNECTOR_ID": "traffic_light_connector_id",
}

_INFO_VALUE_TYPES = {
    "COLLISION_FLAG": "boolean",
    "COLLISION_RESPONSIBILITY": "numeric",
    "GOAL_REACHED_FLAG": "boolean",
    "OFFROAD_EVENT": "category",
    "TTC_ALERT": "numeric",
    "TTO_ALERT": "numeric",
    "TTG_ALERT": "numeric",
    "OVERSPEED_FLAG": "boolean",
    "CURB_TOO_CLOSE": "boolean",
    "LONGITUDINAL_ACCEL": "numeric",
    "LATERAL_ACCEL": "numeric",
    "LONGITUDINAL_JERK": "numeric",
    "LATERAL_JERK": "numeric",
    "V_LONG": "numeric",
    "V_LAT": "numeric",
    "YAW": "numeric",
    "YAW_RATE": "numeric",
    "WHEEL_ANGLE": "numeric",
    "WHEEL_RATE": "numeric",
    "WHEEL_ACCEL": "numeric",
    "COLLISION_REWARD": "numeric",
    "OFFROAD_REWARD": "numeric",
    "GOAL_REACHED_REWARD": "numeric",
    "TTC_REWARD": "numeric",
    "TTO_REWARD": "numeric",
    "TTG_REWARD": "numeric",
    "CROSS_LANE_REWARD": "numeric",
    "CENTERLINE_REWARD": "numeric",
    "COMFORT_REWARD": "numeric",
    "SIMILARITY_REWARD": "numeric",
    "ROUTE_PROGRESS_REWARD": "numeric",
    "OVERSPEED_REWARD": "numeric",
    "COLLISION_POINT_RESPONSIBILITY": "numeric",
    "COLLISION_LANE_RESPONSIBILITY": "numeric",
    "COLLISION_RESPONSIBILITY_OTHER": "numeric",
    "COLLISION_POINT_RESPONSIBILITY_OTHER": "numeric",
    "COLLISION_LANE_RESPONSIBILITY_OTHER": "numeric",
    "COLLISION_RELATIVE_SPEED_RESPONSIBILITY": "numeric",
    "OCC_HIT": "boolean",
    "LANE_CHANGE_INFO": "boolean",
    "COMFORT_JERK_LONG_REWARD": "numeric",
    "COMFORT_A_LONG_REWARD": "numeric",
    "COMFORT_JERK_LAT_REWARD": "numeric",
    "COMFORT_A_LAT_REWARD": "numeric",
    "COMFORT_STEERING_RATE_REWARD": "numeric",
    "COMFORT_STEERING_ACCEL_REWARD": "numeric",
    "TTS_ALERT": "numeric",
    "TTS_REWARD": "numeric",
    "TTC_RESPONSIBILITY": "numeric",
    "TRAFFIC_LIGHT_VIOLATION": "boolean",
    "TRAFFIC_LIGHT_REWARD": "numeric",
    "TRAFFIC_LIGHT_CONNECTOR_INDEX": "numeric",
    "TRAFFIC_LIGHT_CONNECTOR_ID": "numeric",
}


class BaseRewardCalculator(abc.ABC):
    """Base class for reward calculators."""

    def __init__(self, config: EngineMergedConfig):
        """Initialize the reward calculator."""
        self.config = config

    @staticmethod
    def reward_calculator_factory(
        model_name: str, *args, **kwargs
    ) -> "BaseRewardCalculator":
        """Factory method for reward calculators."""
        reward_calculator_name = Registry.get_model_instance_name(model_name)
        supported_reward_calculator_names = REWARD_CALCULATOR_REGISTER.module_keys
        assert reward_calculator_name in supported_reward_calculator_names, (
            f"Current only support {supported_reward_calculator_names}, but got {reward_calculator_name}."
        )
        return REWARD_CALCULATOR_REGISTER.get(reward_calculator_name)(*args, **kwargs)

    def forward(
        self,
        scenario_data: "ScenarioData",
        log_scenario_data: "ScenarioData",
        rewards_and_infos: dict,
        **kwargs,
    ):
        """
        Computes the reward for the current state.

        Args:
            scenario_data (ScenarioData): The current state of the environment.
            log_scenario_data (ScenarioData): The logged scenario data.
            rewards_and_infos (dict): A dictionary to store the rewards and infos.
            **kwargs: Additional arguments that might be needed by specific reward calculators.
        """
        raise NotImplementedError
