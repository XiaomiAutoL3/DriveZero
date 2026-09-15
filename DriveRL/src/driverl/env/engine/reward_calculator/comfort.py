import torch
import torch.nn.functional as F

from driverl.datatypes.scenario_data import ScenarioData
from driverl.env.engine.config import EngineMergedConfig
from driverl.env.engine.reward_calculator.base_reward_calculator import (
    REWARD_CALCULATOR_REGISTER,
    BaseRewardCalculator,
)


def _savgol_weights(
    window_length: int, derivative: int, positions: torch.Tensor
) -> tuple[tuple[float, ...], ...]:
    """Build quadratic Savitzky-Golay weights once, outside the rollout."""
    x = torch.arange(window_length, dtype=torch.float64)
    design = torch.stack((torch.ones_like(x), x, x.square()), dim=1)
    inverse = torch.linalg.pinv(design)
    if derivative == 0:
        basis = torch.stack(
            (torch.ones_like(positions), positions, positions.square()), dim=1
        )
    else:
        basis = torch.stack(
            (torch.zeros_like(positions), torch.ones_like(positions), 2 * positions),
            dim=1,
        )
    return tuple(tuple(row) for row in (basis @ inverse).tolist())


def _savgol_spec(window_length: int, derivative: int) -> tuple:
    half = window_length // 2
    center = torch.tensor([(window_length - 1) / 2], dtype=torch.float64)
    left = torch.arange(half, dtype=torch.float64)
    right = torch.arange(window_length - half, window_length, dtype=torch.float64)
    return (
        window_length,
        half,
        _savgol_weights(window_length, derivative, center)[0],
        _savgol_weights(window_length, derivative, left),
        _savgol_weights(window_length, derivative, right),
    )


_ACCELERATION_SAVGOL_SPEC = _savgol_spec(8, 0)
_JERK_SAVGOL_SPEC = _savgol_spec(15, 1)


@REWARD_CALCULATOR_REGISTER.register_module
class Comfort(BaseRewardCalculator):
    def __init__(self, config: EngineMergedConfig):
        """Initialize the comfort reward calculator."""
        super().__init__(config)
        self.num_steps = config.num_steps
        self.init_steps = config.init_steps
        self.time_interval = config.frame_time_interval
        self.min_jerk_long = config.min_jerk_long
        self.min_a_long = config.min_a_long
        self.a_lat_comfort_low_speed = config.a_lat_comfort_low_speed
        self.a_lat_comfort_high_speed = config.a_lat_comfort_high_speed
        self.a_lat_comfort_threshold_low = config.a_lat_comfort_threshold_low
        self.a_lat_comfort_threshold_high = config.a_lat_comfort_threshold_high
        self.a_long_decel_full_speed = config.a_long_decel_full_speed
        self.a_long_decel_floor = config.a_long_decel_floor
        self.nuplan_min_lon_accel = config.nuplan_comfort_min_lon_accel
        self.nuplan_max_lon_accel = config.nuplan_comfort_max_lon_accel
        self.nuplan_max_abs_lat_accel = config.nuplan_comfort_max_abs_lat_accel
        self.nuplan_max_abs_lon_jerk = config.nuplan_comfort_max_abs_lon_jerk
        self.nuplan_max_abs_mag_jerk = config.nuplan_comfort_max_abs_mag_jerk
        self.nuplan_max_abs_yaw_rate = config.nuplan_comfort_max_abs_yaw_rate
        self.nuplan_max_abs_yaw_accel = config.nuplan_comfort_max_abs_yaw_accel
        self.nuplan_rear_axle_to_center = config.nuplan_bicycle_rear_axle_to_center

    def forward(
        self,
        scenario_data: "ScenarioData",
        log_scenario_data: "ScenarioData",
        rewards_and_infos: dict,
        **kwargs,
    ):
        """
        Computes the off road reward and goal reached information.

        Args:
            scenario_data (ScenarioData): The current state of the environment.
            log_scenario_data (ScenarioData): The logged scenario data.
            rewards_and_infos (dict): A dictionary to store the rewards and infos.
            **kwargs: Additional arguments.
        """
        # Get current positions of all agents

        (
            jerk_long,
            a_long,
            jerk_lat,
            a_lat,
            steering_rate,
            steering_accel,
            steering,
            comfort_weight,
            speed,
            jerk_mag,
            yaw_rate,
            yaw_accel,
            max_abs_jerk_long,
            max_abs_jerk_mag,
        ) = self.data_preprocessing(scenario_data)
        (
            comfort_score,
            jerk_long_reward,
            a_long_reward,
            jerk_lat_reward,
            a_lat_reward,
            steering_rate_reward,
            steering_accel_reward,
            comfort_within_bounds,
        ) = self.calculate_comfort(
            jerk_long,
            a_long,
            jerk_lat,
            a_lat,
            steering_rate,
            steering_accel,
            comfort_weight,
            speed,
            jerk_mag,
            yaw_rate,
            yaw_accel,
            max_abs_jerk_long,
            max_abs_jerk_mag,
        )

        controlled_mask = scenario_data.agent_control_manager.controlled_mask  # [N, A]
        triggered = controlled_mask & ~comfort_within_bounds
        comfort_score = self._apply_triggered_penalty(
            scenario_data, comfort_score, triggered
        )
        comfort_score = torch.where(controlled_mask, comfort_score, 0.0)

        longitudinal_accel = torch.where(
            controlled_mask, a_long, torch.zeros_like(a_long)
        )
        lateral_accel = torch.where(controlled_mask, a_lat, torch.zeros_like(a_lat))
        longitudinal_jerk = torch.where(
            controlled_mask, jerk_long, torch.zeros_like(jerk_long)
        )
        lateral_jerk = torch.where(
            controlled_mask, jerk_lat, torch.zeros_like(jerk_lat)
        )

        comfort_info = torch.stack(
            [
                longitudinal_accel,
                lateral_accel,
                longitudinal_jerk,
                lateral_jerk,
                jerk_long_reward,
                a_long_reward,
                jerk_lat_reward,
                a_lat_reward,
                steering_rate_reward,
                steering_accel_reward,
            ],
            dim=-1,
        )

        reward = comfort_score.float()

        rewards_and_infos["Comfort"] = {"reward": reward, "info": comfort_info}

    @staticmethod
    def _apply_triggered_penalty(
        scenario_data: "ScenarioData",
        comfort_score: torch.Tensor,
        triggered: torch.Tensor,
    ) -> torch.Tensor:
        prev_triggered = getattr(scenario_data, "_comfort_triggered", None)
        if prev_triggered is None or prev_triggered.shape != triggered.shape:
            prev_triggered = torch.zeros_like(triggered)
        comfort_score = comfort_score * torch.where(
            prev_triggered,
            comfort_score.new_tensor(0.75),
            comfort_score.new_tensor(1.0),
        )
        scenario_data._comfort_triggered = prev_triggered | triggered
        return comfort_score

    def calculate_comfort(
        self,
        jerk_long,
        a_long,
        jerk_lat,
        a_lat,
        steering_rate,
        steering_accel,
        comfort_weight,
        speed,
        jerk_mag=None,
        yaw_rate=None,
        yaw_accel=None,
        max_abs_jerk_long=None,
        max_abs_jerk_mag=None,
    ):
        """
        Calculate per-frame nuPlan-style comfort gates.

        Each component returns 1 when it is inside the nuPlan comfort bound and
        `comfort_weight` otherwise. The combined comfort score is the minimum
        over all components, so any failed bound lowers the frame score.
        """
        if jerk_mag is None:
            jerk_mag = torch.linalg.norm(
                torch.stack([jerk_long, jerk_lat], dim=-1), dim=-1
            )
        if yaw_rate is None:
            yaw_rate = torch.zeros_like(a_long)
        if yaw_accel is None:
            yaw_accel = torch.zeros_like(a_long)
        if max_abs_jerk_long is None:
            max_abs_jerk_long = torch.abs(jerk_long)
        if max_abs_jerk_mag is None:
            max_abs_jerk_mag = torch.abs(jerk_mag)

        def gate(condition: torch.Tensor) -> torch.Tensor:
            return torch.where(
                condition, torch.ones_like(comfort_weight), comfort_weight
            )

        jerk_long_ok = max_abs_jerk_long < self.nuplan_max_abs_lon_jerk
        a_long_ok = (a_long > self.nuplan_min_lon_accel) & (
            a_long < self.nuplan_max_lon_accel
        )
        jerk_mag_ok = max_abs_jerk_mag < self.nuplan_max_abs_mag_jerk
        a_lat_ok = torch.abs(a_lat) < self.nuplan_max_abs_lat_accel
        yaw_rate_ok = torch.abs(yaw_rate) < self.nuplan_max_abs_yaw_rate
        yaw_accel_ok = torch.abs(yaw_accel) < self.nuplan_max_abs_yaw_accel

        jerk_long_reward = gate(jerk_long_ok)
        a_long_reward = gate(a_long_ok)
        jerk_lat_reward = gate(jerk_mag_ok)
        a_lat_reward = gate(a_lat_ok)
        steering_rate_reward = gate(yaw_rate_ok)
        steering_accel_reward = gate(yaw_accel_ok)
        comfort_within_bounds = (
            jerk_long_ok
            & a_long_ok
            & jerk_mag_ok
            & a_lat_ok
            & yaw_rate_ok
            & yaw_accel_ok
        )
        comfort_reward = torch.min(
            torch.stack(
                [
                    jerk_long_reward,
                    a_long_reward,
                    jerk_lat_reward,
                    a_lat_reward,
                    steering_rate_reward,
                    steering_accel_reward,
                ],
                dim=-1,
            ),
            dim=-1,
        ).values

        return (
            comfort_reward,
            jerk_long_reward,
            a_long_reward,
            jerk_lat_reward,
            a_lat_reward,
            steering_rate_reward,
            steering_accel_reward,
            comfort_within_bounds,
        )

    @staticmethod
    def _calculate_component_reward(value, min_threshold, max_threshold, weight):
        """
        Calculate reward for a single comfort component using a clipped saturating curve.

        - Reward peaks at 1 when `value` is 0.
        - Penalizes magnitude smoothly within threshold.
        - Once |value| exceeds threshold, reward is clipped to threshold value.
        """

        # Determine scale based on sign (use positive magnitude for negative thresholds).
        scale = torch.where(value >= 0, max_threshold, -min_threshold).clamp(min=1e-6)
        scaled_abs = torch.abs(value) / scale
        scaled_abs = scaled_abs.clamp(max=1.0)

        # Saturating penalty: smooth near 0, then clipped at threshold.
        penalty = (scaled_abs * scaled_abs) / (1.0 + scaled_abs * scaled_abs)
        result = 1.0 - (1.0 - weight) * penalty

        return result

    @staticmethod
    def _calculate_asymmetric_long_reward(
        value, min_threshold, max_threshold, weight, dead_zone=0.8
    ):
        """Asymmetric longitudinal reward: convex (x²) for positive, S-shaped for negative.

        Deceleration (value < 0) uses an S-shaped penalty:
        - |value| < dead_zone: near-zero penalty (normal braking is free)
        - dead_zone to mid_zone (4.0): steep penalty ramp (uncomfortable territory)
        - mid_zone to threshold: penalty growth tapers off (emergency allowed)
        Acceleration (value >= 0) uses the standard convex (x²/(1+x²)) curve.
        """
        pos_scale = torch.as_tensor(
            max_threshold, device=value.device, dtype=value.dtype
        ).clamp(min=1e-6)
        neg_scale = torch.as_tensor(
            -min_threshold, device=value.device, dtype=value.dtype
        ).clamp(min=1e-6)
        abs_val = torch.abs(value)

        # Positive side: convex (standard)
        pos_x = (abs_val / pos_scale).clamp(max=1.0)
        pos_penalty = (pos_x * pos_x) / (1.0 + pos_x * pos_x)

        # Negative side: S-shaped via smoothstep
        dead_frac = dead_zone / neg_scale
        mid_frac = 4.0 / neg_scale  # maps to |value| = 4
        neg_x = abs_val / neg_scale
        # Remap [dead_frac, mid_frac] → [0, 1], clamp outside
        t = ((neg_x - dead_frac) / (mid_frac - dead_frac)).clamp(0.0, 1.0)
        # sin curve: rises steeply in the first half, tapers in the second
        s = torch.sin(t * (torch.pi / 2.0))
        # Blend S-curve with a linear ramp to give non-zero slope in dead zone
        # and continued growth past mid_zone.
        neg_penalty = 0.8 * s + 0.2 * neg_x.clamp(max=1.0)

        penalty = torch.where(value >= 0, pos_penalty, neg_penalty)
        return 1.0 - (1.0 - weight) * penalty

    @staticmethod
    def _calculate_lateral_reward(
        value, min_threshold, max_threshold, weight, dead_zone=0.5
    ):
        """Symmetric lateral reward with dead zone and S-shaped penalty.

        Uses the same S-shaped curve for both positive and negative values
        (lateral discomfort is direction-agnostic), but adds a dead zone so
        steady-state cornering within the dead zone is nearly free.

        - |value| < dead_zone: near-zero penalty
        - dead_zone to threshold: steep S-shaped ramp via sin smoothstep
        - beyond threshold: clipped at max penalty
        """
        max_t = torch.as_tensor(max_threshold, device=value.device, dtype=value.dtype)
        min_t = torch.as_tensor(-min_threshold, device=value.device, dtype=value.dtype)
        scale = torch.where(value >= 0, max_t, min_t).clamp(min=1e-6)
        abs_val = torch.abs(value)
        x = (abs_val / scale).clamp(max=1.0)

        dead_frac = (
            torch.as_tensor(dead_zone, device=value.device, dtype=value.dtype) / scale
        )
        # Remap [dead_frac, 1] -> [0, 1], clamp outside
        t = ((x - dead_frac) / (1.0 - dead_frac).clamp(min=1e-6)).clamp(0.0, 1.0)
        # S-shaped via sin: rises steeply in first half, tapers in second
        s = torch.sin(t * (torch.pi / 2.0))
        # Blend S-curve with small linear ramp to avoid zero gradient in dead zone
        penalty = 0.8 * s + 0.2 * x
        return 1.0 - (1.0 - weight) * penalty

    @staticmethod
    def _calculate_steering_rate_reward(value, min_threshold, max_threshold, weight):
        """
        Calculate steering-rate reward using a linear-cosine mixed decay.

        - Reward peaks at 1 when `value` is 0.
        - Decays to `weight` at min/max thresholds.
        - Returns `weight` for values beyond thresholds.
        """

        scale = torch.where(value >= 0, max_threshold, -min_threshold).clamp(min=1e-6)
        scaled_abs = (torch.abs(value) / scale).clamp(max=1.0)
        linear_mix = 0.5

        cosine_decay = 0.5 * (1.0 - torch.cos(scaled_abs * torch.pi))
        mixed_decay = linear_mix * scaled_abs + (1.0 - linear_mix) * cosine_decay
        result = weight + (1.0 - weight) * (1.0 - mixed_decay)

        return result

    def data_preprocessing(self, scenario_data, time_interval=None):
        if time_interval is None:
            time_interval = self.time_interval
        randomized_features = scenario_data.randomized_features
        comfort_weight = randomized_features.get("comfort_weight", calculate=True)

        velocities = scenario_data.agent_velocity_all[:, :, -1, :]
        speed = torch.linalg.norm(velocities, dim=-1)
        a_lat = torch.zeros_like(speed)
        # Keep the final logged frame as a derivative anchor.  The first frame
        # after ``init_steps`` is the first simulated frame, so dropping the
        # anchor would make its jerk/rates spuriously zero (and diverge from
        # Engine.step).  Peak comfort statistics below exclude this anchor.
        history = slice(self.init_steps, None)
        # Index zero is the final observed frame retained as a derivative
        # anchor. Peak rollout statistics must start at the first simulated
        # frame.
        rollout_start = 1
        rear_a_long_history = scenario_data.agent_acceleration_state_all[:, :, history]
        rear_a_long = rear_a_long_history[:, :, -1]

        jerk_lat = scenario_data.agent_jerk_lat_all[:, :, history][:, :, -1]
        if scenario_data.agent_yaw_rate_all.numel() > 0:
            yaw_rate_history = scenario_data.agent_yaw_rate_all[:, :, history]
            yaw_rate = yaw_rate_history[:, :, -1]
        else:
            yaw_rate_history = torch.zeros_like(rear_a_long_history)
            yaw_rate = torch.zeros_like(rear_a_long)
        if yaw_rate_history.shape[-1] >= 2:
            yaw_accel_history = torch.zeros_like(yaw_rate_history)
            yaw_accel_history[:, :, 1:] = (
                yaw_rate_history[:, :, 1:] - yaw_rate_history[:, :, :-1]
            ) / time_interval
            yaw_accel_history[:, :, 0] = yaw_accel_history[:, :, 1]
            yaw_accel = yaw_accel_history[:, :, -1]
        else:
            yaw_accel_history = torch.zeros_like(yaw_rate_history)
            yaw_accel = torch.zeros_like(yaw_rate)
        a_long_history = rear_a_long_history + self.nuplan_rear_axle_to_center * (
            yaw_rate_history.square() + yaw_accel_history
        )
        if a_long_history.shape[-1] >= 15:
            acceleration_history = torch.stack(
                (a_long_history, rear_a_long_history.abs()), dim=2
            )
            smoothed_acceleration = self._savgol_filter(
                acceleration_history, _ACCELERATION_SAVGOL_SPEC
            )
            jerk_history = self._savgol_filter(
                smoothed_acceleration, _JERK_SAVGOL_SPEC, delta=time_interval
            )
            a_long = smoothed_acceleration[:, :, 0, -1]
            jerk_long_history = jerk_history[:, :, 0]
            jerk_mag_history = jerk_history[:, :, 1]
            jerk_long = jerk_long_history[:, :, -1]
            jerk_mag = jerk_mag_history[:, :, -1]
            peak_jerk_long_history = jerk_long_history[:, :, rollout_start:]
            peak_jerk_mag_history = jerk_mag_history[:, :, rollout_start:]
            max_abs_jerk_long = peak_jerk_long_history.abs().amax(dim=-1)
            max_abs_jerk_mag = peak_jerk_mag_history.abs().amax(dim=-1)
        else:
            a_long = a_long_history[:, :, -1]
            if a_long_history.shape[-1] >= 2:
                jerk_long = (
                    a_long_history[:, :, -1] - a_long_history[:, :, -2]
                ) / time_interval
            else:
                jerk_long = scenario_data.agent_jerk_long_all[:, :, history][:, :, -1]
            jerk_mag = torch.linalg.norm(
                torch.stack([jerk_long, jerk_lat], dim=-1), dim=-1
            )
            max_abs_jerk_long = jerk_long.abs()
            max_abs_jerk_mag = jerk_mag.abs()

        steering_history = scenario_data.agent_steering_state_all[:, :, history]
        _, _, T = steering_history.shape
        if T >= 2:
            # Recalculate steering rate from steering state.
            steerings = steering_history[:, :, -1]
            pre_steerings = steering_history[:, :, -2]
            steering_rate = (steerings - pre_steerings) / time_interval
        else:
            steerings = steering_history[:, :, -1]
            steering_rate = torch.zeros_like(steerings)
        if T >= 3:
            pre_pre_steerings = steering_history[:, :, -3]
            pre_steering_rate = (pre_steerings - pre_pre_steerings) / time_interval
            steering_accel = (steering_rate - pre_steering_rate) / time_interval
        else:
            steering_accel = torch.zeros_like(steerings)

        return (
            jerk_long,
            a_long,
            jerk_lat,
            a_lat,
            steering_rate,
            steering_accel,
            steerings,
            comfort_weight,
            speed,
            jerk_mag,
            yaw_rate,
            yaw_accel,
            max_abs_jerk_long,
            max_abs_jerk_mag,
        )

    @staticmethod
    def _savgol_filter(
        values: torch.Tensor, spec: tuple, delta: float = 1.0
    ) -> torch.Tensor:
        """Torch equivalent of nuPlan's quadratic scipy.signal.savgol_filter."""
        window_length, half, center_values, left_values, right_values = spec
        if values.shape[-1] < window_length:
            return values

        center = values.new_tensor(center_values) / delta
        left = values.new_tensor(left_values) / delta
        right = values.new_tensor(right_values) / delta
        flat = values.reshape(-1, 1, values.shape[-1])
        valid = F.conv1d(flat, center.reshape(1, 1, -1)).reshape(*values.shape[:-1], -1)

        valid_start = (window_length - 1) // 2
        middle_length = values.shape[-1] - 2 * half
        middle_offset = half - valid_start
        middle = valid[..., middle_offset : middle_offset + middle_length]
        left_edge = values[..., :window_length] @ left.T
        right_edge = values[..., -window_length:] @ right.T
        return torch.cat((left_edge, middle, right_edge), dim=-1)
