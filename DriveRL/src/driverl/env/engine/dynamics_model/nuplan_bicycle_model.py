import math

import torch

from driverl.datatypes.scenario_data import ScenarioData
from driverl.env.engine.dynamics_model.base_dynamics_model import (
    DYNAMICS_MODEL_REGISTER,
    BaseDynamicsModel,
)
from driverl.env.engine.dynamics_model.jerk_bicycle_model import (
    apply_jerk_and_clamp_acceleration,
)
from driverl.utils.geometry import wrap_angle


@DYNAMICS_MODEL_REGISTER.register_module
class NuplanBicycleModel(BaseDynamicsModel):
    """nuPlan-compatible kinematic bicycle model with continuous controls.

    Action format is ``[..., 2] = [longitudinal_jerk, tire_steering_rate]``.
    The jerk is integrated into desired longitudinal acceleration, then the
    acceleration and steering commands are filtered with nuPlan's first-order
    actuator model before kinematic bicycle propagation.
    """

    def __init__(self, config=None):
        super().__init__(config)
        self._accel_time_constant = float(
            getattr(config, "nuplan_bicycle_accel_time_constant", 0.2)
        )
        self._steering_angle_time_constant = float(
            getattr(config, "nuplan_bicycle_steering_angle_time_constant", 0.05)
        )
        self._max_steering_angle = float(
            getattr(config, "nuplan_bicycle_max_steering_angle", math.pi / 3.0)
        )
        self._max_acceleration = float(
            getattr(config, "nuplan_bicycle_max_acceleration", 3.0)
        )
        self._max_steering_rate = float(
            getattr(config, "nuplan_bicycle_max_steering_rate", 0.5)
        )
        self._wheelbase = float(getattr(config, "nuplan_bicycle_wheelbase", 3.089))

    def forward(self, scenario_data: "ScenarioData", actions: torch.Tensor):
        return self._forward_for_duration(
            scenario_data,
            actions,
            duration_seconds=float(self.frame_time_interval),
        )

    def _forward_for_duration(
        self,
        scenario_data: "ScenarioData",
        actions: torch.Tensor,
        *,
        duration_seconds: float,
    ):
        positions = scenario_data.agent_positions_all[:, :, -1:, :]
        velocities = scenario_data.agent_velocity_all[:, :, -1:, :]
        yaws = scenario_data.agent_orientation_all[:, :, -1:]
        wheelbases = torch.full_like(
            scenario_data.agent_size_all[:, :, -1:, 0],
            self._wheelbase,
        )
        accelerations = scenario_data.agent_acceleration_state_all[:, :, -1:]
        steerings = scenario_data.agent_steering_state_all[:, :, -1:]
        yaw_rates = scenario_data.agent_yaw_rate_all[:, :, -1:]

        return self._forward_internal(
            positions=positions,
            velocities=velocities,
            accelerations=accelerations,
            yaws=yaws,
            actions=actions,
            wheelbases=wheelbases,
            steerings=steerings,
            yaw_rates=yaw_rates,
            duration_seconds=duration_seconds,
        )

    def _forward_internal(
        self,
        positions: torch.Tensor,
        velocities: torch.Tensor,
        accelerations: torch.Tensor,
        yaws: torch.Tensor,
        actions: torch.Tensor,
        wheelbases: torch.Tensor,
        steerings: torch.Tensor,
        yaw_rates: torch.Tensor | None = None,
        duration_seconds: float | None = None,
    ):
        if not actions.is_floating_point() or actions.shape[-1] != 2:
            raise ValueError(
                "NuplanBicycleModel only supports continuous actions with "
                "shape [..., 2] = [longitudinal_jerk, tire_steering_rate]."
            )

        t = (
            float(self.frame_time_interval)
            if duration_seconds is None
            else float(duration_seconds)
        )
        wheelbases = torch.full_like(wheelbases, self._wheelbase)
        jerk_long = actions[..., 0:1].clamp(
            self.config.min_jerk_long,
            self.config.max_jerk_long,
        )
        steering_rate_command = actions[..., 1:2]

        ideal_acceleration = apply_jerk_and_clamp_acceleration(
            current_acceleration=accelerations,
            jerk=jerk_long,
            time_interval=t,
            min_acceleration=-self._max_acceleration,
            max_acceleration=self._max_acceleration,
            positive_jerk_limit_when_negative_acc=self.config.max_jerk_long,
            positive_jerk_limit_when_nonnegative_acc=getattr(
                self.config, "positive_jerk_limit_when_nonnegative_acc", 1.0
            ),
        )
        steering_rate_command = steering_rate_command.clamp(
            -self._max_steering_rate,
            self._max_steering_rate,
        )

        speed = velocities[..., 0] * torch.cos(yaws) + velocities[..., 1] * torch.sin(
            yaws
        )

        accel_alpha = t / (t + self._accel_time_constant)
        steering_alpha = t / (t + self._steering_angle_time_constant)
        new_accelerations = accelerations + accel_alpha * (
            ideal_acceleration - accelerations
        )
        would_reverse = speed + new_accelerations * t < 0.0
        stop_acceleration = -speed - 0.75 * accelerations
        new_accelerations = torch.where(
            would_reverse,
            stop_acceleration.clamp(-self._max_acceleration, self._max_acceleration),
            new_accelerations,
        )
        ideal_acceleration = torch.where(
            would_reverse,
            new_accelerations,
            ideal_acceleration,
        )

        ideal_steering = steerings + steering_rate_command * t
        filtered_steerings = steerings + steering_alpha * (ideal_steering - steerings)
        updated_steering_rate = (filtered_steerings - steerings) / t
        new_steerings = filtered_steerings.clamp(
            -self._max_steering_angle,
            self._max_steering_angle,
        )

        x_new = positions[..., 0] + speed * torch.cos(yaws) * t
        y_new = positions[..., 1] + speed * torch.sin(yaws) * t
        new_positions = torch.stack([x_new, y_new], dim=-1)

        delta_yaws = speed * torch.tan(steerings) / wheelbases * t
        new_yaws = wrap_angle(yaws + delta_yaws)

        new_speed = speed + new_accelerations * t
        new_vx = new_speed * torch.cos(new_yaws)
        new_vy = new_speed * torch.sin(new_yaws)
        new_velocities = torch.stack([new_vx, new_vy], dim=-1)

        new_yaw_rates = new_speed * torch.tan(new_steerings) / wheelbases

        if yaw_rates is not None:
            new_yaw_rates = new_yaw_rates.to(dtype=yaw_rates.dtype)

        if self.config and self.config.enable_dynamics_noise:
            (
                new_positions,
                new_velocities,
                new_yaws,
                new_accelerations,
                new_steerings,
                new_yaw_rates,
            ) = self.apply_dynamics_noise(
                new_positions,
                new_velocities,
                new_yaws,
                new_accelerations,
                new_steerings,
                new_yaw_rates,
            )

        return (
            new_positions,
            new_velocities,
            new_yaws,
            new_accelerations,
            ideal_acceleration,
            new_steerings,
            updated_steering_rate,
            new_yaw_rates,
            steering_rate_command,
            jerk_long,
        )

    def inverse(self, scenario_data: "ScenarioData"):
        raise NotImplementedError("NuplanBicycleModel only supports forward rollout.")
