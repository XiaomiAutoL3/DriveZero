"""nuPlan ego controller that executes DriveRL one-step actions."""

from __future__ import annotations

from typing import Any, Protocol

import torch

from driverl.env.engine.dynamics_model import BaseDynamicsModel
from driverl.nuplan.config import DriveRLNuPlanControllerConfig
from driverl.nuplan.state_conversion import driverl_state_to_ego_state
from driverl.nuplan.trajectory import (
    DriveRLActionTrajectory,
    nuplan_bicycle_action_targets,
)

try:
    from nuplan.common.actor_state.dynamic_car_state import DynamicCarState
    from nuplan.common.actor_state.ego_state import EgoState
    from nuplan.common.actor_state.state_representation import (
        StateVector2D,
        TimeDuration,
    )
    from nuplan.planning.simulation.controller.abstract_controller import (
        AbstractEgoController,
    )
    from nuplan.planning.simulation.controller.motion_model.kinematic_bicycle import (
        KinematicBicycleModel,
    )
    from nuplan.planning.simulation.simulation_time_controller.simulation_iteration import (
        SimulationIteration,
    )
    from nuplan.planning.simulation.trajectory.abstract_trajectory import (
        AbstractTrajectory,
    )
except ImportError as exc:  # pragma: no cover - import guard for non-nuPlan envs.
    raise ImportError(
        "driverl.nuplan.controller requires nuPlan to be installed."
    ) from exc


class _ScenarioLike(Protocol):
    @property
    def initial_ego_state(self) -> EgoState: ...


class DriveRLOneStageController(AbstractEgoController):
    """Propagate ego state directly from a ``DriveRLActionTrajectory`` action."""

    def __init__(
        self,
        scenario: _ScenarioLike,
        config: DriveRLNuPlanControllerConfig | None = None,
        **config_overrides: Any,
    ) -> None:
        self._scenario = scenario
        base_config = config or DriveRLNuPlanControllerConfig()
        for key, value in config_overrides.items():
            if not hasattr(base_config, key):
                raise TypeError(f"Unknown DriveRLOneStageController config key: {key}")
            setattr(base_config, key, value)
        self._config = base_config
        self._device = torch.device(self._config.device)
        self._dynamics_model = None
        self._motion_model = None
        if self._config.dynamics_model == "nuplan_bicycle_model":
            self._motion_model = KinematicBicycleModel(
                vehicle=scenario.initial_ego_state.car_footprint.vehicle_parameters,
                max_steering_angle=self._config.nuplan_bicycle_max_steering_angle,
                accel_time_constant=self._config.nuplan_bicycle_accel_time_constant,
                steering_angle_time_constant=self._config.nuplan_bicycle_steering_angle_time_constant,
            )
        elif self._config.dynamics_model == "driverl_nuplan_bicycle_model":
            self._dynamics_model = BaseDynamicsModel.dynamics_model_factory(
                "nuplan_bicycle_model",
                config=self._config,
            )
        else:
            self._dynamics_model = BaseDynamicsModel.dynamics_model_factory(
                self._config.dynamics_model,
                config=self._config,
            )
        self._current_state: EgoState | None = None
        self._cached_policy_acceleration_target: float | None = None

    def get_state(self) -> EgoState:
        """Return current ego state, lazily initialized from the scenario."""
        if self._current_state is None:
            self._current_state = self._scenario.initial_ego_state
        return self._current_state

    def reset(self) -> None:
        """Reset controller state to scenario initial state on next access."""
        self._current_state = None
        self._cached_policy_acceleration_target = None

    def update_state(
        self,
        current_iteration: SimulationIteration,
        next_iteration: SimulationIteration,
        ego_state: EgoState,
        trajectory: AbstractTrajectory,
    ) -> None:
        """Execute one DriveRL action and set the next nuPlan EgoState."""
        if not isinstance(trajectory, DriveRLActionTrajectory):
            raise TypeError(
                "DriveRLOneStageController expects DriveRLActionTrajectory, "
                f"got {type(trajectory).__name__}."
            )

        sampling_time = next_iteration.time_point - current_iteration.time_point
        self._config.frame_time_interval = _duration_to_seconds(sampling_time)
        if self._config.dynamics_model == "nuplan_bicycle_model":
            self._current_state = self._propagate_with_nuplan_bicycle(
                ego_state=ego_state,
                trajectory=trajectory,
                sampling_time=sampling_time,
            )
            return
        if self._config.dynamics_model == "driverl_nuplan_bicycle_model":
            self._current_state = self._propagate_with_driverl_nuplan_bicycle(
                ego_state=ego_state,
                trajectory=trajectory,
                time_point=next_iteration.time_point,
            )
            return

        assert self._dynamics_model is not None
        self._dynamics_model.frame_time_interval = self._config.frame_time_interval

        (
            positions,
            velocities,
            accelerations,
            yaws,
            wheelbases,
            steerings,
            yaw_rates,
            accel_control_all,
            steering_control_all,
        ) = _ego_state_to_single_step_tensors(ego_state, device=self._device)
        actions = torch.tensor(
            [[[trajectory.jerk_long, trajectory.lat_command]]],
            dtype=torch.float32,
            device=self._device,
        )

        with torch.no_grad():
            common_kwargs = {
                "positions": positions,
                "velocities": velocities,
                "accelerations": accelerations,
                "yaws": yaws,
                "actions": actions,
                "wheelbases": wheelbases,
                "steerings": steerings,
                "yaw_rates": yaw_rates,
            }
            if self._config.dynamics_model == "jerk_pnc_model":
                common_kwargs.update(
                    {
                        "accel_control_all": accel_control_all,
                        "steering_control_all": steering_control_all,
                    }
                )

            (
                new_positions,
                new_velocities,
                new_yaws,
                new_accelerations,
                new_acceleration_control,
                new_steerings,
                new_steering_control,
                new_yaw_rates,
                _jerk_lat,
                _jerk_long,
            ) = self._dynamics_model._forward_internal(**common_kwargs)
        yaw_accel = (
            new_yaw_rates[0, 0, 0] - yaw_rates[0, 0, 0]
        ) / self._config.frame_time_interval

        next_state = driverl_state_to_ego_state(
            anchor_state=ego_state,
            position_xy=new_positions[0, 0, 0],
            velocity_xy=new_velocities[0, 0, 0],
            heading=new_yaws[0, 0, 0],
            acceleration=new_accelerations[0, 0, 0],
            steering_angle=new_steerings[0, 0, 0],
            yaw_rate=new_yaw_rates[0, 0, 0],
            time_point=next_iteration.time_point,
            yaw_accel=yaw_accel,
        )
        trajectory.acceleration_control = float(
            new_acceleration_control[0, 0, 0].item()
        )
        trajectory.steering_control = float(new_steering_control[0, 0, 0].item())
        self._assert_finite_ego_state(next_state)
        self._current_state = next_state

    def _propagate_with_nuplan_bicycle(
        self,
        *,
        ego_state: EgoState,
        trajectory: DriveRLActionTrajectory,
        sampling_time: TimeDuration,
    ) -> EgoState:
        """Propagate using nuPlan's official bicycle model with DriveRL direct action."""
        assert self._motion_model is not None
        dt = self._config.frame_time_interval
        debug_info = trajectory.debug_info or {}
        action_config = debug_info.get("engine_action_config") or {}
        desired_acceleration, steering_rate, acceleration_control, next_cached = (
            nuplan_bicycle_action_targets(
                ego_state=ego_state,
                jerk_long=trajectory.jerk_long,
                lat_command=trajectory.lat_command,
                frame_dt=dt,
                min_jerk_long=float(
                    action_config.get("min_jerk_long", self._config.min_jerk_long)
                ),
                max_jerk_long=float(
                    action_config.get("max_jerk_long", self._config.max_jerk_long)
                ),
                max_acceleration=float(
                    action_config.get(
                        "nuplan_bicycle_max_acceleration",
                        self._config.nuplan_bicycle_max_acceleration,
                    )
                ),
                max_steering_rate=float(
                    action_config.get(
                        "nuplan_bicycle_max_steering_rate",
                        self._config.nuplan_bicycle_max_steering_rate,
                    )
                ),
                positive_jerk_limit_when_nonnegative_acc=(
                    float(
                        action_config.get(
                            "positive_jerk_limit_when_nonnegative_acc",
                            self._config.positive_jerk_limit_when_nonnegative_acc,
                        )
                    )
                ),
                accel_time_constant=float(
                    action_config.get(
                        "nuplan_bicycle_accel_time_constant",
                        self._config.nuplan_bicycle_accel_time_constant,
                    )
                ),
                policy_interval_s=float(debug_info.get("policy_interval_s") or 0.0),
                policy_forward=bool(debug_info.get("policy_forward", True)),
                cached_acceleration_target=self._cached_policy_acceleration_target,
            )
        )
        self._cached_policy_acceleration_target = next_cached
        ideal_dynamic_state = DynamicCarState.build_from_rear_axle(
            rear_axle_to_center_dist=ego_state.car_footprint.rear_axle_to_center_dist,
            rear_axle_velocity_2d=ego_state.dynamic_car_state.rear_axle_velocity_2d,
            rear_axle_acceleration_2d=StateVector2D(desired_acceleration, 0.0),
            tire_steering_rate=steering_rate,
        )
        next_state = self._motion_model.propagate_state(
            state=ego_state,
            ideal_dynamic_state=ideal_dynamic_state,
            sampling_time=sampling_time,
        )
        trajectory.acceleration_control = acceleration_control
        trajectory.steering_control = next_state.dynamic_car_state.tire_steering_rate
        self._assert_finite_ego_state(next_state)
        return next_state

    def _propagate_with_driverl_nuplan_bicycle(
        self,
        *,
        ego_state: EgoState,
        trajectory: DriveRLActionTrajectory,
        time_point: Any,
    ) -> EgoState:
        """Propagate with DriveRL's tensor bicycle model."""
        assert self._dynamics_model is not None
        self._dynamics_model.frame_time_interval = self._config.frame_time_interval
        (
            positions,
            velocities,
            accelerations,
            yaws,
            wheelbases,
            steerings,
            yaw_rates,
            _accel_control_all,
            _steering_control_all,
        ) = _ego_state_to_single_step_tensors(ego_state, device=self._device)
        actions = torch.tensor(
            [[[trajectory.jerk_long, trajectory.lat_command]]],
            dtype=torch.float32,
            device=self._device,
        )
        with torch.no_grad():
            (
                new_positions,
                new_velocities,
                new_yaws,
                new_accelerations,
                new_acceleration_control,
                new_steerings,
                new_steering_control,
                new_yaw_rates,
                _jerk_lat,
                _jerk_long,
            ) = self._dynamics_model._forward_internal(
                positions=positions,
                velocities=velocities,
                accelerations=accelerations,
                yaws=yaws,
                actions=actions,
                wheelbases=wheelbases,
                steerings=steerings,
                yaw_rates=yaw_rates,
            )
        yaw_accel = (
            new_yaw_rates[0, 0, 0] - yaw_rates[0, 0, 0]
        ) / self._config.frame_time_interval
        next_state = driverl_state_to_ego_state(
            anchor_state=ego_state,
            position_xy=new_positions[0, 0, 0],
            velocity_xy=new_velocities[0, 0, 0],
            heading=new_yaws[0, 0, 0],
            acceleration=new_accelerations[0, 0, 0],
            steering_angle=new_steerings[0, 0, 0],
            yaw_rate=new_yaw_rates[0, 0, 0],
            time_point=time_point,
            yaw_accel=yaw_accel,
        )
        trajectory.acceleration_control = float(
            new_acceleration_control[0, 0, 0].item()
        )
        trajectory.steering_control = float(new_steering_control[0, 0, 0].item())
        self._assert_finite_ego_state(next_state)
        return next_state

    @staticmethod
    def _assert_finite_ego_state(ego_state: EgoState) -> None:
        values = [
            ego_state.rear_axle.x,
            ego_state.rear_axle.y,
            ego_state.rear_axle.heading,
            ego_state.dynamic_car_state.rear_axle_velocity_2d.x,
            ego_state.dynamic_car_state.rear_axle_velocity_2d.y,
            ego_state.dynamic_car_state.rear_axle_acceleration_2d.x,
            ego_state.dynamic_car_state.rear_axle_acceleration_2d.y,
            ego_state.dynamic_car_state.angular_velocity,
            ego_state.tire_steering_angle,
        ]
        if not torch.tensor(values, dtype=torch.float64).isfinite().all():
            raise FloatingPointError(
                "DriveRLOneStageController produced a non-finite EgoState."
            )


def _duration_to_seconds(duration: TimeDuration) -> float:
    if hasattr(duration, "time_s"):
        return float(duration.time_s)
    if hasattr(duration, "time_us"):
        return float(duration.time_us) * 1e-6
    return float(duration)


def _ego_state_to_single_step_tensors(
    ego_state: EgoState,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    """Build single-agent, single-step tensors expected by ``JerkPncModel``."""
    vx = float(ego_state.dynamic_car_state.rear_axle_velocity_2d.x)
    vy = float(ego_state.dynamic_car_state.rear_axle_velocity_2d.y)
    ax = float(ego_state.dynamic_car_state.rear_axle_acceleration_2d.x)
    wheelbase = (
        ego_state.car_footprint.vehicle_parameters.wheel_base
        if hasattr(ego_state.car_footprint.vehicle_parameters, "wheel_base")
        else ego_state.car_footprint.vehicle_parameters.length
    )
    positions = torch.zeros((1, 1, 1, 2), dtype=torch.float32, device=device)
    velocities = torch.tensor(
        [[[[vx, vy]]]],
        dtype=torch.float32,
        device=device,
    )
    accelerations = torch.tensor(
        [[[ax]]],
        dtype=torch.float32,
        device=device,
    )
    yaws = torch.zeros((1, 1, 1), dtype=torch.float32, device=device)
    wheelbases = torch.tensor(
        [[[float(wheelbase)]]], dtype=torch.float32, device=device
    )
    steerings = torch.tensor(
        [[[float(ego_state.tire_steering_angle)]]],
        dtype=torch.float32,
        device=device,
    )
    yaw_rates = torch.tensor(
        [[[float(ego_state.dynamic_car_state.angular_velocity)]]],
        dtype=torch.float32,
        device=device,
    )
    accel_control_all = accelerations.clone()
    steering_control_all = steerings.clone()
    return (
        positions,
        velocities,
        accelerations,
        yaws,
        wheelbases,
        steerings,
        yaw_rates,
        accel_control_all,
        steering_control_all,
    )
