"""DriveRL action-carrying trajectory for nuPlan closed-loop simulation."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch

from driverl.env.engine.dynamics_model.jerk_bicycle_model import (
    apply_jerk_and_clamp_acceleration,
)

try:
    from nuplan.common.actor_state.ego_state import EgoState
    from nuplan.common.actor_state.state_representation import (
        StateSE2,
        StateVector2D,
        TimeDuration,
    )
    from nuplan.planning.simulation.trajectory.interpolated_trajectory import (
        InterpolatedTrajectory,
    )
    from nuplan.planning.simulation.trajectory.trajectory_sampling import (
        TrajectorySampling,
    )
except ImportError as exc:  # pragma: no cover - import guard for non-nuPlan envs.
    raise ImportError(
        "driverl.nuplan.trajectory requires nuPlan to be installed."
    ) from exc


class DriveRLActionTrajectory(InterpolatedTrajectory):
    """Dummy nuPlan trajectory that carries a single DriveRL control action.

    The returned path is an interface placeholder for nuPlan. Execution is done
    by ``DriveRLOneStageController`` reading ``jerk_long`` and ``lat_command``.
    """

    def __init__(
        self,
        *,
        ego_state: EgoState,
        jerk_long: float,
        lat_command: float,
        raw_action: Any | None = None,
        acceleration_control: float | None = None,
        steering_control: float | None = None,
        goal_points: list[tuple[float, float]] | None = None,
        trajectory_sampling: TrajectorySampling = TrajectorySampling(
            num_poses=80,
            interval_length=0.1,
        ),
        reference_trajectory: list[EgoState] | None = None,
        debug_info: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            trajectory=(
                reference_trajectory
                if reference_trajectory is not None
                else _get_dummy_trajectory(
                    ego_state=ego_state,
                    trajectory_sampling=trajectory_sampling,
                    goal_points=goal_points,
                )
            )
        )
        self.ego_state = ego_state
        self.jerk_long = float(jerk_long)
        self.lat_command = float(lat_command)
        self.raw_action = raw_action
        self.acceleration_control = acceleration_control
        self.steering_control = steering_control
        self.goal_points = goal_points
        self.trajectory_sampling = trajectory_sampling
        self.reference_trajectory = reference_trajectory
        self.debug_info = debug_info or {}

    def __reduce__(self) -> tuple[Any, tuple[Any, ...]]:
        return (
            _rebuild_driverl_action_trajectory,
            (
                self.ego_state,
                self.jerk_long,
                self.lat_command,
                self.raw_action,
                self.acceleration_control,
                self.steering_control,
                self.goal_points,
                self.trajectory_sampling,
                self.reference_trajectory,
                self.debug_info,
            ),
        )


def nuplan_bicycle_action_targets(
    *,
    ego_state: EgoState,
    jerk_long: float,
    lat_command: float,
    frame_dt: float,
    min_jerk_long: float,
    max_jerk_long: float,
    max_acceleration: float,
    max_steering_rate: float,
    positive_jerk_limit_when_nonnegative_acc: float = 1.0,
    accel_time_constant: float = 0.2,
    policy_interval_s: float = 0.0,
    policy_forward: bool = True,
    cached_acceleration_target: float | None = None,
) -> tuple[float, float, float, float | None]:
    """Return ideal acceleration/steering-rate controls for nuPlan bicycle."""
    jerk_long = max(min_jerk_long, min(max_jerk_long, jerk_long))
    if (
        policy_interval_s > frame_dt
        and not policy_forward
        and cached_acceleration_target is not None
    ):
        desired_acceleration = cached_acceleration_target
    else:
        action_dt = policy_interval_s if policy_interval_s > frame_dt else frame_dt
        desired_acceleration = float(
            apply_jerk_and_clamp_acceleration(
                current_acceleration=torch.tensor(
                    ego_state.dynamic_car_state.rear_axle_acceleration_2d.x,
                    dtype=torch.float32,
                ),
                jerk=torch.tensor(jerk_long, dtype=torch.float32),
                time_interval=action_dt,
                min_acceleration=-max_acceleration,
                max_acceleration=max_acceleration,
                positive_jerk_limit_when_negative_acc=max_jerk_long,
                positive_jerk_limit_when_nonnegative_acc=(
                    positive_jerk_limit_when_nonnegative_acc
                ),
            ).item()
        )

    acceleration_control = desired_acceleration
    speed = float(ego_state.dynamic_car_state.rear_axle_velocity_2d.x)
    current_acceleration = float(
        ego_state.dynamic_car_state.rear_axle_acceleration_2d.x
    )
    accel_alpha = frame_dt / (frame_dt + accel_time_constant)
    updated_acceleration = current_acceleration + accel_alpha * (
        desired_acceleration - current_acceleration
    )
    if speed + updated_acceleration * frame_dt < 0.0:
        stop_acceleration = -speed - 0.75 * current_acceleration
        acceleration_control = max(
            -max_acceleration,
            min(max_acceleration, stop_acceleration),
        )
        desired_acceleration = (
            acceleration_control - current_acceleration
        ) / accel_alpha + current_acceleration

    steering_rate = max(-max_steering_rate, min(max_steering_rate, lat_command))
    next_cached = desired_acceleration if policy_interval_s > frame_dt else None
    return desired_acceleration, steering_rate, acceleration_control, next_cached


def build_lqr_action_matching_trajectory(
    *,
    ego_state: EgoState,
    target_acceleration: float,
    target_steering_rate: float,
    trajectory_sampling: TrajectorySampling,
    heading_span: float = 0.8,
    lateral_span: float = 3.0,
    heading_samples: int = 9,
    lateral_samples: int = 5,
    discretization_time: float = 0.1,
    tracking_horizon: int = 10,
) -> tuple[list[EgoState], dict[str, float]]:
    """Build a reference trajectory whose default nuPlan LQR command matches action."""
    target_acceleration = float(target_acceleration)
    target_steering_rate = float(np.clip(target_steering_rate, -0.5, 0.5))
    reference_acceleration = target_acceleration / 1.05
    reference_velocity_floor = None
    stopping_guard_applied = False
    initial_speed = float(ego_state.dynamic_car_state.rear_axle_velocity_2d.x)
    reference_velocity = _longitudinal_reference_velocity(
        initial_speed=initial_speed,
        target_acceleration=reference_acceleration,
        discretization_time=discretization_time,
        tracking_horizon=tracking_horizon,
    )
    stopping_velocity = 0.2
    stopping_acceleration = -0.5 * (initial_speed - reference_velocity)
    if (
        initial_speed <= stopping_velocity
        and reference_velocity <= stopping_velocity
        and (
            abs(target_steering_rate) > 1e-3
            or abs(target_acceleration - stopping_acceleration) > 0.05
        )
    ):
        recovery_acceleration = max(
            target_acceleration,
            _low_speed_recovery_acceleration(
                ego_state=ego_state,
                frame_dt=trajectory_sampling.interval_length,
                accel_time_constant=0.2,
            ),
        )
        reference_velocity_floor = _longitudinal_reference_velocity(
            initial_speed=initial_speed,
            target_acceleration=recovery_acceleration,
            discretization_time=discretization_time,
            tracking_horizon=tracking_horizon,
        )
        reference_velocity = max(reference_velocity, reference_velocity_floor)
        stopping_guard_applied = True
    matched_acceleration = _longitudinal_lqr_acceleration(
        initial_speed=initial_speed,
        reference_velocity=reference_velocity,
        discretization_time=discretization_time,
        tracking_horizon=tracking_horizon,
    )
    heading_offset, lateral_offset, matched_steering_rate = (
        _solve_lateral_offsets_for_steering_rate(
            ego_state=ego_state,
            target_steering_rate=target_steering_rate,
            target_acceleration=target_acceleration,
            heading_span=heading_span,
            lateral_span=lateral_span,
            discretization_time=discretization_time,
            tracking_horizon=tracking_horizon,
        )
    )
    reference = _get_lqr_reference_trajectory(
        ego_state=ego_state,
        trajectory_sampling=trajectory_sampling,
        target_acceleration=reference_acceleration,
        heading_offset=heading_offset,
        lateral_offset=lateral_offset,
        reference_velocity_floor=reference_velocity_floor,
        discretization_time=discretization_time,
        tracking_horizon=tracking_horizon,
    )
    return reference, _action_match_debug(
        target_acceleration,
        target_steering_rate,
        matched_acceleration,
        matched_steering_rate,
        heading_offset,
        lateral_offset,
        reference_velocity,
        stopping_guard_applied,
    )


def _get_lqr_reference_trajectory(
    *,
    ego_state: EgoState,
    trajectory_sampling: TrajectorySampling,
    target_acceleration: float,
    heading_offset: float,
    lateral_offset: float,
    reference_velocity_floor: float | None,
    discretization_time: float,
    tracking_horizon: int,
) -> list[EgoState]:
    dt = float(trajectory_sampling.interval_length)
    horizon_time = max(float(discretization_time) * int(tracking_horizon), dt)
    initial_speed = float(ego_state.dynamic_car_state.rear_axle_velocity_2d.x)
    reference_velocity = _longitudinal_reference_velocity(
        initial_speed=initial_speed,
        target_acceleration=target_acceleration,
        discretization_time=discretization_time,
        tracking_horizon=tracking_horizon,
    )
    if reference_velocity_floor is not None:
        reference_velocity = max(reference_velocity, reference_velocity_floor)
    path_acceleration = (reference_velocity - initial_speed) / horizon_time

    states = []
    for idx in range(trajectory_sampling.num_poses + 1):
        t = idx * dt
        distance = initial_speed * t + 0.5 * path_acceleration * t * t
        speed = max(initial_speed + path_acceleration * t, 0.0)
        states.append(
            _reference_state(
                ego_state=ego_state,
                t=t,
                distance=distance,
                speed=speed,
                acceleration=path_acceleration,
                heading_offset=heading_offset,
                lateral_offset=lateral_offset,
            )
        )
    return states


def _longitudinal_reference_velocity(
    *,
    initial_speed: float,
    target_acceleration: float,
    discretization_time: float,
    tracking_horizon: int,
) -> float:
    q_longitudinal = 10.0
    r_longitudinal = 1.0
    b = float(discretization_time) * int(tracking_horizon)
    return float(
        initial_speed
        + target_acceleration
        * (b * b * q_longitudinal + r_longitudinal)
        / max(b * q_longitudinal, 1e-6)
    )


def _longitudinal_lqr_acceleration(
    *,
    initial_speed: float,
    reference_velocity: float,
    discretization_time: float,
    tracking_horizon: int,
) -> float:
    q_longitudinal = 10.0
    r_longitudinal = 1.0
    b = float(discretization_time) * int(tracking_horizon)
    return float(
        b
        * q_longitudinal
        * (reference_velocity - initial_speed)
        / (b * b * q_longitudinal + r_longitudinal)
    )


def _low_speed_recovery_acceleration(
    *,
    ego_state: EgoState,
    frame_dt: float,
    accel_time_constant: float,
) -> float:
    speed = float(ego_state.dynamic_car_state.rear_axle_velocity_2d.x)
    current_acceleration = float(
        ego_state.dynamic_car_state.rear_axle_acceleration_2d.x
    )
    accel_alpha = frame_dt / (frame_dt + accel_time_constant)
    stop_acceleration = -speed - 0.75 * current_acceleration
    return float(
        (stop_acceleration - current_acceleration) / accel_alpha + current_acceleration
    )


def _reference_state(
    *,
    ego_state: EgoState,
    t: float,
    distance: float,
    speed: float,
    acceleration: float,
    heading_offset: float,
    lateral_offset: float,
) -> EgoState:
    anchor = ego_state.rear_axle
    heading = anchor.heading + heading_offset
    cos_anchor = math.cos(anchor.heading)
    sin_anchor = math.sin(anchor.heading)
    x_local = distance * math.cos(heading_offset)
    y_local = lateral_offset + distance * math.sin(heading_offset)
    x = anchor.x + cos_anchor * x_local - sin_anchor * y_local
    y = anchor.y + sin_anchor * x_local + cos_anchor * y_local
    return EgoState.build_from_rear_axle(
        rear_axle_pose=StateSE2(x, y, heading),
        rear_axle_velocity_2d=StateVector2D(speed, 0.0),
        rear_axle_acceleration_2d=StateVector2D(acceleration, 0.0),
        tire_steering_angle=ego_state.tire_steering_angle,
        time_point=ego_state.time_point + TimeDuration.from_s(t),
        vehicle_parameters=ego_state.car_footprint.vehicle_parameters,
        is_in_auto_mode=True,
        angular_vel=0.0,
        angular_accel=0.0,
        tire_steering_rate=0.0,
    )


def _solve_lateral_offsets_for_steering_rate(
    *,
    ego_state: EgoState,
    target_steering_rate: float,
    target_acceleration: float,
    heading_span: float,
    lateral_span: float,
    discretization_time: float,
    tracking_horizon: int,
) -> tuple[float, float, float]:
    base = _fast_lqr_steering_rate(
        ego_state=ego_state,
        target_acceleration=target_acceleration,
        heading_offset=0.0,
        lateral_offset=0.0,
        discretization_time=discretization_time,
        tracking_horizon=tracking_horizon,
    )
    eps = 1e-3
    heading_gain = (
        _fast_lqr_steering_rate(
            ego_state=ego_state,
            target_acceleration=target_acceleration,
            heading_offset=eps,
            lateral_offset=0.0,
            discretization_time=discretization_time,
            tracking_horizon=tracking_horizon,
        )
        - base
    ) / eps
    lateral_gain = (
        _fast_lqr_steering_rate(
            ego_state=ego_state,
            target_acceleration=target_acceleration,
            heading_offset=0.0,
            lateral_offset=eps,
            discretization_time=discretization_time,
            tracking_horizon=tracking_horizon,
        )
        - base
    ) / eps

    candidates = [(0.0, 0.0)]
    delta = target_steering_rate - base
    denom = heading_gain * heading_gain + lateral_gain * lateral_gain
    heading_limit = abs(float(heading_span))
    lateral_limit = abs(float(lateral_span))
    if denom > 1e-12:
        candidates.append(
            (
                float(
                    np.clip(delta * heading_gain / denom, -heading_limit, heading_limit)
                ),
                float(
                    np.clip(delta * lateral_gain / denom, -lateral_limit, lateral_limit)
                ),
            )
        )
    for heading in (-heading_limit, heading_limit):
        if abs(lateral_gain) > 1e-12:
            lateral = (delta - heading_gain * heading) / lateral_gain
            candidates.append(
                (heading, float(np.clip(lateral, -lateral_limit, lateral_limit)))
            )
    for lateral in (-lateral_limit, lateral_limit):
        if abs(heading_gain) > 1e-12:
            heading = (delta - lateral_gain * lateral) / heading_gain
            candidates.append(
                (float(np.clip(heading, -heading_limit, heading_limit)), lateral)
            )

    best_heading, best_lateral = min(
        candidates,
        key=lambda item: abs(
            base
            + heading_gain * item[0]
            + lateral_gain * item[1]
            - target_steering_rate
        ),
    )
    matched = base + heading_gain * best_heading + lateral_gain * best_lateral
    return best_heading, best_lateral, matched


def _fast_lqr_steering_rate(
    *,
    ego_state: EgoState,
    target_acceleration: float,
    heading_offset: float,
    lateral_offset: float,
    discretization_time: float,
    tracking_horizon: int,
) -> float:
    """Match nuPlan LQR lateral math for the straight reference used here."""
    dt = float(discretization_time)
    horizon = int(tracking_horizon)
    velocity = float(ego_state.dynamic_car_state.rear_axle_velocity_2d.x)
    velocity_profile = velocity + target_acceleration * dt * np.arange(
        horizon, dtype=np.float64
    )

    A = np.eye(3, dtype=np.float64)
    B = np.zeros((3, 1), dtype=np.float64)
    input_matrix = np.array([[0.0], [0.0], [dt]], dtype=np.float64)
    wheel_base = float(ego_state.car_footprint.vehicle_parameters.wheel_base)
    for speed in velocity_profile:
        state_matrix = np.eye(3, dtype=np.float64)
        state_matrix[0, 1] = speed * dt
        state_matrix[1, 2] = speed * dt / wheel_base
        A = state_matrix @ A
        B = state_matrix @ B + input_matrix

    lateral_error = -lateral_offset * math.cos(heading_offset)
    initial_state = np.array(
        [lateral_error, -heading_offset, float(ego_state.tire_steering_angle)],
        dtype=np.float64,
    )
    q_lateral = np.diag([1.0, 10.0, 0.0])
    r_lateral = np.asarray([[1.0]], dtype=np.float64)
    error = A @ initial_state
    for idx in (1, 2):
        error[idx] = (error[idx] + math.pi) % (2.0 * math.pi) - math.pi
    command = -np.linalg.inv(B.T @ q_lateral @ B + r_lateral) @ B.T @ q_lateral @ error
    return float(command.item())


def _action_match_debug(
    target_acceleration: float,
    target_steering_rate: float,
    matched_acceleration: float,
    matched_steering_rate: float,
    heading_offset: float,
    lateral_offset: float,
    reference_velocity: float,
    stopping_guard_applied: bool,
) -> dict[str, float]:
    return {
        "target_acceleration": float(target_acceleration),
        "target_steering_rate": float(target_steering_rate),
        "matched_acceleration": float(matched_acceleration),
        "matched_steering_rate": float(matched_steering_rate),
        "acceleration_error": float(matched_acceleration - target_acceleration),
        "steering_rate_error": float(matched_steering_rate - target_steering_rate),
        "heading_offset": float(heading_offset),
        "lateral_offset": float(lateral_offset),
        "reference_velocity": float(reference_velocity),
        "stopping_guard_applied": float(stopping_guard_applied),
    }


def _action_match_score(debug: dict[str, float]) -> float:
    return (
        abs(debug["acceleration_error"]) / 3.0 + abs(debug["steering_rate_error"]) / 0.5
    )


def _get_dummy_trajectory(
    *,
    ego_state: EgoState,
    trajectory_sampling: TrajectorySampling,
    goal_points: list[tuple[float, float]] | None = None,
) -> list[EgoState]:
    """Return a NuBoard-visible debug trajectory carrying DriveRL goal points."""
    time_delta = TimeDuration.from_s(trajectory_sampling.interval_length)
    trajectory = [ego_state]
    if goal_points:
        return _get_goal_debug_trajectory(
            ego_state=ego_state,
            goal_points=goal_points,
            time_delta=time_delta,
        )
    for time_idx in range(1, trajectory_sampling.num_poses + 1):
        time_point = ego_state.time_point + time_idx * time_delta
        state = EgoState.build_from_rear_axle(
            rear_axle_pose=ego_state.rear_axle,
            rear_axle_velocity_2d=ego_state.dynamic_car_state.rear_axle_velocity_2d,
            rear_axle_acceleration_2d=ego_state.dynamic_car_state.rear_axle_acceleration_2d,
            tire_steering_angle=ego_state.tire_steering_angle,
            time_point=time_point,
            vehicle_parameters=ego_state.car_footprint.vehicle_parameters,
            is_in_auto_mode=True,
            angular_vel=ego_state.dynamic_car_state.angular_velocity,
            angular_accel=ego_state.dynamic_car_state.angular_acceleration,
            tire_steering_rate=ego_state.dynamic_car_state.tire_steering_rate,
        )
        trajectory.append(state)
    return trajectory


def _get_goal_debug_trajectory(
    *,
    ego_state: EgoState,
    goal_points: list[tuple[float, float]],
    time_delta: TimeDuration,
) -> list[EgoState]:
    trajectory = [ego_state]
    anchor = ego_state.rear_axle
    previous_x = anchor.x
    previous_y = anchor.y
    cos_h = math.cos(anchor.heading)
    sin_h = math.sin(anchor.heading)
    for idx, (x_local, y_local) in enumerate(goal_points, start=1):
        global_x = anchor.x + cos_h * float(x_local) - sin_h * float(y_local)
        global_y = anchor.y + sin_h * float(x_local) + cos_h * float(y_local)
        heading = (
            math.atan2(global_y - previous_y, global_x - previous_x)
            if math.hypot(global_x - previous_x, global_y - previous_y) > 1e-3
            else anchor.heading
        )
        trajectory.append(
            EgoState.build_from_rear_axle(
                rear_axle_pose=StateSE2(global_x, global_y, heading),
                rear_axle_velocity_2d=StateVector2D(0.0, 0.0),
                rear_axle_acceleration_2d=StateVector2D(0.0, 0.0),
                tire_steering_angle=ego_state.tire_steering_angle,
                time_point=ego_state.time_point + idx * time_delta,
                vehicle_parameters=ego_state.car_footprint.vehicle_parameters,
                is_in_auto_mode=True,
                angular_vel=0.0,
                angular_accel=0.0,
            )
        )
        previous_x = global_x
        previous_y = global_y
    return trajectory


def _rebuild_driverl_action_trajectory(
    ego_state: EgoState,
    jerk_long: float,
    lat_command: float,
    raw_action: Any | None,
    acceleration_control: float | None,
    steering_control: float | None,
    goal_points: list[tuple[float, float]] | None,
    trajectory_sampling: TrajectorySampling,
    reference_trajectory: list[EgoState] | dict[str, Any] | None,
    debug_info: dict[str, Any] | None = None,
) -> DriveRLActionTrajectory:
    if isinstance(reference_trajectory, dict) and debug_info is None:
        debug_info = reference_trajectory
        reference_trajectory = None
    return DriveRLActionTrajectory(
        ego_state=ego_state,
        jerk_long=jerk_long,
        lat_command=lat_command,
        raw_action=raw_action,
        acceleration_control=acceleration_control,
        steering_control=steering_control,
        goal_points=goal_points,
        trajectory_sampling=trajectory_sampling,
        reference_trajectory=reference_trajectory,
        debug_info=debug_info or {},
    )
