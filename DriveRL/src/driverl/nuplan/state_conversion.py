"""State conversion utilities between nuPlan objects and DriveRL tensors.

All DriveRL rows produced here are expressed in the anchor ego rear-axle frame:
positions, velocities, and accelerations are rotated by ``-anchor.heading`` and
headings are wrapped relative to the anchor heading.
"""

from __future__ import annotations

import math
from typing import Any

import torch

try:
    from nuplan.common.actor_state.ego_state import EgoState
    from nuplan.common.actor_state.state_representation import (
        StateSE2,
        StateVector2D,
        TimePoint,
    )
    from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
except ImportError as exc:  # pragma: no cover - import guard for non-nuPlan envs.
    raise ImportError(
        "driverl.nuplan.state_conversion requires nuPlan to be installed."
    ) from exc


DRIVERL_OBJECT_STATE_DIM = 13
# Match ``export_driverl_webdataset.py`` nuPlan object type ids used for the
# vanilla_nuplan checkpoint.
DRIVERL_EGO_TYPE = 6.0
DRIVERL_VEHICLE_TYPE = 6.0
DRIVERL_PEDESTRIAN_TYPE = 4.0
DRIVERL_BICYCLE_TYPE = 5.0
DRIVERL_GENERIC_TYPE = 1.0


def wrap_angle(angle: float) -> float:
    """Wrap an angle in radians to ``[-pi, pi]``."""
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def _rot_to_anchor(x: float, y: float, anchor_heading: float) -> tuple[float, float]:
    """Rotate a global vector into the anchor frame."""
    c = math.cos(anchor_heading)
    s = math.sin(anchor_heading)
    return c * x + s * y, -s * x + c * y


def global_to_anchor_xy(
    x: float,
    y: float,
    anchor_x: float,
    anchor_y: float,
    anchor_heading: float,
) -> tuple[float, float]:
    """Convert a global point to anchor-frame coordinates."""
    return _rot_to_anchor(x - anchor_x, y - anchor_y, anchor_heading)


def global_heading_to_anchor(heading: float, anchor_heading: float) -> float:
    """Convert a global heading to a heading relative to the anchor frame."""
    return wrap_angle(float(heading) - float(anchor_heading))


def vector_to_anchor_xy(
    x: float,
    y: float,
    anchor_heading: float,
) -> tuple[float, float]:
    """Rotate a global-frame vector into the anchor frame."""
    return _rot_to_anchor(float(x), float(y), anchor_heading)


def local_vector_to_anchor_xy(
    x: float,
    y: float,
    local_heading: float,
    anchor_heading: float,
) -> tuple[float, float]:
    """Rotate a vector from a pose-local frame into the anchor frame."""
    delta_heading = float(local_heading) - float(anchor_heading)
    c = math.cos(delta_heading)
    s = math.sin(delta_heading)
    return c * float(x) - s * float(y), s * float(x) + c * float(y)


def anchor_vector_to_local_xy(
    x: float,
    y: float,
    local_heading: float,
    anchor_heading: float,
) -> tuple[float, float]:
    """Rotate an anchor-frame vector into a pose-local frame."""
    delta_heading = float(local_heading) - float(anchor_heading)
    c = math.cos(delta_heading)
    s = math.sin(delta_heading)
    return c * float(x) + s * float(y), -s * float(x) + c * float(y)


def _relative_timestamp_s(time_us: int, anchor_time_us: int) -> float:
    return (int(time_us) - int(anchor_time_us)) * 1e-6


def _type_code(tracked_object_type: Any, *, is_ego: bool = False) -> float:
    if is_ego:
        return DRIVERL_EGO_TYPE
    if tracked_object_type == TrackedObjectType.VEHICLE:
        return DRIVERL_VEHICLE_TYPE
    if tracked_object_type == TrackedObjectType.PEDESTRIAN:
        return DRIVERL_PEDESTRIAN_TYPE
    if tracked_object_type == TrackedObjectType.BICYCLE:
        return DRIVERL_BICYCLE_TYPE
    if tracked_object_type == TrackedObjectType.EGO:
        return DRIVERL_EGO_TYPE
    return DRIVERL_GENERIC_TYPE


def ego_state_to_driverl_row(
    ego_state: EgoState,
    anchor_state: EgoState,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Convert nuPlan ``EgoState`` to a DriveRL 13-dim object row."""
    anchor = anchor_state.rear_axle
    pose = ego_state.rear_axle
    x, y = global_to_anchor_xy(
        pose.x,
        pose.y,
        anchor.x,
        anchor.y,
        anchor.heading,
    )
    # nuPlan DB ego_pose velocity/acceleration fields are rear-axle local-frame
    # vectors. Rotate from the pose-local frame into the DriveRL anchor frame.
    vx, vy = local_vector_to_anchor_xy(
        ego_state.dynamic_car_state.rear_axle_velocity_2d.x,
        ego_state.dynamic_car_state.rear_axle_velocity_2d.y,
        pose.heading,
        anchor.heading,
    )
    ax, ay = local_vector_to_anchor_xy(
        ego_state.dynamic_car_state.rear_axle_acceleration_2d.x,
        ego_state.dynamic_car_state.rear_axle_acceleration_2d.y,
        pose.heading,
        anchor.heading,
    )
    vehicle = ego_state.car_footprint.vehicle_parameters
    row = [
        DRIVERL_EGO_TYPE,
        float(vehicle.length),
        float(vehicle.width),
        global_heading_to_anchor(pose.heading, anchor.heading),
        x,
        y,
        vx,
        vy,
        ax,
        ay,
        _relative_timestamp_s(ego_state.time_us, anchor_state.time_us),
        float(ego_state.dynamic_car_state.angular_velocity),
        float(ego_state.tire_steering_angle),
    ]
    return torch.tensor(row, dtype=dtype, device=device)


def tracked_object_to_driverl_row(
    tracked_object: Any,
    anchor_state: EgoState,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Convert a nuPlan tracked object to a DriveRL 13-dim object row.

    nuPlan tracked-object acceleration is often unavailable at runtime, so the
    acceleration columns are set to zero until a velocity-history estimator is
    added in ``DriveRLNuPlanFeatureBuilder``.
    """
    anchor = anchor_state.rear_axle
    center = tracked_object.center
    x, y = global_to_anchor_xy(
        center.x,
        center.y,
        anchor.x,
        anchor.y,
        anchor.heading,
    )
    velocity = getattr(tracked_object, "velocity", None)
    if velocity is None:
        vx = vy = 0.0
    else:
        vx, vy = vector_to_anchor_xy(velocity.x, velocity.y, anchor.heading)
    angular_velocity = getattr(tracked_object, "angular_velocity", None)
    row = [
        _type_code(tracked_object.tracked_object_type),
        float(tracked_object.box.length),
        float(tracked_object.box.width),
        global_heading_to_anchor(center.heading, anchor.heading),
        x,
        y,
        vx,
        vy,
        0.0,
        0.0,
        _relative_timestamp_s(
            tracked_object.metadata.timestamp_us,
            anchor_state.time_us,
        ),
        0.0 if angular_velocity is None else float(angular_velocity),
        0.0,
    ]
    return torch.tensor(row, dtype=dtype, device=device)


def driverl_state_to_ego_state(
    *,
    anchor_state: EgoState,
    position_xy: torch.Tensor,
    velocity_xy: torch.Tensor,
    heading: torch.Tensor,
    acceleration: torch.Tensor,
    steering_angle: torch.Tensor,
    yaw_rate: torch.Tensor,
    time_point: TimePoint,
    yaw_accel: torch.Tensor | None = None,
) -> EgoState:
    """Convert one DriveRL ego state in the anchor frame back to nuPlan EgoState."""
    anchor = anchor_state.rear_axle
    x_anchor = float(position_xy.reshape(-1)[0].item())
    y_anchor = float(position_xy.reshape(-1)[1].item())
    c = math.cos(anchor.heading)
    s = math.sin(anchor.heading)
    global_x = anchor.x + c * x_anchor - s * y_anchor
    global_y = anchor.y + s * x_anchor + c * y_anchor
    global_heading = wrap_angle(anchor.heading + float(heading.reshape(-1)[0].item()))

    vx_anchor = float(velocity_xy.reshape(-1)[0].item())
    vy_anchor = float(velocity_xy.reshape(-1)[1].item())
    vx_local, vy_local = anchor_vector_to_local_xy(
        vx_anchor,
        vy_anchor,
        global_heading,
        anchor.heading,
    )

    a_long = float(acceleration.reshape(-1)[0].item())
    angular_accel = 0.0 if yaw_accel is None else float(yaw_accel.reshape(-1)[0].item())

    return EgoState.build_from_rear_axle(
        rear_axle_pose=StateSE2(global_x, global_y, global_heading),
        rear_axle_velocity_2d=StateVector2D(vx_local, vy_local),
        rear_axle_acceleration_2d=StateVector2D(a_long, 0.0),
        tire_steering_angle=float(steering_angle.reshape(-1)[0].item()),
        time_point=time_point,
        vehicle_parameters=anchor_state.car_footprint.vehicle_parameters,
        is_in_auto_mode=True,
        angular_vel=float(yaw_rate.reshape(-1)[0].item()),
        angular_accel=angular_accel,
    )
