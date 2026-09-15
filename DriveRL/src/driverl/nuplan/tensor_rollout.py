"""Shared tensor-state rollout helpers for deploy paths."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from driverl.utils.geometry import get_wheelbase_from_length


@dataclass
class TensorRolloutState:
    positions: torch.Tensor
    velocities: torch.Tensor
    sizes: torch.Tensor
    orientations: torch.Tensor
    agent_types: torch.Tensor
    accelerations: torch.Tensor
    acceleration_controls: torch.Tensor
    yaw_rates: torch.Tensor
    steering_angles: torch.Tensor
    steering_controls: torch.Tensor
    jerk_lat: torch.Tensor
    jerk_long: torch.Tensor
    agent_masks: torch.Tensor
    frame_drop_mask: torch.Tensor

def advance_tensor_rollout_state(
    state: TensorRolloutState,
    actions: torch.Tensor,
    dynamics_model,
    *,
    agent_mask_next: torch.Tensor | None = None,
    frame_drop_next: torch.Tensor | None = None,
) -> TensorRolloutState:
    """Advance one step through the deploy dynamics model and append histories."""
    if getattr(dynamics_model, "action_keys_tensor", None) is not None:
        dynamics_model.action_keys_tensor = dynamics_model.action_keys_tensor.to(
            state.positions.device
        )

    dynamics_actions = actions
    if not (
        dynamics_actions.is_floating_point()
        and dynamics_actions.ndim >= 3
        and dynamics_actions.shape[-1] == 2
    ):
        dynamics_actions = dynamics_actions.unsqueeze(2)

    (
        new_positions,
        new_velocities,
        new_orientations,
        new_accelerations,
        new_acceleration_controls,
        new_steering_angles,
        new_steering_controls,
        new_yaw_rates,
        new_jerk_lat,
        new_jerk_long,
    ) = dynamics_model._forward_internal(
        state.positions[:, :, -1:, :],
        state.velocities[:, :, -1:, :],
        state.accelerations[:, :, -1:],
        state.orientations[:, :, -1:],
        dynamics_actions,
        get_wheelbase_from_length(state.sizes[:, :, -1:, 0]),
        state.steering_angles[:, :, -1:],
        state.yaw_rates[:, :, -1:],
        None,
    )

    if agent_mask_next is None:
        agent_mask_next = torch.ones_like(state.agent_masks[:, :, -1:])
    if frame_drop_next is None:
        frame_drop_next = torch.zeros_like(
            state.frame_drop_mask[:, :, -1:], dtype=torch.bool
        )

    return TensorRolloutState(
        positions=torch.cat([state.positions, new_positions], dim=2),
        velocities=torch.cat([state.velocities, new_velocities], dim=2),
        sizes=torch.cat([state.sizes, state.sizes[:, :, -1:, :]], dim=2),
        orientations=torch.cat([state.orientations, new_orientations], dim=2),
        agent_types=torch.cat([state.agent_types, state.agent_types[:, :, -1:]], dim=2),
        accelerations=torch.cat([state.accelerations, new_accelerations], dim=2),
        acceleration_controls=torch.cat(
            [state.acceleration_controls, new_acceleration_controls], dim=2
        ),
        yaw_rates=torch.cat([state.yaw_rates, new_yaw_rates], dim=2),
        steering_angles=torch.cat([state.steering_angles, new_steering_angles], dim=2),
        steering_controls=torch.cat(
            [state.steering_controls, new_steering_controls], dim=2
        ),
        jerk_lat=torch.cat([state.jerk_lat, new_jerk_lat], dim=2),
        jerk_long=torch.cat([state.jerk_long, new_jerk_long], dim=2),
        agent_masks=torch.cat([state.agent_masks, agent_mask_next], dim=2),
        frame_drop_mask=torch.cat([state.frame_drop_mask, frame_drop_next], dim=2),
    )
