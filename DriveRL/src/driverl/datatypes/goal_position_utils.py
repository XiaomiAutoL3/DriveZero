from typing import Optional, Sequence

import torch

from driverl.env.constants import EPS
from driverl.utils.geometry import project_points_to_segments, rotate_2d_points


def normalize_goal_positions(
    goal_positions: torch.Tensor, num_goal_positions: Optional[int] = None
) -> torch.Tensor:
    """Normalize goal tensors to the ``[..., 2 * num_goal_positions]`` contract.

    The last dimension stores a flat sequence of ``(x, y)`` goal points. When a
    target count is provided, legacy single-goal tensors are repeated, longer
    tensors are truncated, and shorter multi-goal tensors repeat their last
    goal to fill the requested count.
    """
    goal_width = goal_positions.shape[-1]
    if goal_width % 2 != 0:
        raise ValueError(
            "goal_positions must have an even last dimension, "
            f"got shape {tuple(goal_positions.shape)}."
        )
    if goal_width < 2:
        raise ValueError(
            "goal_positions must contain at least one (x, y) goal, "
            f"got shape {tuple(goal_positions.shape)}."
        )
    if num_goal_positions is None:
        return goal_positions
    if num_goal_positions < 1:
        raise ValueError("num_goal_positions must be >= 1.")

    current_goal_count = goal_width // 2
    if current_goal_count == num_goal_positions:
        return goal_positions

    goals = goal_positions.reshape(*goal_positions.shape[:-1], current_goal_count, 2)
    if current_goal_count > num_goal_positions:
        goals = goals[..., :num_goal_positions, :]
    else:
        pad = goals[..., -1:, :].expand(
            *goals.shape[:-2], num_goal_positions - current_goal_count, 2
        )
        goals = torch.cat([goals, pad], dim=-2)
    return goals.reshape(*goal_positions.shape[:-1], num_goal_positions * 2)


def apply_goal_pair_mode(goal_positions: torch.Tensor, mode: str) -> torch.Tensor:
    """Apply a two-slot goal layout while preserving the flattened tensor contract."""
    if mode in {"legacy", "ordered_sequential"}:
        return goal_positions
    if mode not in {"duplicate_earlier", "duplicate_later"}:
        raise ValueError(f"Unsupported goal_pair_mode: {mode}.")
    if goal_positions.shape[-1] != 4:
        raise ValueError(
            f"goal_pair_mode={mode!r} requires exactly two goals, got shape "
            f"{tuple(goal_positions.shape)}."
        )
    goals = goal_positions.reshape(*goal_positions.shape[:-1], 2, 2)
    selected = goals[..., :1, :] if mode == "duplicate_earlier" else goals[..., -1:, :]
    return selected.expand_as(goals).reshape_as(goal_positions)


def require_single_goal_position(
    goal_positions: torch.Tensor, context: str = "This model"
) -> torch.Tensor:
    """Return single-goal positions or fail for multi-goal tensors."""
    goal_positions = normalize_goal_positions(goal_positions)
    if goal_positions.shape[-1] != 2:
        raise ValueError(
            f"{context} only supports a single goal position, got "
            f"{goal_positions.shape[-1] // 2}. Set env.goal_count_probs to "
            "[1.0] or use a multi-goal-capable agent."
        )
    return goal_positions


def _normalize_goal_count_probs(
    goal_count_probs: Optional[Sequence[float] | torch.Tensor],
    num_goal_positions: int,
    device: torch.device,
) -> torch.Tensor:
    """Return a probability vector over generated goal counts ``1..G``."""
    if goal_count_probs is None:
        values = [1.0] + [0.0] * (num_goal_positions - 1)
        probs = torch.tensor(values, dtype=torch.float32, device=device)
    elif isinstance(goal_count_probs, torch.Tensor):
        probs = goal_count_probs.to(dtype=torch.float32, device=device)
    else:
        probs = torch.tensor(list(goal_count_probs), dtype=torch.float32, device=device)

    if probs.ndim != 1 or probs.numel() != num_goal_positions:
        raise ValueError(
            "goal_count_probs must be a 1D probability vector with length "
            f"num_goal_positions ({num_goal_positions}), got shape "
            f"{tuple(probs.shape)}."
        )
    if (probs < 0).any():
        raise ValueError("goal_count_probs must be non-negative.")
    prob_sum = probs.sum()
    if not bool(torch.isfinite(prob_sum).item()) or float(prob_sum.item()) <= 0:
        raise ValueError("goal_count_probs must sum to a positive finite value.")
    return probs / prob_sum


def _project_goals_to_nearest_lanes(
    goals: torch.Tensor,  # [B, A, 2]
    lanes_points: torch.Tensor,  # [B, L, P, D]  D >= 2
    lanes_mask: torch.Tensor,  # [B, L]
    k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Find the nearest k lane polylines per goal and return their projections.

    Each polyline is treated as a chain of segments (P-1 segments). The
    polyline-to-goal distance is the minimum point-to-segment distance across
    its segments, and the returned projection is the foot of perpendicular on
    the closest segment of that polyline (clamped to the segment endpoints).

    Returns:
        ``(projections, distances)`` where ``projections`` has shape
        ``[B, A, k, 2]`` and ``distances`` has shape ``[B, A, k]``. Invalid or
        padded slots fall back to the goal itself and carry ``+inf`` distance
        so downstream samplers can zero them out.
    """
    B, A, _ = goals.shape
    _, L, P, _ = lanes_points.shape
    dtype = goals.dtype

    if L == 0 or P < 2:
        projections = goals.unsqueeze(2).expand(B, A, k, 2).contiguous()
        distances = goals.new_full((B, A, k), float("inf"))
        return projections, distances

    # Segment endpoints per lane: [B, L, P-1, 2]
    seg_start = lanes_points[..., :-1, :2]
    seg_end = lanes_points[..., 1:, :2]

    # Per-segment mask: ``lanes_points`` pads shorter polylines with zeros, so
    # gate on "either endpoint non-zero" in addition to the polyline-level
    # mask.
    seg_valid_end = (seg_end.abs().sum(dim=-1) > 0) | (seg_start.abs().sum(dim=-1) > 0)
    seg_mask = lanes_mask.unsqueeze(-1) & seg_valid_end  # [B, L, P-1]

    # Broadcast goals against segments: goals -> [B, A, 1, 1, 2],
    # seg_start/seg_end -> [B, 1, L, P-1, 2].
    goal_exp = goals.unsqueeze(2).unsqueeze(3)
    seg_start_exp = seg_start.unsqueeze(1)
    seg_end_exp = seg_end.unsqueeze(1)
    proj, _, seg_dist = project_points_to_segments(
        goal_exp.expand(B, A, L, P - 1, 2),
        seg_start_exp.expand(B, A, L, P - 1, 2),
        seg_end_exp.expand(B, A, L, P - 1, 2),
        return_dist=True,
    )  # proj: [B, A, L, P-1, 2], seg_dist: [B, A, L, P-1]

    seg_mask_exp = seg_mask.unsqueeze(1).expand(B, A, L, P - 1)
    seg_dist = torch.where(
        seg_mask_exp, seg_dist, torch.full_like(seg_dist, float("inf"))
    )

    # Per-lane nearest-segment distance and the corresponding projection point.
    lane_dist, best_seg_idx = seg_dist.min(dim=-1)  # [B, A, L]
    best_seg_idx_exp = best_seg_idx.unsqueeze(-1).unsqueeze(-1).expand(B, A, L, 1, 2)
    lane_proj = proj.gather(3, best_seg_idx_exp).squeeze(3)  # [B, A, L, 2]

    # Pick top-k nearest lanes per goal.
    k_eff = min(k, L)
    topk_dist, topk_idx = lane_dist.topk(k_eff, dim=-1, largest=False)  # [B, A, k_eff]
    topk_idx_exp = topk_idx.unsqueeze(-1).expand(B, A, k_eff, 2)
    topk_proj = lane_proj.gather(2, topk_idx_exp)  # [B, A, k_eff, 2]

    # If a slot is invalid (distance == inf) fall back to the goal itself so
    # the downstream sampler never draws a garbage coordinate. Same for pad
    # slots when L < k.
    invalid = ~torch.isfinite(topk_dist)  # [B, A, k_eff]
    fallback = goals.unsqueeze(2).expand(B, A, k_eff, 2)
    topk_proj = torch.where(invalid.unsqueeze(-1), fallback, topk_proj)

    if k_eff < k:
        pad_proj = goals.unsqueeze(2).expand(B, A, k - k_eff, 2)
        pad_dist = topk_dist.new_full((B, A, k - k_eff), float("inf"))
        topk_proj = torch.cat([topk_proj, pad_proj], dim=2)
        topk_dist = torch.cat([topk_dist, pad_dist], dim=-1)

    return topk_proj.to(dtype), topk_dist


def sample_valid_positions(
    positions: torch.Tensor,
    masks: torch.Tensor,
    mode: str,
    orientations: torch.Tensor,
    gamma: float = 0.0,
    offset_sigma_major: float = 0.0,
    offset_sigma_minor: float = 0.0,
    lanes_centers_points: Optional[torch.Tensor] = None,
    lanes_centers_mask: Optional[torch.Tensor] = None,
    goal_snap_keep_prob: Optional[float] = None,
    goal_snap_k: int = 4,
    goal_snap_max_distance: float = float("inf"),
    goal_count_probs: Optional[Sequence[float] | torch.Tensor] = None,
    num_goal_positions: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Get a valid position for each agent, either the last valid one
    or a randomly sampled valid one with bias toward later timesteps.

    Args:
        positions: Tensor of shape [batch_size, max_agents, time_steps, 2]
        masks: Tensor of shape [batch_size, max_agents, time_steps] indicating valid positions
        mode: str
            "last"   -> take the last valid position
            "linear" -> sample randomly, probability ~ (t+1)^gamma
            "exp"    -> sample randomly, probability ~ exp(gamma * t_norm)
        gamma: Optional[float]
            Bias strength. Larger = stronger preference for later timesteps.
            - For "linear": exponent power.
            - For "exp": exponential scaling factor.
        lanes_centers_points: Optional tensor ``[B, L, P, D]`` of lane-center
            polylines (first two feature dims are x, y). When provided together
            with ``goal_snap_keep_prob`` the sampled goals can be snapped to
            the projection on one of the K nearest lane polylines.
        lanes_centers_mask: Optional bool tensor ``[B, L]`` marking valid
            polylines. Required whenever ``lanes_centers_points`` is passed.
        goal_snap_keep_prob: Probability (``[0, 1]``) of keeping the raw
            sampled goal. With probability ``1 - keep_prob`` the goal is
            snapped to one of the K nearest lane-center projections, drawn
            with weight proportional to ``1 / distance`` (closer lanes win).
            ``None`` disables snapping.
        goal_snap_k: Number of nearest lane polylines considered for snapping.
            Defaults to 4.
        goal_snap_max_distance: Only lane polylines within this distance of
            the sampled goal are considered snap candidates; if none remain
            for an agent the raw goal is kept. Defaults to ``+inf``.
        goal_count_probs: For stochastic modes (``linear`` / ``exp``),
            probability vector over the number of distinct goals to sample.
            Entry ``i`` gives the probability of sampling ``i + 1`` ordered
            timesteps. Slots beyond the sampled count repeat the last sampled
            goal so the tensor width remains fixed.
        num_goal_positions: Number of goal points to output. ``1`` preserves
            legacy single-goal behavior.

    Returns:
        ``(last_valid_positions, goal_positions)`` where
        ``last_valid_positions`` is ``[B, A, 2]`` and ``goal_positions`` is
        ``[B, A, 2 * num_goal_positions]``.
    """
    batch_size, max_agents, time_steps, _ = positions.shape
    device = positions.device
    if num_goal_positions < 1:
        raise ValueError("num_goal_positions must be >= 1.")

    # Create a range of timesteps [0, 1, 2, ..., time_steps-1]
    timestep_range = torch.arange(time_steps, device=device).view(1, 1, -1)

    # For positions where mask is False, set timestep to -1 (invalid)
    masked_timesteps = torch.where(
        masks, timestep_range, torch.tensor(-1, device=device)
    )

    # Get the maximum valid timestep for each agent
    last_valid_timesteps = masked_timesteps.max(dim=-1).values

    # Create indices for gathering
    batch_indices = torch.arange(batch_size, device=device).view(-1, 1)
    agent_indices = torch.arange(max_agents, device=device).view(1, -1)

    # Gather the last valid positions
    last_valid_positions = positions[batch_indices, agent_indices, last_valid_timesteps]

    if mode == "last":
        sampled_t_all = (
            last_valid_timesteps.clamp(0)
            .unsqueeze(-1)
            .expand(-1, -1, num_goal_positions)
        )
        sampled_positions = last_valid_positions.unsqueeze(2).expand(
            -1, -1, num_goal_positions, -1
        )
        generated_goal_counts = torch.ones(
            (batch_size, max_agents), dtype=torch.long, device=device
        )
        goal_slot_is_generated = torch.arange(num_goal_positions, device=device).view(
            1, 1, -1
        ) < generated_goal_counts.unsqueeze(-1)
    elif mode in ["linear", "exp"]:
        if gamma < 0:
            raise ValueError(f"When mode is {mode}, gamma must have a value.")
        goal_count_probs_t = _normalize_goal_count_probs(
            goal_count_probs, num_goal_positions, device
        )

        t = torch.arange(time_steps, device=device, dtype=torch.float)

        if mode == "linear":
            base_w = (t + 1.0) ** gamma
        else:
            t_norm = t / max(time_steps - 1, 1)
            base_w = torch.exp(gamma * t_norm)

        # Apply mask -> [B, A, T]
        weights = base_w.view(1, 1, time_steps) * masks
        weights_2d = weights.reshape(batch_size * max_agents, time_steps)

        # Handle agents with no valid timesteps: fallback to last timestep
        row_sums = weights_2d.sum(dim=-1, keepdim=True)
        no_valid = row_sums == 0
        if no_valid.any():
            fallback = torch.zeros_like(weights_2d)
            fallback[:, -1] = 1.0
            weights_2d = torch.where(no_valid, fallback, weights_2d)
            row_sums = torch.where(no_valid, torch.ones_like(row_sums), row_sums)

        # Normalize to probabilities
        probs_2d = weights_2d / row_sums

        valid_counts = masks.sum(dim=-1)
        sampled_goal_counts = (
            torch.multinomial(
                goal_count_probs_t.unsqueeze(0).expand(batch_size * max_agents, -1),
                num_samples=1,
                replacement=True,
            ).view(batch_size, max_agents)
            + 1
        )
        generated_goal_counts = torch.minimum(
            sampled_goal_counts, valid_counts.clamp_min(1)
        )

        sample_count = min(num_goal_positions, time_steps)
        gumbel = -torch.log(-torch.log(torch.rand_like(probs_2d).clamp(EPS, 1.0 - EPS)))
        masked_scores = torch.where(
            probs_2d > 0,
            torch.log(probs_2d) + gumbel,
            torch.full_like(probs_2d, float("-inf")),
        )
        sampled_ranked = masked_scores.topk(sample_count, dim=-1).indices.view(
            batch_size, max_agents, sample_count
        )
        if sample_count < num_goal_positions:
            pad = sampled_ranked[:, :, -1:].expand(
                -1, -1, num_goal_positions - sample_count
            )
            sampled_ranked = torch.cat([sampled_ranked, pad], dim=-1)

        goal_slot_is_generated = torch.arange(num_goal_positions, device=device).view(
            1, 1, -1
        ) < generated_goal_counts.unsqueeze(-1)
        sampled_multi = torch.where(
            goal_slot_is_generated,
            sampled_ranked,
            torch.full_like(sampled_ranked, time_steps),
        )
        sampled_multi, _ = sampled_multi.sort(dim=-1)
        last_generated_idx = generated_goal_counts.clamp_min(1).sub(1).unsqueeze(-1)
        last_sampled_t = sampled_multi.gather(dim=-1, index=last_generated_idx)
        sampled_t_all = torch.where(
            goal_slot_is_generated,
            sampled_multi,
            last_sampled_t.expand(-1, -1, num_goal_positions),
        )

        idx = sampled_t_all.unsqueeze(-1).expand(-1, -1, -1, 2)
        sampled_positions = torch.gather(positions, dim=2, index=idx)
    else:
        raise ValueError(f"Unsupported sampling mode: {mode}.")

    if offset_sigma_major < 0 or offset_sigma_minor < 0:
        raise ValueError("offset sigmas must be non-negative.")

    if offset_sigma_major == 0.0 and offset_sigma_minor == 0.0:
        offset_world = torch.zeros(
            (batch_size, max_agents, num_goal_positions, 2),
            device=device,
            dtype=positions.dtype,
        )
    else:
        dx = (
            torch.randn((batch_size, max_agents, num_goal_positions), device=device)
            * offset_sigma_major
        )
        dy = (
            torch.randn((batch_size, max_agents, num_goal_positions), device=device)
            * offset_sigma_minor
        )
        offset_local = torch.stack([dx, dy], dim=-1)  # [B, A, G, 2 xy]
        last_generated_idx = (
            generated_goal_counts.clamp_min(1)
            .sub(1)
            .view(batch_size, max_agents, 1, 1)
            .expand(-1, -1, 1, 2)
        )
        last_offset = offset_local.gather(dim=2, index=last_generated_idx)
        offset_local = torch.where(
            goal_slot_is_generated.unsqueeze(-1),
            offset_local,
            last_offset.expand(-1, -1, num_goal_positions, -1),
        )

        sampled_orientations = torch.gather(orientations, dim=2, index=sampled_t_all)
        offset_world = rotate_2d_points(offset_local, sampled_orientations)

    goal_positions = sampled_positions + offset_world  # [B, A, G, 2]

    # Optional goal snapping: keep the raw goal with probability
    # ``goal_snap_keep_prob``, otherwise project it onto one of the K nearest
    # lane-centre polylines. Snap weights are ∝ 1/distance so closer lanes are
    # more likely. The motivation is to encourage goals that lie on drivable
    # paths rather than arbitrary log waypoints, while still letting the model
    # see the original distribution part of the time.
    if goal_snap_keep_prob is not None and goal_snap_k > 0:
        if not (0.0 <= goal_snap_keep_prob <= 1.0):
            raise ValueError("goal_snap_keep_prob must lie in [0, 1].")
        if lanes_centers_points is None or lanes_centers_mask is None:
            raise ValueError(
                "goal_snap_keep_prob requires lanes_centers_points and "
                "lanes_centers_mask."
            )
        flat_goal_count = max_agents * num_goal_positions
        flat_goal_positions = goal_positions.reshape(batch_size, flat_goal_count, 2)
        lane_projections, lane_distances = _project_goals_to_nearest_lanes(
            flat_goal_positions,
            lanes_centers_points,
            lanes_centers_mask.bool(),
            goal_snap_k,
        )
        # Weights for the K snap candidates are ``(1 - dist/max_dist)^2``,
        # a quadratic decay that reuses ``goal_snap_max_distance`` as the
        # length scale. This avoids the ``1/dist`` blow-up that otherwise
        # concentrates probability on the very closest lane, while still
        # giving near lanes a meaningful edge (e.g. at max=20 the weights for
        # 0.1 / 3 / 8 / 10 m are roughly 0.42 / 0.31 / 0.15 / 0.11 after
        # normalisation). Invalid/padded slots and lanes beyond the radius get
        # zero weight, so agents with no lane inside the radius fall back to
        # the raw goal.
        within_radius = lane_distances <= goal_snap_max_distance
        closeness = (1.0 - lane_distances / max(goal_snap_max_distance, EPS)).clamp_min(
            0.0
        )
        inv_dist = torch.where(
            within_radius & torch.isfinite(lane_distances),
            closeness * closeness,
            torch.zeros_like(lane_distances),
        )  # [B, A, K]
        lane_weight_sum = inv_dist.sum(dim=-1, keepdim=True)  # [B, A, 1]
        # If no valid lane exists for a goal, force "keep raw" so the fallback
        # multinomial doesn't see an all-zero row.
        no_valid_lane = lane_weight_sum.squeeze(-1) <= 0  # [B, A]
        lane_probs = inv_dist / lane_weight_sum.clamp_min(EPS)  # [B, A, K]

        snap_prob = 1.0 - goal_snap_keep_prob
        keep_prob_t = torch.full(
            (batch_size, flat_goal_count, 1), goal_snap_keep_prob, device=device
        )
        snap_prob_t = torch.full(
            (batch_size, flat_goal_count, 1), snap_prob, device=device
        )
        keep_prob_t = torch.where(
            no_valid_lane.unsqueeze(-1), torch.ones_like(keep_prob_t), keep_prob_t
        )
        snap_prob_t = torch.where(
            no_valid_lane.unsqueeze(-1), torch.zeros_like(snap_prob_t), snap_prob_t
        )
        full_weights = torch.cat(
            [keep_prob_t, snap_prob_t * lane_probs], dim=-1
        )  # [B, A*G, K+1]

        candidates = torch.cat(
            [flat_goal_positions.unsqueeze(2), lane_projections], dim=2
        )  # [B, A*G, K+1, 2]

        flat_weights = full_weights.reshape(batch_size * flat_goal_count, -1)
        # Rows with zero sum (shouldn't happen thanks to the no-valid-lane
        # guard above, but be safe) fall back to the raw goal.
        row_sums = flat_weights.sum(dim=-1, keepdim=True)
        zero_rows = row_sums <= 0
        if zero_rows.any():
            fallback = torch.zeros_like(flat_weights)
            fallback[:, 0] = 1.0
            flat_weights = torch.where(zero_rows, fallback, flat_weights)

        choice = torch.multinomial(flat_weights, num_samples=1).view(
            batch_size, flat_goal_count, 1, 1
        )
        flat_goal_positions = candidates.gather(2, choice.expand(-1, -1, 1, 2)).squeeze(
            2
        )
        goal_positions = flat_goal_positions.reshape(
            batch_size, max_agents, num_goal_positions, 2
        )

    last_generated_idx = (
        generated_goal_counts.clamp_min(1)
        .sub(1)
        .view(batch_size, max_agents, 1, 1)
        .expand(-1, -1, 1, 2)
    )
    last_generated_goal = goal_positions.gather(dim=2, index=last_generated_idx)
    goal_positions = torch.where(
        goal_slot_is_generated.unsqueeze(-1),
        goal_positions,
        last_generated_goal.expand(-1, -1, num_goal_positions, -1),
    )

    return last_valid_positions, goal_positions.reshape(
        batch_size, max_agents, num_goal_positions * 2
    )


def route_goal_positions(
    positions: torch.Tensor,
    velocities: torch.Tensor,
    route_points: torch.Tensor,
    route_mask: torch.Tensor,
    *,
    horizon_s: float = 12.0,
    min_speed_mps: float = 5.0,
    num_goal_positions: int = 1,
) -> torch.Tensor:
    """Sample evenly spaced route goals ending at ``speed * horizon_s`` ahead."""
    if num_goal_positions < 1:
        raise ValueError("num_goal_positions must be >= 1.")
    if route_points.dim() != 3 or route_mask.dim() != 2 or route_points.shape[1] < 2:
        return (
            positions.unsqueeze(2)
            .expand(-1, -1, num_goal_positions, -1)
            .reshape(*positions.shape[:-1], num_goal_positions * 2)
        )

    N, A, _ = positions.shape
    route_mask = route_mask.bool()
    seg_start = route_points[:, :-1]
    seg_vec = route_points[:, 1:] - seg_start
    seg_valid = route_mask[:, :-1] & route_mask[:, 1:]
    seg_len = torch.linalg.norm(seg_vec, dim=-1)
    seg_valid = seg_valid & (seg_len > EPS)
    seg_len = torch.where(seg_valid, seg_len, torch.zeros_like(seg_len))

    cum_len = torch.cat(
        [
            torch.zeros((N, 1), dtype=seg_len.dtype, device=seg_len.device),
            torch.cumsum(seg_len, dim=1)[:, :-1],
        ],
        dim=1,
    )

    pos_exp = positions.unsqueeze(2)
    start_exp = seg_start.unsqueeze(1)
    vec_exp = seg_vec.unsqueeze(1)
    seg_len_exp = seg_len.unsqueeze(1)
    seg_valid_exp = seg_valid.unsqueeze(1).expand(N, A, -1)

    rel = pos_exp - start_exp
    t = (rel * vec_exp).sum(dim=-1) / (seg_len_exp.square() + EPS)
    t = t.clamp(0.0, 1.0)
    projected = start_exp + t.unsqueeze(-1) * vec_exp
    dist = torch.linalg.norm(projected - pos_exp, dim=-1)
    dist = torch.where(seg_valid_exp, dist, torch.full_like(dist, float("inf")))
    min_idx = dist.argmin(dim=-1)
    current_progress = (
        (cum_len.unsqueeze(1) + t * seg_len_exp)
        .gather(dim=-1, index=min_idx.unsqueeze(-1))
        .squeeze(-1)
    )

    total_len = seg_len.sum(dim=1)
    speed = torch.linalg.norm(velocities, dim=-1).clamp_min(min_speed_mps)
    final_target_progress = (current_progress + speed * horizon_s).clamp(
        max=total_len.unsqueeze(1)
    )
    fractions = (
        torch.arange(
            1,
            num_goal_positions + 1,
            dtype=current_progress.dtype,
            device=current_progress.device,
        )
        / num_goal_positions
    )
    target_progress = (
        current_progress.unsqueeze(-1)
        + (final_target_progress - current_progress).unsqueeze(-1) * fractions
    )
    target_progress = torch.cat(
        [target_progress[..., :-1], final_target_progress.unsqueeze(-1)], dim=-1
    )

    seg_end = cum_len + seg_len
    target_exp = target_progress.unsqueeze(-1)
    target_seg = seg_valid_exp.unsqueeze(2) & (
        target_exp <= seg_end.unsqueeze(1).unsqueeze(2) + EPS
    )
    target_idx = target_seg.to(torch.long).argmax(dim=-1)
    chosen_start = (
        seg_start.unsqueeze(1)
        .unsqueeze(2)
        .expand(N, A, num_goal_positions, -1, 2)
        .gather(
            3,
            target_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, 1, 2),
        )
        .squeeze(3)
    )
    chosen_vec = (
        seg_vec.unsqueeze(1)
        .unsqueeze(2)
        .expand(N, A, num_goal_positions, -1, 2)
        .gather(
            3,
            target_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, 1, 2),
        )
        .squeeze(3)
    )
    chosen_len = (
        seg_len.unsqueeze(1)
        .unsqueeze(2)
        .expand(N, A, num_goal_positions, -1)
        .gather(3, target_idx.unsqueeze(-1))
        .squeeze(-1)
    )
    chosen_cum = (
        cum_len.unsqueeze(1)
        .unsqueeze(2)
        .expand(N, A, num_goal_positions, -1)
        .gather(3, target_idx.unsqueeze(-1))
        .squeeze(-1)
    )
    ratio = ((target_progress - chosen_cum) / chosen_len.clamp_min(EPS)).clamp(0.0, 1.0)
    goals = chosen_start + ratio.unsqueeze(-1) * chosen_vec

    valid_route = seg_valid.any(dim=1).view(N, 1, 1, 1)
    goals = torch.where(
        valid_route,
        goals,
        positions.unsqueeze(2).expand(-1, -1, num_goal_positions, -1),
    )
    return goals.reshape(N, A, num_goal_positions * 2)
