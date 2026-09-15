from __future__ import annotations

import math
from dataclasses import dataclass, field, fields
from typing import Optional

import torch

from driverl.datatypes import goal_position_utils
from driverl.datatypes.data_enums import AgentControlManager
from driverl.env.domain_randomization.domain_randomization import RandomizedFeatureDict
from driverl.env.engine.config import EngineRuntimeConfig
from driverl.utils.geometry import extract_recent_agent_polygons

try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except Exception:
    _TRITON_AVAILABLE = False


if _TRITON_AVAILABLE:

    @triton.jit
    def _occ_first_hit_kernel(
        occ_ptr,
        row_ptr,
        col_ptr,
        ray_idx_ptr,
        delta_row_ptr,
        delta_col_ptr,
        out_ptr,
        stride_occ_b,
        stride_row_r,
        stride_row_s,
        stride_col_r,
        stride_col_s,
        stride_ray_b,
        stride_ray_r,
        stride_out_b,
        stride_out_r,
        B,
        R,
        H,
        W,
        res,
        S: tl.constexpr,
        BLOCK_R: tl.constexpr,
        BLOCK_S: tl.constexpr,
    ):
        pid_b = tl.program_id(0)
        pid_r = tl.program_id(1)

        ray_offsets = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
        ray_mask = ray_offsets < R
        ray = tl.load(
            ray_idx_ptr + pid_b * stride_ray_b + ray_offsets * stride_ray_r,
            mask=ray_mask,
            other=0,
        )
        delta_row = tl.load(delta_row_ptr + pid_b)
        delta_col = tl.load(delta_col_ptr + pid_b)

        min_idx = tl.full((BLOCK_R,), S, tl.int32)
        offs = tl.arange(0, BLOCK_S)
        for s_base in tl.static_range(0, S, BLOCK_S):
            idx = s_base + offs
            mask = ray_mask[:, None] & (idx[None, :] < S)
            row = tl.load(
                row_ptr + ray[:, None] * stride_row_r + idx[None, :] * stride_row_s,
                mask=mask,
                other=0,
            )
            col = tl.load(
                col_ptr + ray[:, None] * stride_col_r + idx[None, :] * stride_col_s,
                mask=mask,
                other=0,
            )
            row = row + delta_row
            col = col + delta_col
            in_bounds = (row >= 0) & (row < H) & (col >= 0) & (col < W) & mask
            flat = row * W + col
            occ = tl.load(
                occ_ptr + pid_b * stride_occ_b + flat,
                mask=in_bounds,
                other=0,
            )
            hit = in_bounds & (occ > 0)
            idx_hit = tl.where(hit, idx[None, :], S)
            min_idx = tl.minimum(min_idx, tl.min(idx_hit, axis=1))

        out = tl.where(min_idx < S, min_idx.to(tl.float32) * res, -1.0)
        tl.store(
            out_ptr + pid_b * stride_out_b + ray_offsets * stride_out_r,
            out,
            mask=ray_mask,
        )


def _extract_occ_distances_triton(
    occ_grid: torch.Tensor,
    row: torch.Tensor,
    col: torch.Tensor,
    ray_indices: torch.Tensor,
    delta_row: torch.Tensor,
    delta_col: torch.Tensor,
    res: float,
    block_s: int = 256,
    block_r: int = 8,
) -> torch.Tensor:
    B, H, W = occ_grid.shape
    R = ray_indices.shape[1]
    S = row.shape[1]

    out = torch.empty((B, R), device=occ_grid.device, dtype=torch.float32)

    grid = (B, triton.cdiv(R, block_r))
    _occ_first_hit_kernel[grid](
        occ_grid,
        row,
        col,
        ray_indices,
        delta_row,
        delta_col,
        out,
        occ_grid.stride(0),
        row.stride(0),
        row.stride(1),
        col.stride(0),
        col.stride(1),
        ray_indices.stride(0),
        ray_indices.stride(1),
        out.stride(0),
        out.stride(1),
        B,
        R,
        H,
        W,
        res,
        S=S,
        BLOCK_R=block_r,
        BLOCK_S=block_s,
        num_warps=8,
        num_stages=2,
    )
    return out


@dataclass
class ScenarioData:
    """A dataclass to store the state of the environment for autonomous vehicle simulation.

    This dataclass holds all relevant information about agents and the environment from a scene.
    The data is divided into static and dynamic categories. Static data remains constant
    throughout a simulation rollout, while dynamic data is appended at each time step.

    Attributes:
        npc_id_array (torch.Tensor): A tensor containing unique IDs for each non-player character (NPC).
            Shape: `[batch_size, max_agents]`
        controlled_agent_mask (torch.Tensor): A boolean mask indicating which agents are controlled by the policy.
            Shape: `[batch_size, max_agents]`
        lanes_points (torch.Tensor): Batched lane/boundary polylines with semantic type and speed-limit metadata.
            Shape: `[batch_size, num_lanes, num_points_per_lane, 4]`
            where the last dimension stores `(x, y, type, speed_limit)`.
        lanes_points_mask (torch.Tensor): A boolean mask for `lanes_points`, indicating valid lane polylines.
            Shape: `[batch_size, num_lanes]`
        lanes_centers_points (torch.Tensor): Batched lane-center polylines with per-point metadata.
            Shape: `[batch_size, num_lane_centers, num_points_per_lane_center, 4]`
        lanes_centers_mask (torch.Tensor): A boolean mask for `lanes_centers_points`.
            Shape: `[batch_size, num_lane_centers]`
        route_points_array (torch.Tensor): Route points data containing interpolated waypoints along the planned route.
            Shape: `[batch_size, max_route_points, 2]` for [longitude, latitude] coordinates in ego-centric frame
        route_points_mask (torch.Tensor): A boolean mask for `route_points_array`, indicating valid route points.
            Shape: `[batch_size, max_route_points]`
        occupancy_grid (torch.Tensor): Ego-centric occupancy grid at init step.
            Shape: `[batch_size, grid_height, grid_width]`
        agent_positions_all (torch.Tensor): A time series of agent positions (x, y).
            Shape: `[batch_size, max_agents, time_steps, 2]`
        agent_velocity_all (torch.Tensor): A time series of agent velocities (vx, vy).
            Shape: `[batch_size, max_agents, time_steps, 2]`
        agent_size_all (torch.Tensor): A time series of agent sizes (length, width).
            Shape: `[batch_size, max_agents, time_steps, 2]`
        agent_orientation_all (torch.Tensor): A time series of agent orientations (yaw).
            Shape: `[batch_size, max_agents, time_steps]`
        agent_type_all (torch.Tensor): A time series of agent types (e.g., car, pedestrian).
            Shape: `[batch_size, max_agents, time_steps]`
        npc_mask_all (torch.Tensor): A boolean mask indicating valid (non-padded) NPCs at each time step.
            Shape: `[batch_size, max_agents, time_steps]`
        agent_nearest_indices (torch.Tensor): Indices of nearest neighboring agents for each agent at each timestep,
            calculated using k-nearest neighbors.
            Shape: `[batch_size, max_agents, time_steps, max_agents]`
        occ_surface_points_all (torch.Tensor): Cached ego-ray hit points in ego frame.
            Shape: `[batch_size, 1, time_steps, num_rays, 2]`
        agent_acceleration_state_all (torch.Tensor): Longitudinal acceleration state fed into the dynamics model.
            Shape: `[batch_size, max_agents, time_steps]`
        agent_acceleration_control_all (torch.Tensor): Longitudinal acceleration control output by the dynamics model.
            Shape: `[batch_size, max_agents, time_steps]`
        agent_yaw_rate_all (torch.Tensor): A time series of yaw rates (rad/s) emitted by the dynamics model.
            Shape: `[batch_size, max_agents, time_steps]`
        agent_steering_state_all (torch.Tensor): Steering angle state fed into the dynamics model.
            Shape: `[batch_size, max_agents, time_steps]`
        agent_steering_control_all (torch.Tensor): Steering angle control output by the dynamics model.
            Shape: `[batch_size, max_agents, time_steps]`
        goal_positions (torch.Tensor): Per-frame per-agent target goal positions, used for
            reward calculation and model input. Grows with time via ``append`` — shape mirrors
            ``agent_positions_all`` exactly. On a rollout ``scenario_data`` this is the
            ``init_steps + 1`` window extended one frame per ``engine.step``; on
            ``log_scenario_data`` this is the full ``T_total`` timeline resolved by
            :meth:`from_log` (``"log"`` mode keeps ego's per-frame log goals; other modes
            broadcast sampled goals across time). The final dimension stores a
            flat sequence of `(x, y)` target points.
            Shape: `[batch_size, max_agents, time_steps, 2 * num_goal_positions]`
        frame_drop_mask_all (torch.Tensor): A boolean mask indicating which frames are dropped.
            Shape: `[batch_size, max_agents, time_steps]`
    """

    # Static data (should not change over a rollout)
    npc_id_array: torch.Tensor
    lanes_points: torch.Tensor
    lanes_points_mask: torch.Tensor
    lanes_centers_points: torch.Tensor
    lanes_centers_mask: torch.Tensor
    route_points_array: torch.Tensor
    route_points_mask: torch.Tensor
    occupancy_grid: torch.Tensor

    # Dynamic data (appended at each step)
    agent_positions_all: torch.Tensor
    agent_velocity_all: torch.Tensor
    agent_size_all: torch.Tensor
    agent_orientation_all: torch.Tensor
    agent_type_all: torch.Tensor
    npc_mask_all: torch.Tensor
    lanes_centers_groups: torch.Tensor = field(default_factory=torch.Tensor)
    lanes_centers_ids: torch.Tensor = field(default_factory=torch.Tensor)
    lanes_centers_next_groups: torch.Tensor = field(default_factory=torch.Tensor)
    agent_nearest_indices: torch.Tensor = field(default_factory=torch.Tensor)
    advantage_filtering_mask: torch.Tensor | None = None

    # Stitched lane polylines (optional, only available with group data).
    lanes_centers_lines: torch.Tensor = field(default_factory=torch.Tensor)
    lanes_centers_lines_mask: torch.Tensor = field(default_factory=torch.Tensor)
    lanes_centers_tl_states: torch.Tensor = field(default_factory=torch.Tensor)
    lanes_centers_tl_masks: torch.Tensor = field(default_factory=torch.Tensor)
    lanes_tl_states: torch.Tensor = field(default_factory=torch.Tensor)
    lanes_tl_masks: torch.Tensor = field(default_factory=torch.Tensor)
    route_lane_connector_polygon_vertices: torch.Tensor = field(
        default_factory=torch.Tensor
    )
    route_lane_connector_polygon_offsets: torch.Tensor = field(
        default_factory=torch.Tensor
    )
    route_lane_connector_bboxes: torch.Tensor = field(default_factory=torch.Tensor)
    route_lane_connector_tl_status_codes: torch.Tensor = field(
        default_factory=torch.Tensor
    )
    route_lane_connector_ids: torch.Tensor = field(default_factory=torch.Tensor)
    route_lane_connector_batch_indices: torch.Tensor = field(
        default_factory=torch.Tensor
    )
    route_lane_connector_world_offsets: torch.Tensor = field(
        default_factory=torch.Tensor
    )
    route_lane_connector_data_available: torch.Tensor = field(
        default_factory=torch.Tensor
    )

    agent_acceleration_state_all: torch.Tensor = field(default_factory=torch.Tensor)
    agent_acceleration_control_all: torch.Tensor = field(default_factory=torch.Tensor)
    agent_steering_state_all: torch.Tensor = field(default_factory=torch.Tensor)
    agent_steering_control_all: torch.Tensor = field(default_factory=torch.Tensor)
    agent_jerk_lat_all: torch.Tensor = field(default_factory=torch.Tensor)
    agent_jerk_long_all: torch.Tensor = field(default_factory=torch.Tensor)
    agent_yaw_rate_all: torch.Tensor = field(default_factory=torch.Tensor)
    agent_goal_reached_all: torch.Tensor = field(default_factory=torch.Tensor)
    goal_stage: torch.Tensor = field(default_factory=torch.Tensor)  # [N, A]
    agent_not_blind_mask: torch.Tensor = field(default_factory=torch.Tensor)
    agent_visible_mask: torch.Tensor = field(
        default_factory=torch.Tensor
    )  # [N, A, T, A]
    debug_weight: torch.Tensor = field(default_factory=torch.Tensor)  # [N, A, T, A]
    occ_surface_points_all: torch.Tensor = field(
        default_factory=torch.Tensor
    )  # [N, 1, T, R, 2]

    # Reward data
    goal_positions: torch.Tensor = field(default_factory=torch.Tensor)
    goal_pair_mode: str = "legacy"

    _agent_control_manager: AgentControlManager | None = field(default=None)
    _randomized_features: RandomizedFeatureDict | None = field(default=None)

    # Valid frames
    frame_drop_mask_all: torch.Tensor = field(default_factory=torch.Tensor)

    # Cache for Polygons, only maintain in one step for different reward calculation
    _polygons_now: Optional[torch.Tensor] = field(default=None, init=False, repr=False)
    _polygons_last: Optional[torch.Tensor] = field(default=None, init=False, repr=False)
    _valid_mask: Optional[torch.Tensor] = field(default=None, init=False, repr=False)
    _cross_lane_deadband_counter: Optional[torch.Tensor] = field(
        default=None, init=False, repr=False
    )  # [N, A]
    _direction_progress_buffer: Optional[torch.Tensor] = field(
        default=None, init=False, repr=False
    )  # [N, A, D]
    _direction_progress_last: Optional[torch.Tensor] = field(
        default=None, init=False, repr=False
    )  # [N, A]
    _direction_segment_last: Optional[torch.Tensor] = field(
        default=None, init=False, repr=False
    )  # [N, A]

    def get_recent_agent_polygons(self):
        """Gets agent polygons, using a cache to avoid re-calculation."""
        if (
            self._polygons_now is None
            or self._polygons_last is None
            or self._valid_mask is None
        ):
            (
                self._polygons_now,
                self._polygons_last,
                self._valid_mask,
            ) = extract_recent_agent_polygons(self)
        return self._polygons_now, self._polygons_last, self._valid_mask

    def clean_polygon_cache(self):
        """Clears the cached polygon data."""
        self._polygons_now = None
        self._polygons_last = None
        self._valid_mask = None

    def make_contiguous(self):
        """Ensures all tensor attributes are stored in contiguous memory blocks."""
        for f in fields(self):
            value = getattr(self, f.name)
            if isinstance(value, torch.Tensor):
                setattr(self, f.name, value.contiguous())
        return self

    def normalize_angle(self, x):
        return (x + 2 * torch.pi) % (2 * torch.pi)

    @property
    def randomized_features(self) -> RandomizedFeatureDict:
        """Returns the randomized features used for this scenario, if any."""
        if self._randomized_features is None:
            raise ValueError(
                "Randomized features have not been set for this ScenarioData instance."
            )
        return self._randomized_features

    @property
    def agent_control_manager(self) -> AgentControlManager:
        """Returns the AgentControlManager for this scenario, if set."""
        if self._agent_control_manager is None:
            raise ValueError(
                "AgentControlManager has not been set for this ScenarioData instance."
            )
        return self._agent_control_manager

    def update_visible_mask(self, distance_limit=250, k=5):
        positions = self.agent_positions_all[:, :, -1, :]  # [N, A, 2]
        N, A, _ = positions.shape
        device = positions.device
        polygons, _, valid = self.get_recent_agent_polygons()  # [N, A, 4, 2]

        batch_indices, agent_indices = torch.where(
            self.agent_control_manager.controlled_mask
        )
        batch_indices.shape[0]

        distance = torch.cdist(positions, positions)  # [N, A, A]
        diagonal = torch.arange(A, device=device)
        distance[:, diagonal, diagonal] = torch.inf
        distance_all = distance[batch_indices, agent_indices]  # [num_controlled, A]

        positions = positions[batch_indices, agent_indices, :]  # [num_controlled, 2]
        polygons = polygons[batch_indices, :, :, :]  # [num_controlled, A, 4, 2]

        positions_expanded = positions.unsqueeze(1).unsqueeze(
            1
        )  # [num_controlled, 1, 1, 2]

        angle = (
            torch.atan2(
                polygons[:, :, :, 1] - positions_expanded[:, :, :, 1],
                polygons[:, :, :, 0] - positions_expanded[:, :, :, 0],
            )
            + torch.pi
        )  # [num_controlled, A, 4]
        # range: [0, 2pi]
        max_angle = torch.max(angle, dim=-1).values  # [num_controlled, A]
        min_angle = torch.min(angle, dim=-1).values  # [num_controlled, A]
        boundary_tag = angle <= torch.pi
        max_angle_boundary = torch.max(
            torch.where(boundary_tag, angle, -100.0), dim=-1
        ).values
        min_angle_boundary = torch.min(
            torch.where(~boundary_tag, angle, 100), dim=-1
        ).values

        tag = max_angle - min_angle > torch.pi
        range_L = torch.where(tag, min_angle_boundary, min_angle)
        range_R = torch.where(tag, max_angle_boundary, max_angle)

        L_i = range_L.unsqueeze(-1)  # [num_controlled, A, 1]
        L_j = range_L.unsqueeze(-2)  # [num_controlled, 1, A]
        R_i = range_R.unsqueeze(-1)  # [num_controlled, A, 1]
        R_j = range_R.unsqueeze(-2)  # [num_controlled, 1, A]
        dis_i = distance_all.unsqueeze(-1)  # [num_controlled, A, 1]
        dis_j = distance_all.unsqueeze(-2)  # [num_controlled, 1, A]

        weights = torch.arange(k, device=device)
        range_length = torch.where(
            R_i > L_i, (R_i - L_i) / k, (2 * torch.pi + R_i - L_i) / k
        ).unsqueeze(-1)  # [num_controlled, A, 1, 1]
        # [num_controlled, A, 1, 1] * [k, 1]
        L_i_k = self.normalize_angle(
            L_i.unsqueeze(-1) + range_length * (weights.unsqueeze(-1))
        )  # [num_controlled, A, k, 1]
        L_j_k = L_j.unsqueeze(-2)  # [num_controlled, 1, 1, A]
        R_j_k = R_j.unsqueeze(-2)  # [num_controlled, 1, 1, A]

        # 为了方便划分和计算，把L_i旋转到0°的位置
        L_j_k = self.normalize_angle(L_j_k - L_i_k)  # [num_controlled, A, k, A]
        R_j_k = self.normalize_angle(R_j_k - L_i_k)  # [num_controlled, A, k, A]

        intersect_mask = (
            (L_j_k < range_length) | (R_j_k < range_length) | (L_j_k > R_j_k)
        )  # [num_controlled, A, k, A]

        valid_all = valid[batch_indices]
        valid_mask = valid_all.unsqueeze(-1) & valid_all.unsqueeze(
            -2
        )  # [num_controlled, A, A]
        dist_mask = (
            dis_i > dis_j
        )  # i can be occluded by j when j is nearer [num_controlled, A, A]

        intersect_mask = (
            intersect_mask & valid_mask.unsqueeze(-2) & dist_mask.unsqueeze(-2)
        )  # [num_controlled, A, k, A]
        occlusion_weight = (
            1.0 * (intersect_mask.any(-1)).sum(-1) / k
        )  # [num_controlled, A]
        occlusion_mask = torch.rand_like(occlusion_weight) < occlusion_weight

        weight_matrix = torch.zeros((N, A, A), device=device, dtype=torch.float)
        mask_matrix = torch.zeros((N, A, A), device=device, dtype=torch.bool)

        weight_matrix[batch_indices, agent_indices] = occlusion_weight
        mask_matrix[batch_indices, agent_indices] = ~occlusion_mask
        mask_matrix = mask_matrix & (distance <= distance_limit)

        weight_matrix[:, diagonal, diagonal] = 0
        mask_matrix[:, diagonal, diagonal] = True

        if self.agent_visible_mask.numel() == 0:
            self.agent_visible_mask = torch.zeros(
                [N, A, 0, A], device=device, dtype=torch.bool
            )
        if self.debug_weight.numel() == 0:
            self.debug_weight = torch.zeros(
                [N, A, 0, A], device=device, dtype=torch.float
            )
        self.agent_visible_mask = torch.cat(
            [self.agent_visible_mask, mask_matrix.unsqueeze(-2)], dim=-2
        )
        self.debug_weight = torch.cat(
            [self.debug_weight, weight_matrix.unsqueeze(-2)], dim=-2
        )

    @staticmethod
    def _get_occ_ray_cache(
        grid_shape: tuple[int, int],
        device: torch.device,
        config: EngineRuntimeConfig,
        occ_ray_cache: dict,
    ) -> dict:
        H, W = grid_shape
        if occ_ray_cache:
            return occ_ray_cache
        cache_rays = config.occ_num_rays * 2

        res = config.occ_grid_resolution
        xmax = config.occ_grid_xmax
        ymax = config.occ_grid_ymax
        max_steps = max(H, W) + 1
        if config.occ_grid_max_range is not None:
            max_steps = min(
                max_steps,
                int(round(config.occ_grid_max_range / res)) + 1,
            )

        dirs_theta = (
            torch.arange(cache_rays, device=device, dtype=torch.float32)
            * (2.0 * math.pi / cache_rays)
            - math.pi
        )
        dirs = torch.stack([torch.cos(dirs_theta), torch.sin(dirs_theta)], dim=-1)
        steps = torch.arange(max_steps, device=device, dtype=torch.float32) * res
        pos = torch.zeros(cache_rays, max_steps, 2, device=device)
        pos[..., 0] = steps * dirs[:, 0:1]
        pos[..., 1] = steps * dirs[:, 1:2]

        row = torch.floor((xmax - pos[..., 0]) / res).to(torch.int16)
        col = torch.floor((ymax - pos[..., 1]) / res).to(torch.int16)
        cached = {
            "row": row.to(torch.int32),
            "col": col.to(torch.int32),
            "ray_step": 2.0 * math.pi / cache_rays,
            "cache_rays": cache_rays,
            "H": H,
            "W": W,
            "ray_offsets": torch.arange(
                -cache_rays // 4,
                cache_rays // 2 - cache_rays // 4,
                device=device,
            ),
        }
        occ_ray_cache.update(cached)
        return occ_ray_cache

    @staticmethod
    def _compute_occ_shift(
        ego_pos: torch.Tensor, res: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Grid index uses x = xmax - (row+0.5)*res, y = ymax - (col+0.5)*res,
        # so positive world x/y correspond to negative row/col shifts.
        shift = torch.round(-ego_pos / res).to(torch.long)
        return shift[..., 0], shift[..., 1]

    @staticmethod
    def _get_occ_ray_indices(yaw: torch.Tensor, cache: dict) -> torch.Tensor:
        yaw = torch.atan2(torch.sin(yaw), torch.cos(yaw))
        ray_step = cache["ray_step"]
        cache_rays = cache["cache_rays"]
        center = torch.round((yaw + math.pi) / ray_step).long() % cache_rays
        offsets = cache["ray_offsets"].to(device=yaw.device)
        return (center.unsqueeze(-1) + offsets) % cache_rays

    @staticmethod
    def extract_occ_points(
        occ_grid: torch.Tensor,  # [B, H, W] uint8
        ego_pos: torch.Tensor,  # [B, 2] world
        ego_yaw: torch.Tensor,  # [B] world
        config: EngineRuntimeConfig,
        occ_ray_cache: dict,
    ) -> torch.Tensor:
        """Ray cast against the occupancy grid and return first hit points per ray.

        Notes:
        - Rays cover a 180deg forward FOV centered at ego_yaw.
        - Points are in ego-centric frame. Rays with no hit return (-1, -1).
        - `occ_ray_cache` must contain precomputed ray lookup tensors:
          `row`, `col`, `ray_step`, `cache_rays`, `ray_offsets`.
        """
        if occ_grid.numel() == 0 or occ_grid.dim() != 3:
            B = ego_pos.shape[0]
            distances = torch.full(
                (B, config.occ_num_rays),
                -1.0,
                device=ego_pos.device,
                dtype=torch.float32,
            )
            return ScenarioData._distances_to_occ_points(distances)

        B, H, W = occ_grid.shape
        cache = occ_ray_cache
        row = cache["row"]  # [cache_rays, S]
        col = cache["col"]  # [cache_rays, S]

        # 180deg ray indices centered at ego_yaw.
        ray_indices = ScenarioData._get_occ_ray_indices(ego_yaw, cache)  # [B, R]

        # Shift rays by ego translation in grid coords.
        delta_row, delta_col = ScenarioData._compute_occ_shift(
            ego_pos, config.occ_grid_resolution
        )

        if (
            _TRITON_AVAILABLE
            and occ_grid.is_cuda
            and not torch.onnx.is_in_onnx_export()
        ):
            distances = _extract_occ_distances_triton(
                occ_grid,
                row,
                col,
                ray_indices.to(torch.int32),
                delta_row.to(torch.int32),
                delta_col.to(torch.int32),
                config.occ_grid_resolution,
            )
            return ScenarioData._distances_to_occ_points(distances)

        row_shift = row.unsqueeze(0) + delta_row.view(B, 1, 1)  # [B, cache_rays, S]
        col_shift = col.unsqueeze(0) + delta_col.view(B, 1, 1)  # [B, cache_rays, S]

        in_bounds = (
            (row_shift >= 0) & (row_shift < H) & (col_shift >= 0) & (col_shift < W)
        )
        flat_idx = row_shift * W + col_shift
        flat_idx = torch.where(in_bounds, flat_idx, torch.full_like(flat_idx, -1))
        valid_mask = flat_idx >= 0

        occ_flat = occ_grid.view(B, -1)
        flat_idx_safe = flat_idx.clamp(min=0, max=H * W - 1).long()
        occ_vals = occ_flat.gather(1, flat_idx_safe.view(B, -1)).view_as(flat_idx)
        occ_hit = (occ_vals > 0) & valid_mask
        occ_hit = occ_hit.gather(
            1, ray_indices.unsqueeze(-1).expand(-1, -1, occ_hit.shape[2])
        )

        hit_any = occ_hit.any(dim=2)
        hit_idx = occ_hit.to(torch.int64).argmax(dim=2)
        distances = hit_idx.to(torch.float32) * config.occ_grid_resolution
        distances = torch.where(hit_any, distances, torch.full_like(distances, -1.0))
        return ScenarioData._distances_to_occ_points(distances)

    @staticmethod
    def _distances_to_occ_points(distances: torch.Tensor) -> torch.Tensor:
        """Convert per-ray distances to ego-centric hit points.

        Invalid rays (distance < 0) are filled with (-1, -1).
        """
        R = distances.shape[1]
        ray_step = math.pi / R
        theta = (
            torch.arange(R, device=distances.device, dtype=distances.dtype) - (R / 2)
        ) * ray_step
        ray_dir = torch.stack([torch.cos(theta), torch.sin(theta)], dim=-1)  # [R, 2]
        points = distances.unsqueeze(-1) * ray_dir  # [B, R, 2]
        invalid = distances < 0
        points = torch.where(
            invalid.unsqueeze(-1), torch.full_like(points, -1.0), points
        )
        return points

    def update_occ_surface_points(
        self, config: EngineRuntimeConfig, occ_ray_cache: dict
    ):
        """Compute and cache ego ray hit points for the current step."""

        B = self.agent_positions_all.shape[0]
        if not getattr(config, "enable_occupancy_grid", True):
            points = torch.full(
                (B, config.occ_num_rays, 2),
                -1.0,
                device=self.agent_positions_all.device,
            )
            self._append_occ_surface_points(points)
            return

        occ_grid = self.occupancy_grid
        device = (
            occ_grid.device if occ_grid.numel() > 0 else self.agent_positions_all.device
        )

        if occ_grid.numel() == 0 or occ_grid.dim() != 3:
            points = torch.full((B, config.occ_num_rays, 2), -1.0, device=device)
            self._append_occ_surface_points(points)
            return

        B, H, W = occ_grid.shape
        cache = ScenarioData._get_occ_ray_cache((H, W), device, config, occ_ray_cache)
        ego_pos = self.agent_positions_all[:, 0, -1, :]
        ego_yaw = self.agent_orientation_all[:, 0, -1]
        points = ScenarioData.extract_occ_points(
            occ_grid, ego_pos, ego_yaw, config, cache
        )
        self._append_occ_surface_points(points)

    def _append_occ_surface_points(self, points: torch.Tensor) -> None:
        points = points.unsqueeze(1).unsqueeze(2)  # [N, 1, 1, R, 2]
        self.occ_surface_points_all = torch.cat(
            [self.occ_surface_points_all, points], dim=2
        )

    def update_nearest_neighbors(self):
        """Calculates and stores nearest neighbor indices for the current timestep.

        Downstream reward calculators only consume the latest neighbor order via
        ``agent_nearest_indices[:, :, -1, :]``. Keep a singleton time dimension
        for that interface, but do not accumulate rollout history here; storing
        ``[N, A, T, A]`` int64 indices is prohibitively expensive for large
        batches.
        """
        N, A, _, _ = self.agent_positions_all.shape
        now_positions = self.agent_positions_all[:, :, -1, :]
        dist_matrix = torch.cdist(now_positions, now_positions)  # [N, A, A]

        # Set distance to infinity for invalid agents to exclude them from NN search
        invalid_mask = ~self.npc_mask_all[:, :, -1]
        dist_matrix[invalid_mask.unsqueeze(1).expand(-1, A, -1)] = float("inf")
        dist_matrix[invalid_mask.unsqueeze(2).expand(-1, -1, A)] = float(
            "inf"
        )  # [N, A, A]
        _, sorted_indices = torch.sort(
            dist_matrix, dim=-1, descending=False
        )  # [N, A, A]

        self.agent_nearest_indices = sorted_indices.unsqueeze(-2)

    def append(
        self,
        positions,
        velocities,
        yaws,
        sizes,
        masks,
        accelerations_state,
        accelerations_control,
        steerings_state,
        steerings_control,
        yaw_rates,
        jerk_lat,
        jerk_long,
        frame_drop_mask,
        goal_positions,
    ):
        """Appends a new time step to the dynamic agent data."""
        # The size information is assumed to be constant, so we reuse the latest frame's data.
        new_type = (
            self.agent_type_all[:, :, -1:]
            if self.agent_type_all.shape[2] > 0
            else torch.zeros_like(yaws.unsqueeze(2))
        )

        self.agent_positions_all = torch.cat(
            [self.agent_positions_all, positions], dim=2
        )
        self.agent_velocity_all = torch.cat(
            [self.agent_velocity_all, velocities], dim=2
        )
        self.agent_orientation_all = torch.cat(
            [self.agent_orientation_all, yaws], dim=2
        )
        self.npc_mask_all = torch.cat([self.npc_mask_all, masks], dim=2)
        self.agent_size_all = torch.cat([self.agent_size_all, sizes], dim=2)
        self.agent_type_all = torch.cat([self.agent_type_all, new_type], dim=2)
        self.agent_acceleration_state_all = torch.cat(
            [self.agent_acceleration_state_all, accelerations_state], dim=2
        )
        self.agent_acceleration_control_all = torch.cat(
            [self.agent_acceleration_control_all, accelerations_control], dim=2
        )
        self.agent_steering_state_all = torch.cat(
            [self.agent_steering_state_all, steerings_state], dim=2
        )
        self.agent_steering_control_all = torch.cat(
            [self.agent_steering_control_all, steerings_control], dim=2
        )
        self.agent_yaw_rate_all = torch.cat([self.agent_yaw_rate_all, yaw_rates], dim=2)
        self.agent_jerk_lat_all = torch.cat([self.agent_jerk_lat_all, jerk_lat], dim=2)
        self.agent_jerk_long_all = torch.cat(
            [self.agent_jerk_long_all, jerk_long], dim=2
        )
        self.frame_drop_mask_all = torch.cat(
            [self.frame_drop_mask_all, frame_drop_mask], dim=2
        )
        num_goal_positions = (
            self.goal_positions.shape[-1] // 2
            if self.goal_positions.numel() > 0
            else None
        )
        goal_positions = goal_position_utils.normalize_goal_positions(
            goal_positions, num_goal_positions
        )
        self.goal_positions = torch.cat([self.goal_positions, goal_positions], dim=2)

    def update_goal_reached(self, goal_reached) -> None:
        goal_reached |= self.agent_goal_reached_all[..., -1]
        self.agent_goal_reached_all = torch.cat(
            [self.agent_goal_reached_all, goal_reached.unsqueeze(-1)], dim=2
        )
