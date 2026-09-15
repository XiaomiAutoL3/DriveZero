import torch

from driverl.datatypes.data_enums import LaneType
from driverl.datatypes.scenario_data import ScenarioData
from driverl.env import constants
from driverl.env.engine.config import EngineMergedConfig
from driverl.env.engine.reward_calculator.base_reward_calculator import (
    REWARD_CALCULATOR_REGISTER,
    BaseRewardCalculator,
)
from driverl.utils.geometry import (
    batch_segment_intersection,
    expand_polygon_with_buffer,
    project_points_to_segments,
)


def judge_solid_line(lanes_info, mask):
    return (
        (lanes_info == LaneType.SOLID_LINE.code)
        | (lanes_info == LaneType.DOUBLE_SOLID_LINE.code)
    ) & mask


@REWARD_CALCULATOR_REGISTER.register_module
class OffRoad(BaseRewardCalculator):
    def __init__(self, config: EngineMergedConfig):
        """Initialize the off road reward calculator."""
        super().__init__(config)
        self.collision_speed_buffer_base = getattr(
            config, "collision_speed_buffer_base", 0.0
        )
        self.collision_speed_buffer_max = getattr(
            config, "collision_speed_buffer_max", 0.0
        )
        self.collision_relative_speed_buffer_gain = getattr(
            config, "collision_relative_speed_buffer_gain", 0.0
        )

    def _compute_collision_speed_buffer(
        self, speed: torch.Tensor, gain: float
    ) -> torch.Tensor:
        return (self.collision_speed_buffer_base + gain * speed).clamp(
            max=self.collision_speed_buffer_max
        )

    @staticmethod
    def _build_vehicle_segments(poly_now: torch.Tensor, poly_last: torch.Tensor):
        """Build motion + box-edge segments used by boundary intersection checks."""
        motion_segments = torch.cat(
            [poly_now.unsqueeze(-2), poly_last.unsqueeze(-2)], dim=-2
        )  # [N, A, 4, 2, 2]
        front_segments = torch.cat(
            [
                poly_now[:, :, 0:1].unsqueeze(-2),
                poly_now[:, :, 1:2].unsqueeze(-2),
            ],
            dim=-2,
        )  # [N, A, 1, 2, 2]
        right_segments = torch.cat(
            [
                poly_now[:, :, 1:2].unsqueeze(-2),
                poly_now[:, :, 2:3].unsqueeze(-2),
            ],
            dim=-2,
        )  # [N, A, 1, 2, 2]
        rear_segments = torch.cat(
            [
                poly_now[:, :, 2:3].unsqueeze(-2),
                poly_now[:, :, 3:4].unsqueeze(-2),
            ],
            dim=-2,
        )  # [N, A, 1, 2, 2]
        left_segments = torch.cat(
            [
                poly_now[:, :, 3:4].unsqueeze(-2),
                poly_now[:, :, 0:1].unsqueeze(-2),
            ],
            dim=-2,
        )  # [N, A, 1, 2, 2]
        return torch.cat(
            [
                motion_segments,
                front_segments,
                right_segments,
                rear_segments,
                left_segments,
            ],
            dim=2,
        )  # [N, A, 8, 2, 2]

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

        Returns:
            A tuple containing:
            - reward (torch.Tensor): A tensor of shape (N, A) containing the reward.
            - info (torch.Tensor): An integer tensor of shape (N, A) indicating off-road and cross-lane events.
                - 0: No event.
                - 1: Agent is off-road.
                - 2: Agent has crossed a solid lane line.
        """
        # Get current positions of all agents

        polygons_now, polygons_last, valid = (
            scenario_data.get_recent_agent_polygons()
        )  # [N, A, 4, 2]

        randomized_features = scenario_data.randomized_features
        off_road_weight = randomized_features.get(
            "collision_reward_weight", calculate=True
        )
        cross_lane_weight = randomized_features.get("cross_lane_weight", calculate=True)

        speed = torch.linalg.norm(scenario_data.agent_velocity_all[:, :, -1, :], dim=-1)
        base_buffer = self._compute_collision_speed_buffer(
            speed, self.collision_relative_speed_buffer_gain
        )
        collision_width_buffer = base_buffer
        collision_length_buffer = base_buffer

        lane_points = scenario_data.lanes_points
        lane_points_mask = scenario_data.lanes_points_mask
        lane_positions = lane_points[..., :2]
        lane_types_first = lane_points[..., 0, 2]

        hard_boundary_mask = lane_points_mask & (lane_types_first == LaneType.CURB.code)
        solid_boundary_mask = judge_solid_line(lane_types_first, lane_points_mask)
        other_lane_mask = (
            lane_points_mask & (~hard_boundary_mask) & (~solid_boundary_mask)
        )

        # 0 = invalid/padding, 1 = curb boundary (off-road),
        # 2 = solid lane (cross-lane), 3 = other non-solid lanes (reserved for lane-change)
        boundary_mask = hard_boundary_mask.to(dtype=torch.int32)
        boundary_mask = torch.where(
            solid_boundary_mask,
            torch.full_like(boundary_mask, 2),
            boundary_mask,
        )
        boundary_mask = torch.where(
            other_lane_mask,
            torch.full_like(boundary_mask, 3),
            boundary_mask,
        )

        controlled_mask = (
            valid & scenario_data.agent_control_manager.controlled_mask
        )  # [N, A]
        off_road_info, cross_lane_info, lane_change_info = self.detect_cross_line_knn(
            polygons_now,
            polygons_last,
            controlled_mask,
            lane_positions,
            boundary_mask,
            collision_width_buffer,
            collision_length_buffer,
            k=12,
        )  # [N, A]

        solid_lane_change_info = cross_lane_info & lane_change_info

        # Optional deadband for plain cross-lane contact. When set to 0, only the
        # current solid-line contact frame is penalized.
        deadband_frames = max(
            int(getattr(self.config, "cross_lane_deadband_frames", 0)), 0
        )
        if deadband_frames > 0:
            if (
                scenario_data._cross_lane_deadband_counter is None
                or scenario_data._cross_lane_deadband_counter.shape
                != cross_lane_info.shape
                or scenario_data._cross_lane_deadband_counter.device
                != cross_lane_info.device
            ):
                scenario_data._cross_lane_deadband_counter = torch.zeros(
                    cross_lane_info.shape,
                    dtype=torch.int32,
                    device=cross_lane_info.device,
                )

            counter = scenario_data._cross_lane_deadband_counter
            counter = torch.clamp(counter - 1, min=0)
            counter = torch.where(
                cross_lane_info,
                torch.full_like(counter, deadband_frames),
                counter,
            )
            counter = torch.where(controlled_mask, counter, torch.zeros_like(counter))
            scenario_data._cross_lane_deadband_counter = counter
            cross_lane_info = counter > 0
        else:
            scenario_data._cross_lane_deadband_counter = None

        # Priority: off-road always overrides cross-lane.
        cross_lane_info = cross_lane_info & (~off_road_info)

        speed_range = constants.MAX_SPEED - constants.MIN_SPEED
        normalized_speed = (
            (speed - constants.MIN_SPEED) / (speed_range + constants.EPS)
        ).clamp(0.0, 1.0)
        speed_scale = 1 + 5.0 * torch.tanh(2.0 * normalized_speed)

        off_road_reward = off_road_info.float() * off_road_weight * speed_scale
        cross_lane_reward = torch.where(cross_lane_info, cross_lane_weight, 1.0)

        combined_info = torch.zeros_like(off_road_info, dtype=torch.int32)
        combined_info[cross_lane_info] = 2
        combined_info[off_road_info] = 1

        if getattr(self.config, "enable_occupancy_grid", True):
            occ_hit = self._occ_polygon_hit(
                polygons_now,
                polygons_last,
                valid,
                controlled_mask,
                scenario_data.occupancy_grid,
                speed,
            )
        else:
            occ_hit = torch.zeros_like(valid, dtype=torch.bool)
        occ_corner_reward = occ_hit.float() * off_road_weight * speed_scale
        off_road_reward = torch.minimum(off_road_reward, occ_corner_reward)

        rewards_and_infos["OffRoad"] = {
            "reward": off_road_reward,
            "info": combined_info,
            "occ_hit": occ_hit.int(),
        }
        rewards_and_infos["CrossLane"] = {
            "reward": cross_lane_reward,
            # This buffer covers the adjacent-lane geometry used by the detector.
            "info": cross_lane_info.int(),
            "lane_change_info": lane_change_info.int(),
            "solid_lane_change_info": solid_lane_change_info.int(),
        }

    def _occ_polygon_hit(
        self, polygons_now, polygons_last, valid, controlled, occ_grid, speed
    ):
        """Check if sampled polygon motion points hit occupied occ-grid cells.

        Uses points on the polygon: 4 corners + 3 non-front edge midpoints +
        3 samples on the front edge, then samples motion segments between frames.

        Only evaluates the ego agent (index 0).
        """
        occ_hit = torch.zeros_like(valid, dtype=torch.bool)
        if occ_grid.numel() == 0:
            return occ_hit

        # Only compute for ego agent (index 0)
        ego_mask = torch.zeros_like(controlled)
        ego_mask[:, 0] = True
        ego_controlled = ego_mask & controlled & valid
        batch_idx, agent_idx = torch.where(ego_controlled)
        if batch_idx.numel() == 0:
            return occ_hit

        # OCC config + grid shape
        occ_res = self.config.occ_grid_resolution
        occ_xmax = self.config.occ_grid_xmax
        occ_ymax = self.config.occ_grid_ymax
        H, W = occ_grid.shape[1], occ_grid.shape[2]

        # --- 1) Build motion points (P points per agent)
        corners_now = polygons_now[batch_idx, agent_idx]  # [C, 4, 2]
        corners_last = polygons_last[batch_idx, agent_idx]  # [C, 4, 2]
        occ_buffer = torch.zeros_like(speed[batch_idx, agent_idx])
        occ_buffer = occ_buffer.to(dtype=corners_now.dtype)
        corners_now = expand_polygon_with_buffer(corners_now, occ_buffer, occ_buffer)
        corners_last = expand_polygon_with_buffer(corners_last, occ_buffer, occ_buffer)
        edge_mid_now = 0.5 * (corners_now + corners_now.roll(shifts=-1, dims=1))
        edge_mid_last = 0.5 * (corners_last + corners_last.roll(shifts=-1, dims=1))
        non_front_mid_now = edge_mid_now[:, 1:, :]  # skip front edge mid
        non_front_mid_last = edge_mid_last[:, 1:, :]

        front_t = torch.tensor(
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            device=polygons_now.device,
            dtype=corners_now.dtype,
        )
        front_start_now = corners_now[:, 0:1, :]
        front_end_now = corners_now[:, 1:2, :]
        front_start_last = corners_last[:, 0:1, :]
        front_end_last = corners_last[:, 1:2, :]
        front_extra_now = front_start_now + (
            front_end_now - front_start_now
        ) * front_t.view(1, -1, 1)
        front_extra_last = front_start_last + (
            front_end_last - front_start_last
        ) * front_t.view(1, -1, 1)

        points_now = torch.cat(
            [corners_now, non_front_mid_now, front_extra_now], dim=1
        )  # [C, P, 2]
        points_last = torch.cat(
            [corners_last, non_front_mid_last, front_extra_last], dim=1
        )  # [C, P, 2]

        # --- 2) Sample motion segments
        seg_vec = points_now - points_last
        seg_len = torch.linalg.norm(seg_vec, dim=-1)  # [C, P]
        max_samples = 6
        num_steps = torch.clamp(
            torch.ceil(seg_len / occ_res).long() + 1, min=1, max=max_samples
        )
        t = torch.linspace(0, 1, max_samples, device=polygons_now.device)
        t_max = (num_steps - 1) / max(1, (max_samples - 1))
        sample_mask = t.view(1, 1, -1) <= t_max.unsqueeze(-1)  # [C, P, S]
        points = points_last.unsqueeze(2) + t.view(1, 1, -1, 1) * seg_vec.unsqueeze(
            2
        )  # [C, P, S, 2]

        # --- 3) Map to OCC grid and gather hits
        x = points[..., 0]
        y = points[..., 1]
        row = torch.floor((occ_xmax - x) / occ_res).long()
        col = torch.floor((occ_ymax - y) / occ_res).long()
        in_bounds = (row >= 0) & (row < H) & (col >= 0) & (col < W)
        flat_idx = (row * W + col).clamp(min=0, max=H * W - 1)

        occ_flat = occ_grid[batch_idx].view(batch_idx.shape[0], -1)
        occ_vals = occ_flat.gather(1, flat_idx.view(batch_idx.shape[0], -1)).view(
            batch_idx.shape[0], points_now.shape[1], max_samples
        )
        occ_hit[batch_idx, agent_idx] = ((occ_vals > 0) & in_bounds & sample_mask).any(
            dim=(1, 2)
        )
        return occ_hit

    def detect_cross_line_knn(
        self,
        polygons_now,
        polygons_last,
        controlled,
        boundary_lines,
        boundary_mask,
        collision_width_buffer,
        collision_length_buffer,
        k: int = 12,
    ):
        """
        Args:
            polygons_last (torch.Tensor): Vehicle polygons at the previous timestep, with shape [N, A, 4, 2].
            polygons_now (torch.Tensor): Vehicle polygons at the current timestep, with shape [N, A, 4, 2].
            controlled (torch.Tensor): A boolean tensor indicating controlled vehicles, with shape [N, A].
            boundary_lines (torch.Tensor): shape [N, C, M, 2].
            boundary_mask (torch.Tensor): shape [N, C], values {0, 1, 2, 3}.
            k (int): Number of nearest boundaries used for intersection checks.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - off_road_info: [N, A] curb crossing flags.
                - cross_lane_info: [N, A] solid-line crossing flags.
                - lane_change_info: [N, A] lane-change flags on solid/non-solid lanes.
        """
        # Only curb/off-road detection uses expanded polygons.
        # Crossing solid lane lines should use the original footprint (no buffer).
        width_buffer = collision_width_buffer.reshape(-1)
        length_buffer = collision_length_buffer.reshape(-1)
        buffered_polygons_now = expand_polygon_with_buffer(
            polygons_now.view(-1, 4, 2),
            width_buffer,
            length_buffer,
        ).view_as(polygons_now)
        buffered_polygons_last = expand_polygon_with_buffer(
            polygons_last.view(-1, 4, 2),
            width_buffer,
            length_buffer,
        ).view_as(polygons_last)

        vehicle_segments_for_off_road = self._build_vehicle_segments(
            buffered_polygons_now, buffered_polygons_last
        )
        vehicle_segments_for_cross_lane = self._build_vehicle_segments(
            polygons_now, polygons_last
        )

        N, A, P, _, _ = vehicle_segments_for_off_road.shape  # P=8
        _, C, M, _ = boundary_lines.shape
        assert M == 2, f"OffRoad.detect_cross_line_knn expects M == 2, got {M}"

        # Lane boundaries are always represented by exactly one segment.
        road_segments = torch.stack(
            [boundary_lines[:, :, 0, :], boundary_lines[:, :, 1, :]], dim=-2
        ).unsqueeze(2)  # [N, C, 1, 2, 2]
        road_seg_start = boundary_lines[:, :, 0, :]  # [N, C, 2]
        road_seg_end = boundary_lines[:, :, 1, :]  # [N, C, 2]
        num_boundary_segments = 1

        # Use point-to-segment distance for KNN candidate selection; point-to-endpoint
        # cdist is too inaccurate for long lane segments.
        agent_centers = polygons_now.mean(dim=2)  # [N, A, 2]
        _, _, dist_to_segments = project_points_to_segments(
            agent_centers.unsqueeze(2),
            road_seg_start.unsqueeze(1),
            road_seg_end.unsqueeze(1),
            return_dist=True,
        )  # [N, A, C]
        dist_matrix = dist_to_segments

        # Set distance to infinity for invalid/padding boundaries.
        # Keep 1(curb), 2(solid), 3(non-solid) for different downstream checks.
        excluded_mask = boundary_mask == 0
        dist_matrix[excluded_mask.unsqueeze(1).expand(-1, A, -1)] = float("inf")

        # Find indices of controlled vehicles
        batch_indices, agent_indices = torch.where(controlled)
        num_controlled = batch_indices.shape[0]

        # If no controlled vehicles, return all False
        if num_controlled == 0:
            zeros = torch.zeros_like(controlled)
            return zeros, zeros, zeros

        dist_subset = dist_matrix[
            batch_indices, agent_indices, :
        ]  # [num_controlled, C]
        num_used = min(k, C)
        _, used_indices = torch.topk(
            dist_subset, k=num_used, dim=-1, largest=False
        )  # [num_controlled, num_used]

        # Extract segments of controlled vehicles
        controlled_segs_off_road = vehicle_segments_for_off_road[
            batch_indices, agent_indices
        ]  # [num_controlled, P, 2, 2]
        controlled_segs_cross_lane = vehicle_segments_for_cross_lane[
            batch_indices, agent_indices
        ]  # [num_controlled, P, 2, 2]

        # Get road segments and masks for the batches of controlled vehicles
        batch_expanded = torch.repeat_interleave(batch_indices, repeats=num_used)
        batch_road_segs = road_segments[batch_expanded, used_indices.view(-1)]
        batch_road_segs = batch_road_segs.view(
            num_controlled, num_used, num_boundary_segments, 2, 2
        )
        batch_road_masks = boundary_mask[batch_expanded, used_indices.view(-1)]
        batch_road_masks = batch_road_masks.view(num_controlled, num_used)

        # Expand dimensions for parallel computation
        veh_segs_off_road_expanded = controlled_segs_off_road.unsqueeze(2).unsqueeze(
            2
        )  # [num_controlled, P, 1, 1, 2, 2]
        veh_segs_cross_lane_expanded = controlled_segs_cross_lane.unsqueeze(
            2
        ).unsqueeze(2)  # [num_controlled, P, 1, 1, 2, 2]
        road_segs_expanded = batch_road_segs.unsqueeze(
            1
        )  # [num_controlled, 1, num_used, M-1, 2, 2]

        # Batch check all segment pairs for intersection.
        intersections_off_road = batch_segment_intersection(
            veh_segs_off_road_expanded.expand(
                num_controlled, P, num_used, num_boundary_segments, 2, 2
            ),
            road_segs_expanded.expand(
                num_controlled, P, num_used, num_boundary_segments, 2, 2
            ),
        )  # [num_controlled, P, num_used, 1]
        intersections_cross_lane = batch_segment_intersection(
            veh_segs_cross_lane_expanded.expand(
                num_controlled, P, num_used, num_boundary_segments, 2, 2
            ),
            road_segs_expanded.expand(
                num_controlled, P, num_used, num_boundary_segments, 2, 2
            ),
        )  # [num_controlled, P, num_used, 1]

        # Apply road boundary mask to filter out invalid boundaries
        mask_expanded = batch_road_masks.unsqueeze(1).unsqueeze(
            3
        )  # [num_controlled, 1, num_used, 1]

        off_road = intersections_off_road & (
            mask_expanded == 1
        )  # [num_controlled, P, num_used, 1]
        cross_lane = intersections_cross_lane & (
            mask_expanded == 2
        )  # [num_controlled, P, num_used, 1]
        lane_change_cross = intersections_cross_lane & (
            (mask_expanded == 2) | (mask_expanded == 3)
        )  # [num_controlled, P, num_used, 1]

        off_road_result = off_road.any(dim=1).any(dim=1).any(dim=1)  # [num_controlled]
        cross_lane_result = (
            cross_lane.any(dim=1).any(dim=1).any(dim=1)
        )  # [num_controlled]

        # Lane-change: while crossing lane boundary (solid/non-solid), the
        # leading front-edge 15% sample switches side relative to the crossed
        # segment. Left lane changes use the left-front 15% sample; right lane
        # changes use the right-front 15% sample.
        lane_change_hit = lane_change_cross.any(dim=1)  # [num_controlled, num_used, 1]
        road_seg_start = batch_road_segs[..., 0, :]  # [num_controlled, num_used, 1, 2]
        road_seg_end = batch_road_segs[..., 1, :]  # [num_controlled, num_used, 1, 2]
        road_seg_vec = road_seg_end - road_seg_start

        controlled_polygons_now = polygons_now[batch_indices, agent_indices]
        controlled_polygons_last = polygons_last[batch_indices, agent_indices]
        front_samples_now, front_samples_last = self._front_edge_lane_change_samples(
            controlled_polygons_now,
            controlled_polygons_last,
        )
        rel_now = front_samples_now[:, :, None, None, :] - road_seg_start.unsqueeze(1)
        rel_last = front_samples_last[:, :, None, None, :] - road_seg_start.unsqueeze(1)
        road_seg_vec = road_seg_vec.unsqueeze(1)
        side_now = (
            road_seg_vec[..., 0] * rel_now[..., 1]
            - road_seg_vec[..., 1] * rel_now[..., 0]
        )
        side_last = (
            road_seg_vec[..., 0] * rel_last[..., 1]
            - road_seg_vec[..., 1] * rel_last[..., 0]
        )
        side_eps = 1e-5
        opposite_side = ((side_now > side_eps) & (side_last < -side_eps)) | (
            (side_now < -side_eps) & (side_last > side_eps)
        )
        opposite_sample_last = side_last.flip(dims=(1,))
        opposite_sample_already_target = (
            (opposite_sample_last > side_eps) & (side_now > side_eps)
        ) | ((opposite_sample_last < -side_eps) & (side_now < -side_eps))
        lane_change_result = (
            (
                lane_change_hit.unsqueeze(1)
                & opposite_side
                & ~opposite_sample_already_target
            )
            .any(dim=1)
            .any(dim=1)
            .any(dim=1)
        )

        off_road_info = torch.zeros_like(controlled)
        cross_lane_info = torch.zeros_like(controlled)
        lane_change_info = torch.zeros_like(controlled)

        off_road_info[batch_indices, agent_indices] = off_road_result
        cross_lane_info[batch_indices, agent_indices] = cross_lane_result
        lane_change_info[batch_indices, agent_indices] = lane_change_result
        return off_road_info, cross_lane_info, lane_change_info

    @staticmethod
    def _front_edge_lane_change_samples(
        polygons_now: torch.Tensor,
        polygons_last: torch.Tensor,
        sample_fraction: float = 0.15,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return left-front and right-front 15% samples."""
        left_now = polygons_now[:, 0, :]
        right_now = polygons_now[:, 1, :]
        left_last = polygons_last[:, 0, :]
        right_last = polygons_last[:, 1, :]

        front_vec_now = right_now - left_now
        front_vec_last = right_last - left_last
        left_sample_now = left_now + sample_fraction * front_vec_now
        right_sample_now = left_now + (1.0 - sample_fraction) * front_vec_now
        left_sample_last = left_last + sample_fraction * front_vec_last
        right_sample_last = left_last + (1.0 - sample_fraction) * front_vec_last
        return (
            torch.stack([left_sample_now, right_sample_now], dim=1),
            torch.stack([left_sample_last, right_sample_last], dim=1),
        )

    def detect_off_road(
        self, polygons_now, polygons_last, controlled, road_bound, road_bound_mask
    ) -> torch.Tensor:
        motion_segments = torch.cat(
            [polygons_now.unsqueeze(-2), polygons_last.unsqueeze(-2)], dim=-2
        )  # [N, A, 4, 2, 2]
        front_segments = torch.cat(
            [
                polygons_now[:, :, 0:1].unsqueeze(-2),
                polygons_now[:, :, 1:2].unsqueeze(-2),
            ],
            dim=-2,
        )  # [N, A, 1, 2, 2]
        right_segments = torch.cat(
            [
                polygons_now[:, :, 1:2].unsqueeze(-2),
                polygons_now[:, :, 2:3].unsqueeze(-2),
            ],
            dim=-2,
        )  # [N, A, 1, 2, 2]
        rear_segments = torch.cat(
            [
                polygons_now[:, :, 2:3].unsqueeze(-2),
                polygons_now[:, :, 3:4].unsqueeze(-2),
            ],
            dim=-2,
        )  # [N, A, 1, 2, 2]
        left_segments = torch.cat(
            [
                polygons_now[:, :, 3:4].unsqueeze(-2),
                polygons_now[:, :, 0:1].unsqueeze(-2),
            ],
            dim=-2,
        )  # [N, A, 1, 2, 2]
        vehicle_segments = torch.cat(
            [
                motion_segments,
                front_segments,
                right_segments,
                rear_segments,
                left_segments,
            ],
            dim=2,
        )  # [N, A, 8, 2, 2]
        N, A, P, _, _ = vehicle_segments.shape  # P=6
        _, C, M, _ = road_bound.shape

        # Find indices of controlled vehicles
        batch_indices, agent_indices = torch.where(controlled)
        num_controlled = batch_indices.shape[0]

        # If no controlled vehicles, return all False
        if num_controlled == 0:
            return torch.zeros_like(controlled)

        # Extract segments of controlled vehicles
        controlled_segs = vehicle_segments[
            batch_indices, agent_indices
        ]  # [num_controlled, P, 2, 2]

        # Convert road boundaries to line segments
        road_segments = torch.stack(
            [road_bound[..., :-1, :], road_bound[..., 1:, :]], dim=-2
        )  # [N, C, M-1, 2, 2]  (M=5)

        # Get road segments and masks for the batches of controlled vehicles
        batch_road_segs = road_segments[batch_indices]  # [num_controlled, C, M-1, 2, 2]
        batch_road_masks = road_bound_mask[batch_indices]  # [num_controlled, C]

        # Expand dimensions for parallel computation
        veh_segs_expanded = controlled_segs.unsqueeze(2).unsqueeze(
            2
        )  # [num_controlled, P, 1, 1, 2, 2]
        road_segs_expanded = batch_road_segs.unsqueeze(
            1
        )  # [num_controlled, 1, C, M-1, 2, 2]

        # Batch check all segment pairs for intersection
        intersections = batch_segment_intersection(
            veh_segs_expanded.expand(-1, -1, C, M - 1, -1, -1),
            road_segs_expanded.expand(-1, P, -1, -1, -1, -1),
        )  # [num_controlled, P, C, M-1]

        # Apply road boundary mask to filter out invalid boundaries
        mask_expanded = batch_road_masks.unsqueeze(1).unsqueeze(
            3
        )  # [num_controlled, 1, C, 1]
        intersections = intersections & mask_expanded  # [num_controlled, P, C, M-1]

        # Check if any segment of the vehicle intersects with any valid road boundary
        vehicle_intersects = (
            intersections.any(dim=-1).any(dim=-1).any(dim=-1)
        )  # [num_controlled]

        result = torch.zeros_like(controlled)
        result[batch_indices, agent_indices] = vehicle_intersects

        return result
