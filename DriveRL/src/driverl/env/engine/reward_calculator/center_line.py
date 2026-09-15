from math import inf

import torch

from driverl.datatypes.data_enums import LaneType
from driverl.datatypes.scenario_data import ScenarioData
from driverl.env.engine.reward_calculator.base_reward_calculator import (
    REWARD_CALCULATOR_REGISTER,
    BaseRewardCalculator,
)
from driverl.utils.geometry import (
    expand_polygon_with_buffer,
    project_points_to_segments,
    segment_polygons_intersection2,
)


@REWARD_CALCULATOR_REGISTER.register_module
class CenterLine(BaseRewardCalculator):
    def __init__(self, config=None):
        """Initialize the off road reward calculator."""
        super().__init__(config)
        self._direction_compliance_threshold = getattr(
            config, "driving_direction_compliance_threshold", 2.0
        )
        self._direction_violation_threshold = getattr(
            config, "driving_direction_violation_threshold", 6.0
        )
        self._direction_time_horizon = getattr(
            config, "driving_direction_time_horizon", 1.0
        )
        self._frame_time_interval = getattr(config, "frame_time_interval", 0.2)

    def forward(
        self,
        scenario_data: "ScenarioData",
        log_scenario_data: "ScenarioData",
        rewards_and_infos: dict,
        **kwargs,
    ):
        """
        Computes the reward and information related to center line.

        Args:
            scenario_data (ScenarioData): The current state of the environment.
            log_scenario_data (ScenarioData): The logged scenario data.
            rewards_and_infos (dict): A dictionary to store the rewards and infos.
            **kwargs: Additional arguments.
        """

        # 1) Prep lane geometry and agent state
        lane_centers = scenario_data.lanes_centers_points
        N, C, M, _ = lane_centers.shape

        position = scenario_data.agent_positions_all[:, :, -1, :]  # [N, A, 2]
        orientation = scenario_data.agent_orientation_all[:, :, -1]  # [N, A]
        N, A, _ = position.shape
        velocities = scenario_data.agent_velocity_all[:, :, -1, :]  # [N, A, 2]
        speed = torch.linalg.norm(velocities, dim=-1)  # [N, A]

        randomized_features = scenario_data.randomized_features
        static_speed_weight = randomized_features.get(
            "static_speed_weight", calculate=True
        )
        max_overspeed_value_threshold = randomized_features.get(
            "over_speed_threshold", calculate=True
        )
        over_speed_scale = randomized_features.get("over_speed_scale", calculate=True)
        wrong_way_weight = randomized_features.get("wrong_way_weight", calculate=True)
        deviation_distance_weight = randomized_features.get(
            "deviation_distance_weight", calculate=True
        )
        # deviation_angle_weight = randomized_features.get("deviation_angle_weight", calculate=True)  # unused
        # deviation_angle_limit = randomized_features.get("deviation_angle_limit", calculate=True)    # unused
        max_deviation_distance_for_penalty = randomized_features.get(
            "max_deviation_distance_for_penalty", calculate=True
        )
        curb_clearance_distance = randomized_features.get(
            "curb_clearance_distance", calculate=True
        )
        curb_clearance_weight = randomized_features.get(
            "curb_clearance_weight", calculate=True
        )

        center_line = torch.cat(
            [
                lane_centers[..., :-1, 0:2].unsqueeze(-2),
                lane_centers[..., 1:, 0:2].unsqueeze(-2),
            ],
            dim=-2,
        )  # [N, C, M-1, 2, 2]
        center_line = center_line.reshape(N, C * (M - 1), 2, 2)  # [N, C * (M-1), 2, 2]
        center_line_mask = (
            scenario_data.lanes_centers_mask.unsqueeze(-1)
            .expand(N, C, M - 1)
            .reshape(N, C * (M - 1))
        )  # [N, C * (M-1)]

        min_dist, min_idx, _ = self.calculate_nearest_line(
            position, orientation, center_line, center_line_mask
        )

        direction_lines = getattr(scenario_data, "lanes_centers_lines", None)
        direction_lines_mask = getattr(scenario_data, "lanes_centers_lines_mask", None)
        if (
            not isinstance(direction_lines, torch.Tensor)
            or direction_lines.numel() == 0
            or not isinstance(direction_lines_mask, torch.Tensor)
            or direction_lines_mask.numel() == 0
        ):
            direction_lines = lane_centers[..., :2]
            direction_lines_mask = scenario_data.lanes_centers_mask
        direction_line_idx, progress_along_line = self.calculate_baseline_progress(
            position, direction_lines, direction_lines_mask
        )

        polygons_now, _, valid = scenario_data.get_recent_agent_polygons()

        lane_points = scenario_data.lanes_points
        lane_points_mask = scenario_data.lanes_points_mask
        lane_positions = lane_points[..., :2]
        lane_types_first = lane_points[..., 0, 2]
        road_bound_mask = lane_points_mask & (lane_types_first == LaneType.CURB.code)
        non_curb_mask = lane_points_mask & (lane_types_first != LaneType.CURB.code)

        curb_too_close, curb_topk = self._calculate_curb_too_close(
            polygons_now,
            valid,
            scenario_data.agent_control_manager.controlled_mask,
            lane_positions,
            road_bound_mask,
            curb_clearance_distance,
            return_topk=True,
        )

        has_near_non_curb_lane, _boundary_speed_limit, nearest_non_curb_lane_dist = (
            self._calculate_lane_speed_limit_and_proximity(
                position,
                lane_positions,
                lane_points_mask,
                non_curb_mask,
                lane_points[..., 3],
                max_deviation_distance_for_penalty,
            )
        )
        speed_limit = self._gather_nearest_centerline_speed_limit(
            lane_centers=lane_centers,
            min_idx=min_idx,
            min_dist=min_dist,
        )

        # 2) Overspeed reward
        overspeed_info = speed > speed_limit
        overspeed_reward = self.calculate_speed_reward(
            speed,
            speed_limit,
            static_speed_weight,
            max_overspeed_value_threshold,
            over_speed_scale,
        )

        # 3) Preconditions for wrong-way handling
        assert "OffRoad" in rewards_and_infos, (
            "CenterLine expects OffRoad to run before it; "
            "ensure reward calculator order includes OffRoad first."
        )
        base_offroad_info = rewards_and_infos["OffRoad"]["info"]

        # 4) Distance and direction penalties
        in_penalty_band = min_dist <= max_deviation_distance_for_penalty
        direction_mask, severe_direction_mask = (
            self.compute_driving_direction_violation(
                scenario_data=scenario_data,
                baseline_idx=direction_line_idx,
                progress_along_baseline=progress_along_line,
                in_penalty_band=in_penalty_band,
                valid_centerline=torch.isfinite(min_dist),
            )
        )
        if self.config.enable_narrow_road_right_preference_reward:
            narrow_road_wrong_way = self._narrow_road_right_preferance(
                position, orientation, curb_topk, has_near_non_curb_lane
            )
            # Apply narrow-road right-preference only on low-speed-limit roads (<= 60 kph).
            speed_limit_eps = 1e-4
            map_speed_limit_ok = speed_limit <= (60.0 / 3.6 + speed_limit_eps)
            narrow_road_wrong_way = narrow_road_wrong_way & map_speed_limit_ok
        else:
            narrow_road_wrong_way = torch.zeros_like(min_dist, dtype=torch.bool)

        # Wrong-way soft reward (separate from OffRoad hard reward)
        wrong_way_active = (
            direction_mask & in_penalty_band & (base_offroad_info != 1)
        ) | (narrow_road_wrong_way & (base_offroad_info != 1))
        wrong_way_reward = torch.where(
            wrong_way_active,
            wrong_way_weight,
            torch.ones_like(speed),
        )

        # Update OffRoad info for logging unless already offroad=1.
        rewards_and_infos["OffRoad"]["info"] = torch.where(
            severe_direction_mask & (base_offroad_info != 1),
            torch.full_like(base_offroad_info, 4),
            torch.where(
                wrong_way_active & (base_offroad_info != 1),
                torch.full_like(base_offroad_info, 3),
                base_offroad_info,
            ),
        )
        rewards_and_infos["OffRoad"]["reward"] = torch.where(
            severe_direction_mask & (base_offroad_info != 1),
            torch.full_like(rewards_and_infos["OffRoad"]["reward"], -1.0),
            rewards_and_infos["OffRoad"]["reward"],
        )
        # CrossLane carries cross-lane (1), wrong-way (3), or severe wrong-way (4).
        if "CrossLane" in rewards_and_infos:
            cross = rewards_and_infos["CrossLane"]
            cross_info = cross["info"].to(dtype=torch.int32)
            cross_reward = cross["reward"]

            cross["info"] = torch.where(
                severe_direction_mask,
                torch.full_like(cross_info, 4),
                torch.where(
                    wrong_way_active, torch.full_like(cross_info, 3), cross_info
                ),
            )
            cross["reward"] = torch.where(
                wrong_way_active, wrong_way_reward, cross_reward
            )
            cross_lane_active = cross["info"] != 0
        else:
            cross_lane_active = torch.zeros_like(wrong_way_active)

        # Continuous centerline reward:
        # centerline distance is normalized by the current available lateral
        # room before the vehicle would press the nearest non-curb lane line.
        vehicle_half_width = scenario_data.agent_size_all[:, :, -1, 1] * 0.5
        centerline_distance_reward = self.compute_centerline_distance_reward(
            min_dist=min_dist,
            nearest_lane_boundary_dist=nearest_non_curb_lane_dist,
            vehicle_half_width=vehicle_half_width,
            deviation_distance_weight=deviation_distance_weight,
        )
        centerline_distance_reward = torch.where(
            has_near_non_curb_lane,
            centerline_distance_reward,
            torch.ones_like(centerline_distance_reward),
        )

        # Curb clearance/cross-lane/wrong-way suppress centerline penalty.
        centerline_reward = torch.where(
            curb_too_close,
            torch.ones_like(speed),
            torch.where(
                cross_lane_active | wrong_way_active,
                torch.ones_like(speed),
                centerline_distance_reward,
            ),
        )

        # 5) Write rewards in order: Overspeed -> CrossLane (incl. wrong-way) -> CenterLine
        rewards_and_infos["Overspeed"] = {
            "reward": overspeed_reward,
            "info": overspeed_info,
        }
        curb_speed_factor = self.compute_curb_speed_factor(speed)
        rewards_and_infos["CurbClearance"] = {
            "reward": torch.where(
                curb_too_close,
                curb_clearance_weight * curb_speed_factor,
                torch.ones_like(speed),
            ),
            "info": curb_too_close,
        }
        rewards_and_infos["CenterLine"] = {
            "reward": centerline_reward,
            "info": min_dist,
        }

    @staticmethod
    def compute_curb_speed_factor(speed: torch.Tensor) -> torch.Tensor:
        max_speed = 40.0 / 3.6
        exponent = 0.5
        speed_norm = (speed / max_speed).clamp_min(0.0)
        return (1.0 - 0.7 * speed_norm.pow(exponent)).clamp(0.3, 1.0)

    @staticmethod
    def compute_centerline_distance_reward(
        min_dist: torch.Tensor,
        nearest_lane_boundary_dist: torch.Tensor,
        vehicle_half_width: torch.Tensor,
        deviation_distance_weight: torch.Tensor,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        """Map centerline distance to reward using dynamic lane clearance."""
        safe_offset = (
            min_dist + nearest_lane_boundary_dist - vehicle_half_width
        ).clamp_min(eps)
        pressing_boundary = nearest_lane_boundary_dist <= vehicle_half_width
        raw_t = min_dist / safe_offset
        t = torch.where(
            pressing_boundary,
            torch.ones_like(min_dist),
            torch.where(torch.isfinite(raw_t), raw_t, torch.ones_like(min_dist)),
        ).clamp(0.0, 1.0)
        return 1.0 - (1.0 - deviation_distance_weight) * t

    def _calculate_lane_speed_limit_and_proximity(
        self,
        position: torch.Tensor,  # [N, A, 2]
        lane_positions: torch.Tensor,  # [N, C, M, 2]
        lane_points_mask: torch.Tensor,  # [N, C]
        non_curb_mask: torch.Tensor,  # [N, C]
        lane_speed: torch.Tensor,  # [N, C, M]
        max_deviation_distance_for_penalty: torch.Tensor,  # [N, A]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute non-curb-near flag and nearest-lane speed limit.

        Assumes each lane has exactly 2 points (one segment).
        """
        N, A, _ = position.shape
        _, C, M, _ = lane_positions.shape
        if M != 2:
            raise ValueError(
                f"Expected lane_positions shape [N, C, 2, 2], but got M={M}."
            )

        pos_exp = position.unsqueeze(2)  # [N, A, 1, 2]
        seg_start = lane_positions[:, None, :, 0, :]  # [N, 1, C, 2]
        seg_end = lane_positions[:, None, :, 1, :]  # [N, 1, C, 2]
        lane_dist_sq = self._squared_distance_points_to_segments(
            pos_exp, seg_start, seg_end
        )  # [N, A, C]
        lane_dist_sq = lane_dist_sq.masked_fill(
            ~lane_points_mask.unsqueeze(1), float("inf")
        )

        max_dev_sq = max_deviation_distance_for_penalty.unsqueeze(-1).square()
        near_non_curb = non_curb_mask.unsqueeze(1) & (lane_dist_sq < max_dev_sq)
        has_near_non_curb_lane = near_non_curb.any(dim=2)
        non_curb_dist_sq = lane_dist_sq.masked_fill(
            ~non_curb_mask.unsqueeze(1), float("inf")
        )
        nearest_non_curb_lane_dist_sq, nearest_non_curb_idx = non_curb_dist_sq.min(
            dim=2
        )
        nearest_non_curb_lane_dist = torch.sqrt(nearest_non_curb_lane_dist_sq)
        speed_candidates = lane_speed[..., 0].unsqueeze(1).expand(N, A, C)
        nearest_speed_limit = speed_candidates.gather(
            2, nearest_non_curb_idx.unsqueeze(-1)
        ).squeeze(-1)
        speed_limit = torch.where(
            torch.isfinite(nearest_non_curb_lane_dist) & (nearest_speed_limit > 0.0),
            nearest_speed_limit,
            torch.zeros_like(nearest_speed_limit),
        )
        return has_near_non_curb_lane, speed_limit, nearest_non_curb_lane_dist

    @staticmethod
    def _gather_nearest_centerline_speed_limit(
        lane_centers: torch.Tensor,
        min_idx: torch.Tensor,
        min_dist: torch.Tensor,
    ) -> torch.Tensor:
        """Return the speed limit of the nearest lane-center segment."""
        N, C, M, _ = lane_centers.shape
        if M < 2:
            return torch.zeros_like(min_dist)

        segment_speed_limits = lane_centers[..., :-1, 3].reshape(N, C * (M - 1))
        speed_limit = (
            segment_speed_limits.unsqueeze(1)
            .expand(N, min_idx.shape[1], C * (M - 1))
            .gather(2, min_idx.unsqueeze(-1))
            .squeeze(-1)
        )
        return torch.where(
            torch.isfinite(min_dist) & (speed_limit > 0.0),
            speed_limit,
            torch.zeros_like(speed_limit),
        )

    @staticmethod
    def _squared_distance_points_to_segments(
        points: torch.Tensor,
        seg_starts: torch.Tensor,
        seg_ends: torch.Tensor,
        eps: float = 1e-7,
    ) -> torch.Tensor:
        """Compute squared distance between points and line segments."""
        seg_vec = seg_ends - seg_starts
        w = points - seg_starts
        v_sq = (seg_vec * seg_vec).sum(dim=-1)
        t = (w * seg_vec).sum(dim=-1) / (v_sq + eps)
        t = t.clamp_(0.0, 1.0)
        proj = seg_starts + t.unsqueeze(-1) * seg_vec
        diff = proj - points
        return (diff * diff).sum(dim=-1)

    def _calculate_curb_too_close(
        self,
        polygons_now,
        valid,
        controlled,
        road_bound,
        road_bound_mask,
        curb_clearance_distance,
        k: int = 12,
        return_topk: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict | None]:
        N, A, _, _ = polygons_now.shape
        device = polygons_now.device

        curb_too_close_all = torch.zeros(N, A, dtype=torch.bool, device=device)

        batch_indices, agent_indices = torch.where(controlled)
        num_controlled = batch_indices.shape[0]
        if num_controlled == 0:
            return (curb_too_close_all, None) if return_topk else curb_too_close_all

        poly = polygons_now[batch_indices, agent_indices]  # [num_controlled, 4, 2]
        valid_vec = valid[batch_indices, agent_indices]  # [num_controlled]
        width_buf = curb_clearance_distance[
            batch_indices, agent_indices
        ]  # [num_controlled]
        length_buf = torch.zeros_like(width_buf)
        poly = expand_polygon_with_buffer(
            poly, width_buf, length_buf
        )  # [num_controlled, 4, 2]

        road_centers = road_bound.mean(dim=2)
        batch_road_centers = road_centers[batch_indices]  # [num_controlled, C, 2]

        # Batched cdist avoids repeating road centers across vertices.
        vertex_to_road_dist = torch.cdist(
            poly, batch_road_centers
        )  # [num_controlled, 4, C]
        dist_matrix = vertex_to_road_dist.amin(dim=1)

        invalid_mask = ~(road_bound_mask[batch_indices].bool())
        dist_matrix[invalid_mask] = float("inf")

        num_used = min(k, dist_matrix.shape[1])
        if num_used == 0:
            return (curb_too_close_all, None) if return_topk else curb_too_close_all

        _, used_indices = torch.topk(dist_matrix, k=num_used, dim=-1, largest=False)

        road_segments = torch.stack(
            [road_bound[..., :-1, :], road_bound[..., 1:, :]], dim=-2
        )

        batch_expanded = torch.repeat_interleave(batch_indices, repeats=num_used)
        used_flat = used_indices.reshape(-1)
        batch_road_segs = road_segments[batch_expanded, used_flat].view(
            num_controlled, num_used, -1, 2, 2
        )
        batch_road_masks = road_bound_mask[batch_expanded, used_flat].view(
            num_controlled, num_used
        )

        if not batch_road_masks.any():
            return (curb_too_close_all, None) if return_topk else curb_too_close_all

        valid_segment_mask = batch_road_masks.unsqueeze(-1).expand(
            -1, -1, batch_road_segs.shape[2]
        )

        num_segments_per_boundary = batch_road_segs.shape[2]
        total_segments = num_controlled * num_used * num_segments_per_boundary

        all_segments = batch_road_segs.reshape(total_segments, 2, 2)
        segment_to_vehicle = torch.arange(
            num_controlled, device=device
        ).repeat_interleave(num_used * num_segments_per_boundary)

        valid_mask_flat = valid_segment_mask.reshape(-1)
        valid_indices = torch.nonzero(valid_mask_flat, as_tuple=False).squeeze(-1)

        if valid_indices.numel() == 0:
            return (curb_too_close_all, None) if return_topk else curb_too_close_all

        segments_valid = all_segments[valid_indices]
        vehicle_for_segment = segment_to_vehicle[valid_indices]

        poly_for_segments = poly[vehicle_for_segment]
        hit = segment_polygons_intersection2(segments_valid, poly_for_segments)
        collision_counts = torch.zeros(num_controlled, dtype=torch.int32, device=device)
        collision_counts.scatter_add_(0, vehicle_for_segment, hit.to(torch.int32))
        curb_too_close = (collision_counts > 0) & valid_vec
        curb_too_close_all[batch_indices, agent_indices] = curb_too_close

        if return_topk:
            curb_topk = {
                "batch_indices": batch_indices,
                "agent_indices": agent_indices,
                "segments": batch_road_segs,  # [num_controlled, K, M-1, 2, 2]
                "masks": batch_road_masks,  # [num_controlled, K]
            }
            return curb_too_close_all, curb_topk
        return curb_too_close_all

    def calculate_speed_reward(
        self,
        speed: torch.Tensor,
        speed_limit: torch.Tensor,
        static_speed_weight: torch.Tensor,
        max_overspeed_value_threshold: torch.Tensor,
        over_speed_scale: torch.Tensor,
    ):
        zero_limit = speed_limit <= 0
        reward = self.compute_speed_reward_regular(
            speed,
            speed_limit,
            static_speed_weight,
            max_overspeed_value_threshold,
            over_speed_scale,
        )
        reward = torch.where(zero_limit, torch.ones_like(reward), reward)
        return reward

    def _narrow_road_right_preferance(
        self,
        position: torch.Tensor,
        orientation: torch.Tensor,
        curb_topk: dict | None,
        has_near_non_curb_lane: torch.Tensor,
        max_width_threshold: float = 7.8,
        min_width_threshold: float = 4.5,
        left_bias_threshold: float = 0.5,
    ) -> torch.Tensor:
        """Detect off-center on narrow roads where ego should keep right.

        Rule:
        - Compute nearest left/right curb distances in ego frame.
        - If min_width_threshold <= (left + right) <= max_width_threshold and
          left + left_bias_threshold < right and no non-curb lane nearby,
          treat as wrong_way_active (too close to left on a lane-less rural road).
        """
        N, A, _ = position.shape
        device = position.device
        off_center = torch.zeros((N, A), dtype=torch.bool, device=device)

        if not curb_topk:
            return off_center

        batch_indices = curb_topk["batch_indices"]
        agent_indices = curb_topk["agent_indices"]
        curb_segments_topk = curb_topk["segments"]  # [num_controlled, K, M-1, 2, 2]
        curb_masks = curb_topk["masks"]  # [num_controlled, K]

        if batch_indices.numel() == 0:
            return off_center

        pos_ctrl = position[batch_indices, agent_indices]  # [num_controlled, 2]
        ori_ctrl = orientation[batch_indices, agent_indices]  # [num_controlled]

        curb_mask_topk = curb_masks.unsqueeze(-1).expand(
            -1, -1, curb_segments_topk.shape[2]
        )  # [num_controlled, K, M-1]

        # Project each controlled position onto nearby curb segments and get distances.
        pos_exp = pos_ctrl.unsqueeze(1).unsqueeze(2)  # [num_controlled, 1, 1, 2]
        start_exp = curb_segments_topk[..., 0, :]
        end_exp = curb_segments_topk[..., 1, :]

        proj, t, dist = project_points_to_segments(
            pos_exp, start_exp, end_exp, return_dist=True
        )
        diff = proj - pos_exp
        valid_proj = (t > 1e-6) & (t < 1.0 - 1e-6)
        dist = torch.where(
            curb_mask_topk & valid_proj, dist, torch.full_like(dist, inf)
        )

        # Signed lateral distance in ego frame: >0 left, <0 right.
        left_axis = torch.stack(
            (-torch.sin(ori_ctrl), torch.cos(ori_ctrl)), dim=-1
        )  # [num_controlled, 2]
        lateral = (diff * left_axis.unsqueeze(1).unsqueeze(2)).sum(dim=-1)

        # Nearest curb distances on each side.
        left_dist = torch.where(lateral > 0, dist, torch.full_like(dist, inf))
        right_dist = torch.where(lateral < 0, dist, torch.full_like(dist, inf))
        left_min = left_dist.amin(dim=(1, 2))
        right_min = right_dist.amin(dim=(1, 2))

        finite_sides = torch.isfinite(left_min) & torch.isfinite(right_min)
        total_width = left_min + right_min
        narrow_road = (total_width >= min_width_threshold) & (
            total_width <= max_width_threshold
        )
        left_too_close = left_min - right_min < left_bias_threshold
        no_lane_between_curbs = ~has_near_non_curb_lane[batch_indices, agent_indices]

        off_center_ctrl = (
            finite_sides & narrow_road & left_too_close & no_lane_between_curbs
        )
        off_center[batch_indices, agent_indices] = off_center_ctrl
        return off_center

    @staticmethod
    def compute_speed_reward_regular(
        speed: torch.Tensor,
        speed_limit: torch.Tensor,
        static_speed_weight: torch.Tensor,
        max_overspeed_value_threshold: torch.Tensor,
        over_speed_scale: torch.Tensor,
    ) -> torch.Tensor:
        del static_speed_weight, over_speed_scale
        max_overspeed_value_threshold = torch.as_tensor(
            max_overspeed_value_threshold, device=speed.device, dtype=speed.dtype
        ).clamp_min(1e-3)

        exceeding_speed = speed - speed_limit
        violation_loss = exceeding_speed / max_overspeed_value_threshold
        over = (1.0 - violation_loss).clamp(0.0, 1.0)
        return torch.where(exceeding_speed > 0.0, over, torch.ones_like(speed))

    @staticmethod
    def compute_speed_reward_zero(
        speed: torch.Tensor, decay: float = 5.0
    ) -> torch.Tensor:
        # Smoothly decay from 1.0 at standstill to an asymptotic floor of 0.2.
        return 0.2 + 0.8 * torch.exp(-speed / decay)

    def calculate_nearest_line(
        self,
        position: torch.Tensor,
        orientation: torch.Tensor,
        center_line: torch.Tensor,
        center_line_mask,
    ):
        N, S, _, _ = center_line.shape
        _, A, _ = position.shape

        pos_exp = position.unsqueeze(2)  # [N, A, 1, 2]
        start_exp = center_line[:, :, 0, :].unsqueeze(1)  # [N, 1, S, 2]
        end_exp = center_line[:, :, 1, :].unsqueeze(1)  # [N, 1, S, 2]
        mask_exp = center_line_mask.unsqueeze(1).expand(N, A, S)

        # Project onto segments, then compute distances and baseline progress.
        _, proj_t, dist = project_points_to_segments(
            pos_exp, start_exp, end_exp, return_dist=True
        )
        dist = torch.where(mask_exp, dist, inf)
        min_dist, min_idx = dist.min(dim=2)  # [N, A]

        seg_vec = center_line[:, :, 1, :] - center_line[:, :, 0, :]  # [N, S, 2]
        seg_len = torch.linalg.norm(seg_vec, dim=-1)
        gather_idx = min_idx.unsqueeze(-1)
        nearest_len = (
            seg_len.unsqueeze(1).expand(N, A, S).gather(2, gather_idx).squeeze(-1)
        )
        nearest_t = proj_t.gather(2, gather_idx).squeeze(-1)
        progress_on_segment = nearest_t.clamp(0.0, 1.0) * nearest_len

        return min_dist, min_idx, progress_on_segment

    @staticmethod
    def calculate_baseline_progress(
        position: torch.Tensor,
        baselines: torch.Tensor,
        baseline_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return nearest baseline index and arc length from its first point."""
        num_envs, num_lines, num_points, _ = baselines.shape
        num_segments = num_points - 1
        segments = torch.stack(
            (baselines[..., :-1, :2], baselines[..., 1:, :2]), dim=-2
        ).reshape(num_envs, num_lines * num_segments, 2, 2)

        starts = segments[:, :, 0].unsqueeze(1)
        ends = segments[:, :, 1].unsqueeze(1)
        _, projection_t, distance = project_points_to_segments(
            position.unsqueeze(2), starts, ends, return_dist=True
        )
        segment_mask = (
            baseline_mask.unsqueeze(-1)
            .expand(num_envs, num_lines, num_segments)
            .reshape(num_envs, num_lines * num_segments)
        )
        distance = torch.where(
            segment_mask.unsqueeze(1), distance, torch.full_like(distance, inf)
        )
        nearest_segment = distance.argmin(dim=2)
        baseline_idx = nearest_segment // num_segments

        segment_lengths = torch.linalg.norm(
            segments[..., 1, :] - segments[..., 0, :], dim=-1
        )
        lengths_by_line = segment_lengths.reshape(num_envs, num_lines, num_segments)
        distance_before_segment = torch.cat(
            (
                torch.zeros_like(lengths_by_line[..., :1]),
                lengths_by_line.cumsum(dim=-1)[..., :-1],
            ),
            dim=-1,
        ).reshape(num_envs, num_lines * num_segments)
        gather_idx = nearest_segment.unsqueeze(-1)
        segment_progress = projection_t.gather(2, gather_idx).squeeze(-1).clamp(
            0.0, 1.0
        ) * segment_lengths.unsqueeze(1).expand_as(projection_t).gather(
            2, gather_idx
        ).squeeze(-1)
        progress_along_line = segment_progress + distance_before_segment.unsqueeze(
            1
        ).expand_as(projection_t).gather(2, gather_idx).squeeze(-1)
        return baseline_idx, progress_along_line

    def compute_driving_direction_violation(
        self,
        scenario_data: ScenarioData,
        baseline_idx: torch.Tensor,
        progress_along_baseline: torch.Tensor,
        in_penalty_band: torch.Tensor,
        valid_centerline: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        horizon_frames = max(
            int(round(self._direction_time_horizon / self._frame_time_interval)), 1
        )
        last_progress = getattr(scenario_data, "_direction_progress_last", None)
        last_baseline = getattr(scenario_data, "_direction_baseline_last", None)
        if (
            last_progress is None
            or last_progress.shape != progress_along_baseline.shape
        ):
            last_progress = progress_along_baseline
            last_baseline = baseline_idx

        same_baseline = last_baseline == baseline_idx
        valid = in_penalty_band & valid_centerline & same_baseline
        progress_delta = torch.where(
            valid,
            progress_along_baseline - last_progress,
            torch.zeros_like(progress_along_baseline),
        )

        buf = getattr(scenario_data, "_direction_progress_buffer", None)
        if (
            buf is None
            or buf.shape[:2] != progress_delta.shape
            or buf.shape[2] != horizon_frames
        ):
            buf = torch.zeros(
                (*progress_delta.shape, horizon_frames),
                device=progress_delta.device,
                dtype=progress_delta.dtype,
            )
        buf = torch.cat([buf[:, :, 1:], progress_delta.unsqueeze(-1)], dim=2)

        scenario_data._direction_progress_buffer = buf
        scenario_data._direction_progress_last = progress_along_baseline
        scenario_data._direction_baseline_last = baseline_idx

        progress_over_horizon = buf.sum(dim=2)
        return (
            progress_over_horizon < -self._direction_compliance_threshold,
            progress_over_horizon < -self._direction_violation_threshold,
        )
