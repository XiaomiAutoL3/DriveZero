import math

import torch

from driverl.datatypes.scenario_data import ScenarioData
from driverl.env.engine.config import EngineMergedConfig
from driverl.env.engine.reward_calculator.base_reward_calculator import (
    REWARD_CALCULATOR_REGISTER,
    BaseRewardCalculator,
)
from driverl.env.engine.reward_calculator.collision.nuplan_collision import (
    NuPlanCollision,
)


@REWARD_CALCULATOR_REGISTER.register_module
class NuPlanTTC(BaseRewardCalculator):
    """nuPlan-style time_to_collision_within_bound reward."""

    TIME_STEP_SIZE = 0.1
    TIME_HORIZON = 3.0
    LEAST_MIN_TTC = 0.95
    STOPPED_SPEED_THRESHOLD = 5e-3
    AHEAD_COS_THRESHOLD = math.cos(math.radians(30.0))
    BEHIND_COS_THRESHOLD = math.cos(math.radians(150.0))

    def __init__(self, config: EngineMergedConfig):
        super().__init__(config)
        self._collision = NuPlanCollision(config)

    @classmethod
    def _reward_from_ttc(cls, ttc_info: torch.Tensor) -> torch.Tensor:
        return (
            (ttc_info - cls.LEAST_MIN_TTC) / (cls.TIME_HORIZON - cls.LEAST_MIN_TTC)
        ).clamp(0.0, 1.0)

    @staticmethod
    def _box_dimensions(polygons: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        front_center = 0.5 * (polygons[:, 0] + polygons[:, 1])
        rear_center = 0.5 * (polygons[:, 2] + polygons[:, 3])
        length = torch.linalg.norm(front_center - rear_center, dim=-1)
        width = torch.linalg.norm(polygons[:, 0] - polygons[:, 1], dim=-1)
        return length, width

    @staticmethod
    def _build_oriented_boxes(
        centers: torch.Tensor,
        yaws: torch.Tensor,
        lengths: torch.Tensor,
        widths: torch.Tensor,
    ) -> torch.Tensor:
        heading = torch.stack([torch.cos(yaws), torch.sin(yaws)], dim=-1)
        left = torch.stack([-torch.sin(yaws), torch.cos(yaws)], dim=-1)
        half_length = 0.5 * lengths.unsqueeze(-1)
        half_width = 0.5 * widths.unsqueeze(-1)

        front = centers + heading * half_length
        rear = centers - heading * half_length
        return torch.stack(
            (
                front + left * half_width,
                front - left * half_width,
                rear - left * half_width,
                rear + left * half_width,
            ),
            dim=1,
        )

    def _classify_current_at_fault_collisions_for_ttc(
        self,
        scenario_data: "ScenarioData",
        polygons_now: torch.Tensor,
        collision_matrix: torch.Tensor,
    ) -> torch.Tensor:
        """Lightweight nuPlan current-collision gate for TTC.

        Official TTC only needs timestamps where ego is already in an at-fault
        collision. It reuses nuPlan's collision type logic, but the lateral
        lane-change part depends on map route objects. We keep the exact
        stopped-track/front/rear rules here and intentionally avoid the
        lane-mismatch responsibility path used by NuPlanCollision.
        """
        at_fault = torch.zeros_like(collision_matrix)
        batch_idx, ego_idx, other_idx = torch.where(collision_matrix)
        if batch_idx.numel() == 0:
            return at_fault

        velocities = scenario_data.agent_velocity_all[:, :, -1, :]
        speed = torch.linalg.norm(velocities, dim=-1)
        ego_stopped = (
            speed[batch_idx, ego_idx] <= NuPlanCollision.STOPPED_SPEED_THRESHOLD
        )
        other_stopped = (
            speed[batch_idx, other_idx] <= NuPlanCollision.STOPPED_SPEED_THRESHOLD
        )

        ego_centers = polygons_now[batch_idx, ego_idx].mean(dim=1)
        other_centers = polygons_now[batch_idx, other_idx].mean(dim=1)
        ego_yaw = scenario_data.agent_orientation_all[batch_idx, ego_idx, -1]
        rel = other_centers - ego_centers
        rel_norm = torch.linalg.norm(rel, dim=-1).clamp_min(1e-6)
        heading = torch.stack([torch.cos(ego_yaw), torch.sin(ego_yaw)], dim=-1)
        cos_angle = (rel * heading).sum(dim=-1) / rel_norm
        active_rear = (
            (~ego_stopped)
            & (~other_stopped)
            & (cos_angle < NuPlanCollision.BEHIND_COS_THRESHOLD)
        )

        front_edges = polygons_now[batch_idx, ego_idx][:, [0, 1]]
        active_front = (
            (~ego_stopped)
            & (~other_stopped)
            & (~active_rear)
            & self._collision._polygon_overlap_pairs(
                front_edges,
                polygons_now[batch_idx, other_idx],
            )
        )
        at_fault_pair = active_front | ((~ego_stopped) & other_stopped)
        at_fault[batch_idx, ego_idx, other_idx] = at_fault_pair
        return at_fault

    def forward(
        self,
        scenario_data: "ScenarioData",
        log_scenario_data: "ScenarioData",
        rewards_and_infos: dict,
        **kwargs,
    ):
        polygons_now, _, valid = scenario_data.get_recent_agent_polygons()
        controlled = scenario_data.agent_control_manager.controlled_mask
        knn_indices = scenario_data.agent_nearest_indices[:, :, -1, :]

        current_collision = self._collision.detect_current_collisions_knn(
            polygons_now,
            valid,
            controlled,
            knn_indices,
        )
        at_fault = self._classify_current_at_fault_collisions_for_ttc(
            scenario_data,
            polygons_now,
            current_collision,
        )

        prev_collided = getattr(scenario_data, "_nuplan_ttc_collided_matrix", None)
        if prev_collided is None or prev_collided.shape != current_collision.shape:
            prev_collided = torch.zeros_like(current_collision)
        excluded_collisions = prev_collided | current_collision
        scenario_data._nuplan_ttc_collided_matrix = excluded_collisions

        ttc_info = self.compute_time_to_collision(
            scenario_data=scenario_data,
            polygons_now=polygons_now,
            valid=valid,
            controlled=controlled,
            knn_indices=knn_indices,
            excluded_collisions=excluded_collisions,
        )
        ttc_info = torch.where(
            at_fault.any(dim=2), torch.zeros_like(ttc_info), ttc_info
        )

        ttc_reward = self._reward_from_ttc(ttc_info)
        triggered = controlled & (ttc_info <= self.LEAST_MIN_TTC)
        prev_triggered = getattr(scenario_data, "_nuplan_ttc_triggered", None)
        if prev_triggered is None or prev_triggered.shape != triggered.shape:
            prev_triggered = torch.zeros_like(triggered)
        ttc_reward = ttc_reward * torch.where(
            prev_triggered, ttc_reward.new_tensor(0.5), ttc_reward.new_tensor(1.0)
        )
        scenario_data._nuplan_ttc_triggered = prev_triggered | triggered
        noop_reward = torch.ones_like(ttc_reward)
        noop_info = ttc_info.new_full(ttc_info.shape, self.TIME_HORIZON)

        rewards_and_infos["NuPlanTTC"] = {
            "reward": ttc_reward,
            "info": torch.stack(
                (ttc_info, noop_info, noop_info, noop_info, noop_info), dim=-1
            ),
            "ttc_reward": ttc_reward,
            "tto_reward": noop_reward,
            "ttg_reward": noop_reward,
            "tts_reward": noop_reward,
            "ttc_worst_case_reward": noop_reward,
            "ttc_responsibility": torch.zeros_like(ttc_reward),
        }

    def compute_time_to_collision(
        self,
        scenario_data: "ScenarioData",
        polygons_now: torch.Tensor,
        valid: torch.Tensor,
        controlled: torch.Tensor,
        knn_indices: torch.Tensor,
        excluded_collisions: torch.Tensor,
    ) -> torch.Tensor:
        N, A, _, _ = polygons_now.shape
        device = polygons_now.device
        ttc_all = polygons_now.new_full((N, A), self.TIME_HORIZON)

        batch_idx, ego_idx = torch.where(controlled)
        if batch_idx.numel() == 0 or A <= 1:
            return ttc_all

        num_neighbors = min(A, knn_indices.shape[-1])
        nn_indices = knn_indices[batch_idx, ego_idx, :num_neighbors]
        ego_expanded = ego_idx.view(-1, 1).expand(-1, num_neighbors)
        pair_row, pair_col = torch.where(ego_expanded != nn_indices)
        if pair_row.numel() == 0:
            return ttc_all

        pair_batch = batch_idx[pair_row]
        pair_ego = ego_idx[pair_row]
        pair_other = nn_indices[pair_row, pair_col]

        velocities = scenario_data.agent_velocity_all[:, :, -1, :]
        orientations = scenario_data.agent_orientation_all[:, :, -1]
        speed = torch.linalg.norm(velocities, dim=-1)
        ego_speed = speed[pair_batch, pair_ego]
        other_speed = speed[pair_batch, pair_other]
        ego_yaw = orientations[pair_batch, pair_ego]
        other_yaw = orientations[pair_batch, pair_other]

        ego_centers = polygons_now[pair_batch, pair_ego].mean(dim=1)
        other_centers = polygons_now[pair_batch, pair_other].mean(dim=1)
        rel = other_centers - ego_centers
        rel_norm = torch.linalg.norm(rel, dim=-1).clamp_min(1e-6)
        ego_heading = torch.stack([torch.cos(ego_yaw), torch.sin(ego_yaw)], dim=-1)
        cos_angle = (rel * ego_heading).sum(dim=-1) / rel_norm
        ahead = cos_angle > self.AHEAD_COS_THRESHOLD
        behind = cos_angle < self.BEHIND_COS_THRESHOLD

        # Route/intersection masks are not available in the tensor state here;
        # keep all non-rear tracks as TTC candidates.
        relevant = ahead | (~behind)
        pair_valid = (
            valid[pair_batch, pair_ego]
            & valid[pair_batch, pair_other]
            & relevant
            & (ego_speed > self.STOPPED_SPEED_THRESHOLD)
            & (~excluded_collisions[pair_batch, pair_ego, pair_other])
        )
        if not pair_valid.any():
            return ttc_all

        valid_pair_idx = torch.where(pair_valid)[0]
        if valid_pair_idx.numel() == 0:
            return ttc_all

        pair_batch = pair_batch[valid_pair_idx]
        pair_ego = pair_ego[valid_pair_idx]
        pair_other = pair_other[valid_pair_idx]
        pair_row = pair_row[valid_pair_idx]
        ego_speed = ego_speed[valid_pair_idx]
        other_speed = other_speed[valid_pair_idx]
        ego_yaw = ego_yaw[valid_pair_idx]
        other_yaw = other_yaw[valid_pair_idx]
        ego_heading = ego_heading[valid_pair_idx]

        ego_polygons = polygons_now[pair_batch, pair_ego]
        other_polygons = polygons_now[pair_batch, pair_other]
        ego_centers = ego_centers[valid_pair_idx]
        other_centers = other_centers[valid_pair_idx]

        ego_length, ego_width = self._box_dimensions(ego_polygons)
        other_length, other_width = self._box_dimensions(other_polygons)
        other_heading = torch.stack(
            [torch.cos(other_yaw), torch.sin(other_yaw)], dim=-1
        )

        ego_distance = ego_speed * self.TIME_HORIZON
        other_distance = other_speed * self.TIME_HORIZON
        ego_elongated = self._build_oriented_boxes(
            ego_centers + ego_heading * (0.5 * ego_distance).unsqueeze(-1),
            ego_yaw,
            ego_length + ego_distance,
            ego_width,
        )
        other_elongated = self._build_oriented_boxes(
            other_centers + other_heading * (0.5 * other_distance).unsqueeze(-1),
            other_yaw,
            other_length + other_distance,
            other_width,
        )
        relevant_pair = self._collision._polygon_overlap_pairs(
            ego_elongated,
            other_elongated,
        )
        if not relevant_pair.any():
            return ttc_all

        pair_batch = pair_batch[relevant_pair]
        pair_ego = pair_ego[relevant_pair]
        pair_other = pair_other[relevant_pair]
        pair_row = pair_row[relevant_pair]
        ego_speed = ego_speed[relevant_pair]
        other_speed = other_speed[relevant_pair]
        ego_heading = ego_heading[relevant_pair]
        other_heading = other_heading[relevant_pair]
        ego_polygons = ego_polygons[relevant_pair]
        other_polygons = other_polygons[relevant_pair]

        v1 = ego_heading * ego_speed.unsqueeze(-1)
        v2 = other_heading * other_speed.unsqueeze(-1)

        times = torch.arange(
            self.TIME_STEP_SIZE,
            self.TIME_HORIZON,
            self.TIME_STEP_SIZE,
            device=device,
            dtype=polygons_now.dtype,
        )
        poly1 = ego_polygons.unsqueeze(0) + (
            v1.unsqueeze(0).unsqueeze(2) * times.view(-1, 1, 1, 1)
        )
        poly2 = other_polygons.unsqueeze(0) + (
            v2.unsqueeze(0).unsqueeze(2) * times.view(-1, 1, 1, 1)
        )
        S, P = poly1.shape[:2]
        hit = self._collision._polygon_overlap_pairs(
            poly1.reshape(S * P, 4, 2),
            poly2.reshape(S * P, 4, 2),
        ).view(S, P)
        any_hit = hit.any(dim=0)
        first_hit = hit.float().argmax(dim=0)
        ttc_pair = torch.where(
            any_hit,
            times[first_hit],
            polygons_now.new_full((P,), self.TIME_HORIZON),
        )

        num_controlled = batch_idx.shape[0]
        controlled_idx = pair_row
        ttc_min = polygons_now.new_full((num_controlled,), self.TIME_HORIZON)
        ttc_min = ttc_min.scatter_reduce(
            0, controlled_idx, ttc_pair, reduce="amin", include_self=True
        )
        ttc_all[batch_idx, ego_idx] = ttc_min
        return ttc_all
