import math

import torch

from driverl.datatypes.scenario_data import ScenarioData
from driverl.env.engine.config import EngineMergedConfig
from driverl.env.engine.reward_calculator.base_reward_calculator import (
    REWARD_CALCULATOR_REGISTER,
    BaseRewardCalculator,
)
from driverl.utils.geometry import segment_polygons_intersection2


@REWARD_CALCULATOR_REGISTER.register_module
class NuPlanCollision(BaseRewardCalculator):
    """CaRL-style nuPlan collision reward: penalize new non-stationary collisions."""

    STOPPED_SPEED_THRESHOLD = 5e-2
    BEHIND_COS_THRESHOLD = math.cos(math.radians(150.0))

    def __init__(self, config: EngineMergedConfig):
        super().__init__(config)

    def forward(
        self,
        scenario_data: "ScenarioData",
        log_scenario_data: "ScenarioData",
        rewards_and_infos: dict,
        **kwargs,
    ):
        randomized_features = scenario_data.randomized_features
        polygons_now, _, valid = scenario_data.get_recent_agent_polygons()
        controlled = scenario_data.agent_control_manager.controlled_mask
        collision_matrix = self.detect_current_collisions_knn(
            polygons_now,
            valid,
            controlled,
            scenario_data.agent_nearest_indices[:, :, -1, :],
        )
        speed = torch.linalg.norm(scenario_data.agent_velocity_all[:, :, -1, :], dim=-1)
        ego_moving = speed > self.STOPPED_SPEED_THRESHOLD
        collision_info = collision_matrix.any(dim=2) & controlled & ego_moving
        collision_reward_weight = randomized_features.get(
            "collision_reward_weight", True
        )
        reward = collision_info.float() * collision_reward_weight

        rewards_and_infos["NuPlanCollision"] = {
            "reward": reward,
            "info": collision_info,
        }

    @staticmethod
    def _polygon_overlap_pairs(
        poly_i: torch.Tensor, poly_j: torch.Tensor
    ) -> torch.Tensor:
        edges_i = torch.stack([poly_i, torch.roll(poly_i, shifts=-1, dims=1)], dim=2)
        edges_j = torch.stack([poly_j, torch.roll(poly_j, shifts=-1, dims=1)], dim=2)

        i_hits_j = segment_polygons_intersection2(
            edges_i.reshape(-1, 2, 2),
            poly_j.repeat_interleave(poly_i.shape[1], dim=0),
        ).view(poly_i.shape[0], poly_i.shape[1])
        j_hits_i = segment_polygons_intersection2(
            edges_j.reshape(-1, 2, 2),
            poly_i.repeat_interleave(poly_j.shape[1], dim=0),
        ).view(poly_i.shape[0], poly_j.shape[1])
        return i_hits_j.any(dim=1) | j_hits_i.any(dim=1)

    def detect_current_collisions_knn(
        self,
        polygons_now: torch.Tensor,
        valid: torch.Tensor,
        controlled: torch.Tensor,
        knn_indices: torch.Tensor,
    ) -> torch.Tensor:
        N, A, _, _ = polygons_now.shape
        device = polygons_now.device
        collision_matrix = torch.zeros((N, A, A), dtype=torch.bool, device=device)

        batch_idx, ego_idx = torch.where(controlled)
        if batch_idx.numel() == 0:
            return collision_matrix

        num_neighbors = min(9, A, knn_indices.shape[-1])
        nn_indices = knn_indices[batch_idx, ego_idx, :num_neighbors]
        ego_expanded = ego_idx.view(-1, 1).expand(-1, num_neighbors)
        pair_row, pair_col = torch.where(ego_expanded != nn_indices)
        if pair_row.numel() == 0:
            return collision_matrix

        pair_batch = batch_idx[pair_row]
        pair_ego = ego_idx[pair_row]
        pair_other = nn_indices[pair_row, pair_col]
        pair_valid = valid[pair_batch, pair_ego] & valid[pair_batch, pair_other]

        hit = self._polygon_overlap_pairs(
            polygons_now[pair_batch, pair_ego],
            polygons_now[pair_batch, pair_other],
        )
        collision_matrix[pair_batch, pair_ego, pair_other] = hit & pair_valid
        return collision_matrix
