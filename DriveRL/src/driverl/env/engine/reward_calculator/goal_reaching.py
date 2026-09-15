import torch

from driverl.datatypes.goal_position_utils import normalize_goal_positions
from driverl.datatypes.scenario_data import ScenarioData
from driverl.env.engine.config import EngineMergedConfig
from driverl.env.engine.reward_calculator.base_reward_calculator import (
    REWARD_CALCULATOR_REGISTER,
    BaseRewardCalculator,
)


@REWARD_CALCULATOR_REGISTER.register_module
class GoalReaching(BaseRewardCalculator):
    """
    Calculates a reward for reaching the log agent's last position as fast as possible.
    The reward is based on the distance to the target position and provides a positive
    reward when the agent gets close to the target.
    """

    def __init__(self, config: EngineMergedConfig):
        """Initialize the goal reaching reward calculator."""
        super().__init__(config)
        # Weight for distance-based reward (encourages getting closer)
        self.num_steps = config.num_steps
        self.done_after_reaching_goal = config.done_after_reaching_goal

    @staticmethod
    def compute_rewards_from_goal_reached(
        distances: torch.Tensor,
        goal_reached: torch.Tensor,
        goal_reached_before: torch.Tensor,
        goal_reaching_distance_weight,
        num_steps,
        goal_reaching_weight,
        done_after_reaching_goal,
        survival_reward,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Compute rewards
        # 1. When goal is not reached, add a distance penalty.
        total_rewards = -distances * goal_reaching_distance_weight
        total_rewards = torch.clamp(total_rewards, min=-1.0 / num_steps)

        # 2. The first time we reached the goal, set a goal_rewards.
        first_time_reached = goal_reached & ~goal_reached_before
        total_rewards[first_time_reached] = goal_reaching_weight[first_time_reached]

        # 3. After reaching the goal, set our survival_reward if the episode doesn't end.
        if not done_after_reaching_goal:
            total_rewards[goal_reached_before] = survival_reward[goal_reached_before]

        return total_rewards, first_time_reached

    def forward(
        self,
        scenario_data: "ScenarioData",
        log_scenario_data: "ScenarioData",
        rewards_and_infos: dict,
        **kwargs,
    ):
        """
        Computes the goal reaching reward and goal reached information.

        Args:
            scenario_data (ScenarioData): The current state of the environment.
            log_scenario_data (ScenarioData): The logged scenario data.
            rewards_and_infos (dict): A dictionary to store the rewards and infos.
            **kwargs: Additional arguments.
        """
        T = scenario_data.agent_positions_all.shape[2]
        if T > 1:
            previous_positions = scenario_data.agent_positions_all[
                :, :, -2
            ]  # [N, A, 2]
        else:
            previous_positions = scenario_data.agent_positions_all[
                :, :, -1
            ]  # [N, A, 2]

        randomized_features = scenario_data.randomized_features
        goal_reaching_threshold = randomized_features.get(
            "goal_reaching_threshold", calculate=True
        )
        goal_reaching_weight = randomized_features.get(
            "goal_reaching_weight", calculate=True
        )
        goal_reaching_distance_weight = randomized_features.get(
            "goal_reaching_distance_weight", calculate=True
        )
        survival_reward = (
            randomized_features.get("survival_reward_numerator", calculate=True)
            / self.num_steps
        )

        # Get current positions of all agents
        current_positions = scenario_data.agent_positions_all[:, :, -1]  # [N, A, 2]

        # Get all goal positions from the scenario data at the current frame.
        target_positions = normalize_goal_positions(
            scenario_data.goal_positions[:, :, -1, :]
        )  # [N, A, 2 * G]
        target_positions = target_positions.reshape(
            *target_positions.shape[:2], target_positions.shape[-1] // 2, 2
        )  # [N, A, G, 2]

        # Calculate vectors
        AP = target_positions - previous_positions.unsqueeze(2)  # [N, A, G, 2]
        AB = current_positions - previous_positions  # [N, A, 2]
        AB_exp = AB.unsqueeze(2)  # [N, A, 1, 2]

        # Calculate the projection of AP on AB (and the scale)
        dot_product = (AP * AB_exp).sum(dim=-1)  # [N, A, G]
        ab_length_sq = (AB * AB).sum(dim=2).unsqueeze(-1)  # [N, A, 1]
        ab_length_sq = torch.clamp(ab_length_sq, min=1e-6)

        scale = (dot_product / ab_length_sq).unsqueeze(-1)  # [N, A, G, 1]
        scale_clamped = torch.clamp(scale, 0.0, 1.0)  # Cut off within the line segment

        # Get the nearest point on the segment
        nearest_points = (
            previous_positions.unsqueeze(2) + scale_clamped * AB_exp
        )  # [N, A, G, 2]

        # Calculate distances to target positions
        per_goal_distances = torch.norm(
            target_positions - nearest_points, dim=-1
        )  # [N, A, G]
        controlled_mask = scenario_data.agent_control_manager.controlled_mask  # [N, A]
        if getattr(scenario_data, "goal_pair_mode", "legacy") == "ordered_sequential":
            goal_count = per_goal_distances.shape[-1]
            goal_stage = scenario_data.goal_stage
            if goal_stage.shape != per_goal_distances.shape[:2]:
                raise ValueError(
                    "ordered_sequential goal_stage must have shape [N, A], got "
                    f"{tuple(goal_stage.shape)}."
                )
            active = goal_stage < goal_count
            active_index = goal_stage.clamp(max=goal_count - 1).unsqueeze(-1)
            distances = per_goal_distances.gather(-1, active_index).squeeze(-1)
            stage_reached = active & (distances < goal_reaching_threshold)
            stage_reached &= controlled_mask
            next_goal_stage = torch.where(
                stage_reached, goal_stage + 1, goal_stage
            ).clamp(max=goal_count)

            total_rewards = -distances * goal_reaching_distance_weight
            total_rewards = torch.clamp(total_rewards, min=-1.0 / self.num_steps)
            total_rewards[stage_reached] = goal_reaching_weight[stage_reached]
            completed_before = goal_stage >= goal_count
            if not self.done_after_reaching_goal:
                total_rewards[completed_before] = survival_reward[completed_before]
            final_goal_reached = stage_reached & (next_goal_stage == goal_count)
            first_time_reached = final_goal_reached
        else:
            distances = per_goal_distances.min(dim=-1).values  # [N, A]
            goal_reached = distances < goal_reaching_threshold  # [N, A]
            goal_reached_before = scenario_data.agent_goal_reached_all[..., -1]
            total_rewards, first_time_reached = self.compute_rewards_from_goal_reached(
                distances,
                goal_reached,
                goal_reached_before,
                goal_reaching_distance_weight,
                self.num_steps,
                goal_reaching_weight,
                self.done_after_reaching_goal,
                survival_reward,
            )
            next_goal_stage = None
            stage_reached = first_time_reached

        reward = torch.where(controlled_mask, total_rewards, 0.0)
        info = torch.where(controlled_mask, first_time_reached.int(), 0)

        result = {"reward": reward, "info": info, "stage_reached": stage_reached}
        if next_goal_stage is not None:
            result["next_goal_stage"] = next_goal_stage
        rewards_and_infos["GoalReaching"] = result
