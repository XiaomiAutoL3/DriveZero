"""Dynamics and reward evaluation used by DriveRL test-time scaling."""

import torch

from driverl.datatypes.scenario_data import ScenarioData
from driverl.env.engine.config import (
    EngineConfig,
    EngineMergedConfig,
    EngineRuntimeConfig,
)
from driverl.env.engine.dynamics_model import BaseDynamicsModel
from driverl.env.engine.reward_calculator import BaseRewardCalculator, InfoDimension
from driverl.env.engine.reward_decomposition import split_soft_product_reward
from driverl.utils.geometry import velocity_body_frame_components

STEERING_WHEEL_RATIO = 12.6


class Engine:
    """Evaluate short model-based rollouts for TTS candidate selection."""

    def __init__(self, config: EngineConfig, runtime_config: EngineRuntimeConfig):
        self.config = EngineMergedConfig.from_configs(config, runtime_config)
        self._dynamics_model = BaseDynamicsModel.dynamics_model_factory(
            runtime_config.dynamics_model, config=self.config
        )
        self._occ_ray_cache: dict = {}
        if self.config.enable_occupancy_grid:
            occ_length = int(
                round(
                    (self.config.occ_grid_xmax - self.config.occ_grid_xmin)
                    / self.config.occ_grid_resolution
                )
            )
            occ_width = int(
                round(
                    (self.config.occ_grid_ymax - self.config.occ_grid_ymin)
                    / self.config.occ_grid_resolution
                )
            )
            self._occ_ray_cache = ScenarioData._get_occ_ray_cache(
                (occ_length, occ_width),
                torch.device(self.config.device),
                self.config,
                self._occ_ray_cache,
            )

        calculator_names = (
            "NuPlanCollision",
            "GoalReaching",
            "OffRoad",
            "CenterLine",
            "Comfort",
            "NuPlanTTC",
        )
        self._reward_calculators = [
            (
                name,
                BaseRewardCalculator.reward_calculator_factory(
                    name, config=self.config
                ),
            )
            for name in calculator_names
        ]
        self.device = self.config.device
        self.num_steps = self.config.num_steps
        self.frame_time_interval = self.config.frame_time_interval
        self._next_goal_stage = None

    def _calculate_rewards(
        self,
        scenario_data: ScenarioData,
        log_scenario_data: ScenarioData,
    ):
        scenario_data.update_nearest_neighbors()
        if self.config.enable_visible_mask:
            scenario_data.update_visible_mask()
        scenario_data.update_occ_surface_points(self.config, self._occ_ray_cache)

        rewards_and_infos = {}
        for _, calculator in self._reward_calculators:
            calculator.forward(scenario_data, log_scenario_data, rewards_and_infos)

        result = self._combine_rewards(rewards_and_infos, scenario_data)
        scenario_data.clean_polygon_cache()
        return result

    def _combine_rewards(
        self,
        rewards_and_infos: dict,
        scenario_data: ScenarioData,
    ):
        num_envs, num_agents = next(iter(rewards_and_infos.values()))["reward"].shape
        dones = torch.zeros(
            (num_envs, num_agents), device=self.device, dtype=torch.bool
        )
        infos = torch.zeros(
            (num_envs, num_agents, InfoDimension.size()), device=self.device
        )
        hard_reward = torch.zeros((num_envs, num_agents), device=self.device)
        goal_reward = torch.zeros((num_envs, num_agents), device=self.device)
        soft_scores: dict[str, torch.Tensor] = {}
        self._next_goal_stage = None
        self._set_kinematics_info(infos, scenario_data)

        for name, results in rewards_and_infos.items():
            reward = results["reward"]
            calculator_info = results["info"]
            if name == "NuPlanCollision":
                hard_reward += reward
                infos[:, :, InfoDimension.COLLISION_FLAG.value] = calculator_info
                infos[:, :, InfoDimension.COLLISION_REWARD.value] = reward
                dones |= calculator_info.bool()
            elif name == "OffRoad":
                hard_reward += reward
                infos[:, :, InfoDimension.OFFROAD_EVENT.value] = calculator_info
                infos[:, :, InfoDimension.OFFROAD_REWARD.value] = reward
                infos[:, :, InfoDimension.OCC_HIT.value] = results["occ_hit"]
                dones |= calculator_info == 1
                dones |= calculator_info == 4
                dones |= results["occ_hit"].bool()
            elif name == "CrossLane":
                infos[:, :, InfoDimension.LANE_CHANGE_INFO.value] = results[
                    "lane_change_info"
                ]
                infos[:, :, InfoDimension.CROSS_LANE_REWARD.value] = reward
                soft_scores["cross_lane"] = reward
                lane_change = results["lane_change_info"].bool()
                dones |= results["solid_lane_change_info"].bool()
                lane_mask = scenario_data.lanes_points_mask.bool()
                lane_speed = scenario_data.lanes_points[..., 0, 3]
                no_speed_limit = lane_mask.any(dim=1) & ~(
                    (lane_speed.abs() > 1e-6) & lane_mask
                ).any(dim=1)
                goal_reached = scenario_data.agent_goal_reached_all[..., -1].bool()
                dones |= lane_change & goal_reached & no_speed_limit.unsqueeze(1)
            elif name == "GoalReaching":
                goal_reward = reward
                self._next_goal_stage = results.get("next_goal_stage")
                infos[:, :, InfoDimension.GOAL_REACHED_FLAG.value] = calculator_info
                infos[:, :, InfoDimension.GOAL_REACHED_REWARD.value] = reward
            elif name == "NuPlanTTC":
                infos[:, :, InfoDimension.TTC_ALERT.value] = calculator_info[..., 0]
                infos[:, :, InfoDimension.TTC_REWARD.value] = results["ttc_reward"]
                infos[:, :, InfoDimension.TTO_ALERT.value] = calculator_info[..., 1]
                infos[:, :, InfoDimension.TTO_REWARD.value] = results["tto_reward"]
                infos[:, :, InfoDimension.TTG_ALERT.value] = calculator_info[..., 2]
                infos[:, :, InfoDimension.TTG_REWARD.value] = results["ttg_reward"]
                infos[:, :, InfoDimension.TTS_ALERT.value] = calculator_info[..., 3]
                infos[:, :, InfoDimension.TTS_REWARD.value] = results["tts_reward"]
                soft_scores["ttc"] = reward
            elif name == "CenterLine":
                infos[:, :, InfoDimension.CENTERLINE_DEVIATION.value] = calculator_info
                infos[:, :, InfoDimension.CENTERLINE_REWARD.value] = reward
                soft_scores["centerline"] = reward
            elif name == "CurbClearance":
                infos[:, :, InfoDimension.CURB_TOO_CLOSE.value] = calculator_info
                infos[:, :, InfoDimension.CURB_CLEARANCE_REWARD.value] = reward
                soft_scores["curb_clearance"] = reward
            elif name == "Comfort":
                infos[:, :, InfoDimension.LONGITUDINAL_ACCEL.value] = calculator_info[
                    :, :, 0
                ]
                infos[:, :, InfoDimension.LATERAL_ACCEL.value] = calculator_info[
                    :, :, 1
                ]
                infos[:, :, InfoDimension.LONGITUDINAL_JERK.value] = calculator_info[
                    :, :, 2
                ]
                infos[:, :, InfoDimension.LATERAL_JERK.value] = calculator_info[:, :, 3]
                infos[:, :, InfoDimension.COMFORT_REWARD.value] = reward
                soft_scores["comfort"] = reward
            elif name == "Overspeed":
                infos[:, :, InfoDimension.OVERSPEED_FLAG.value] = calculator_info
                infos[:, :, InfoDimension.OVERSPEED_REWARD.value] = reward
                soft_scores["overspeed"] = reward
            else:
                soft_scores[name] = reward

        product_of_scores = torch.prod(torch.stack(list(soft_scores.values())), dim=0)
        soft_mask = 1 - dones.float()
        soft_product_reward = soft_mask / self.num_steps * product_of_scores
        goal_component = soft_mask * goal_reward
        soft_components = split_soft_product_reward(soft_scores, soft_product_reward)
        total_reward = hard_reward + soft_mask * (
            goal_reward + product_of_scores / self.num_steps
        )
        reward_components = torch.cat(
            (
                hard_reward.unsqueeze(-1),
                goal_component.unsqueeze(-1),
                soft_components,
            ),
            dim=-1,
        )
        return total_reward, dones, infos, reward_components

    def _set_kinematics_info(
        self, infos: torch.Tensor, scenario_data: ScenarioData
    ) -> None:
        yaws = scenario_data.agent_orientation_all[..., -1]
        velocities = scenario_data.agent_velocity_all[..., -1, :]
        v_long, v_lat = velocity_body_frame_components(velocities, yaws)
        steerings = scenario_data.agent_steering_state_all[..., -1]
        steering_rates = torch.zeros_like(steerings)
        steering_accels = torch.zeros_like(steerings)
        if scenario_data.agent_steering_state_all.shape[-1] >= 2:
            previous = scenario_data.agent_steering_state_all[..., -2]
            steering_rates = (steerings - previous) / self.frame_time_interval
        if scenario_data.agent_steering_state_all.shape[-1] >= 3:
            previous_previous = scenario_data.agent_steering_state_all[..., -3]
            previous_rates = (previous - previous_previous) / self.frame_time_interval
            steering_accels = (
                steering_rates - previous_rates
            ) / self.frame_time_interval
        yaw_rates = (
            scenario_data.agent_yaw_rate_all[..., -1]
            if scenario_data.agent_yaw_rate_all.numel() > 0
            else torch.zeros_like(yaws)
        )
        infos[:, :, InfoDimension.V_LONG.value] = v_long
        infos[:, :, InfoDimension.V_LAT.value] = v_lat
        infos[:, :, InfoDimension.YAW.value] = yaws
        infos[:, :, InfoDimension.YAW_RATE.value] = yaw_rates
        infos[:, :, InfoDimension.WHEEL_ANGLE.value] = (
            torch.rad2deg(steerings) * STEERING_WHEEL_RATIO
        )
        infos[:, :, InfoDimension.WHEEL_RATE.value] = (
            torch.rad2deg(steering_rates) * STEERING_WHEEL_RATIO
        )
        infos[:, :, InfoDimension.WHEEL_ACCEL.value] = (
            torch.rad2deg(steering_accels) * STEERING_WHEEL_RATIO
        )
