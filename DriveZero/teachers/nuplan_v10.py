"""nuPlan Hermes teacher pinned to the v10 mixed50 checkpoint at update 2400."""

from __future__ import annotations

import copy
import math
from typing import Any

import numpy as np
import torch

from hermes.agents import BaseAgent
from hermes.agents.config import AgentConfig
from hermes.datatypes.data_enums import AgentControlManager
from hermes.datatypes.goal_position_utils import route_goal_positions, sample_valid_positions
from hermes.datatypes.scenario_data import ScenarioData
from hermes.env.config import EnvConfig
from hermes.env.constants import MAX_STEERING_ANGLE, MIN_STEERING_ANGLE
from hermes.env.dataloader.web_dataset import HermesWebDataset
from hermes.env.domain_randomization.config import DomainRandomizationConfig
from hermes.env.domain_randomization.domain_randomization import sample_randomized_features
from hermes.env.engine.config import EngineMergedConfig, EngineRuntimeConfig
from hermes.env.engine.engine import Engine
from hermes.env.engine.reward_calculator.base_reward_calculator import InfoDimension
from hermes.env.hermes_env import HermesEnv
from hermes.utils.gym_compat import Box
from hermes.utils.geometry import get_wheelbase_from_length

from utils import drivor_long_trajectory_target, load_model_checkpoint


EXPECTED_HERMES_COMMIT = "4fcda82a845262ea24b5bb4b2019dfd40c70406d"
ROUTE_GOAL_HORIZONS_S = (2.0, 4.0)
FIXED_AGENT_CONFIG = {
    "agent_name": "vanilla_net_agent",
    "network_name": "vanilla_net",
    "enable_occupancy_grid": False,
    "dynamics_model": "nuplan_bicycle_model",
    "enable_value_decomposition": True,
    "embed_dim": 256,
    "num_heads": 4,
    "history_steps": 5,
    "max_agents": 96,
    "max_lanes": 256,
    "max_lane_centers": 64,
    "min_vehicle_agents": 32,
    "min_pedestrian_agents": 32,
    "agent_filter_radius": 200.0,
    "lane_forward_range": 200.0,
    "lane_backward_range": 100.0,
    "use_raw_lane_types": True,
    "use_steering_rate": True,
    "use_ego_history": True,
    "enable_traffic_light_features": True,
    "enable_lane_attention": True,
    "num_lane_attention_layers": 1,
    "num_heads_lane": 4,
    "num_agent_attention_layers": 2,
    "num_goal_positions": 2,
    "scenario_control_type": "mixed",
    "feature_dropout_rate": 0.0,
    "sampling_method": "expectation",
    "mode": "eval",
}

FIXED_ENV_CONFIG = {
    "scenario_control_type": "non_reactive",
    "enable_occupancy_grid": False,
    "action_type": "continuous",
    "dynamics_model": "nuplan_bicycle_model",
    "route_resample_near_goal_prob": 0.0,
    "engine_config": {
        "is_pretrain": False,
        "done_after_reaching_goal": False,
        "dynamics_model_use_velocity_direction": True,
        "enable_visible_mask": False,
        "frame_drop_prob": 0.0,
        "blind_npc_prob": 0.0,
        "enable_dynamics_noise": False,
        "enable_speed_limit_randomization": False,
        "curb_drop_prob": 0.0,
        "occ_drop_prob": 0.0,
        "enable_narrow_road_right_preference_reward": False,
        "enable_route_progress_reward": False,
        "enable_traffic_light_reward": False,
        "a_lat_comfort_low_speed": 5.0,
        "a_lat_comfort_high_speed": 30.0,
        "a_lat_comfort_threshold_low": 4.0,
        "a_lat_comfort_threshold_high": 6.0,
        "a_long_decel_full_speed": 2.78,
        "a_long_decel_floor": -0.8,
        "ttc_heading_deviation_rad": 0.0261799388,
        "enable_ttc_responsibility": True,
        "num_jerk_lat_actions": 64,
        "num_jerk_long_actions": 24,
        "ttc_deadband_frames": 5,
        "nuplan_comfort_min_lon_accel": -4.05,
        "nuplan_comfort_max_lon_accel": 2.4,
        "nuplan_comfort_max_abs_lat_accel": 4.89,
        "nuplan_comfort_max_abs_lon_jerk": 4.13,
        "nuplan_comfort_max_abs_mag_jerk": 8.37,
        "nuplan_comfort_max_abs_yaw_rate": 0.95,
        "nuplan_comfort_max_abs_yaw_accel": 1.93,
        "enable_collision_responsibility": False,
        "collision_responsibility_aggregation": "max",
        "collision_lane_history_sec": 5.0,
        "collision_lane_sample_dt": 0.2,
        "collision_lane_q_sigma": 1.0,
        "collision_lane_occ_sigma": 1.4,
        "collision_segment_bisection_iters": 5,
        "collision_speed_buffer_base": 0.0,
        "collision_speed_buffer_max": 0.0,
        "collision_relative_speed_buffer_gain": 0.0,
        "collision_absolute_speed_buffer_gain": 0.0,
        "braking_editor": {"enabled": False},
        "future_only_editor": {
            "enabled": True,
            "trigger_position_threshold_m": 5.0,
            "trigger_heading_threshold_rad": 0.35,
        },
        "min_jerk_long": -8.0,
        "max_jerk_long": 5.0,
        "positive_jerk_limit_when_nonnegative_acc": 1.0,
        "min_jerk_lat": -1.0,
        "max_jerk_lat": 1.0,
        "min_a_long": -6.5,
        "max_a_long": 1.5,
        "min_a_lat": -3.0,
        "max_a_lat": 3.0,
        "nuplan_bicycle_accel_time_constant": 0.2,
        "nuplan_bicycle_steering_angle_time_constant": 0.05,
        "nuplan_bicycle_max_steering_angle": 1.0471975511965976,
        "nuplan_bicycle_max_acceleration": 4.0,
        "nuplan_bicycle_max_steering_rate": 0.8,
        "nuplan_bicycle_wheelbase": 3.089,
        "nuplan_bicycle_rear_axle_to_center": 1.461,
        "domain_randomization": {
            "enabled": False,
            "collision_reward_weight": {"default": -1.0},
            "collision_segment_time": {"default": 0.2, "calculator_default": 0.2},
            "collision_distance_threshold": {"default": 0.0},
            "ttc_weight": {"default": 0.0},
            "ttc_time": {"default": 4.0},
            "ttc_width_buffer": {"default": 0.0},
            "ttc_length_buffer": {"default": 0.0},
            "off_road_weight": {"default": -2.0},
            "cross_lane_weight": {"default": 1.0},
            "wrong_way_weight": {"default": 0.0},
            "goal_reaching_threshold": {"default": 1.5},
            "goal_reaching_weight": {"default": 0.5},
            "goal_reaching_distance_weight": {"default": 0.0},
            "survival_reward_numerator": {"default": 0.0},
            "comfort_weight": {"default": 0.0},
            "deviation_distance_weight": {"default": 0.75},
            "deviation_angle_limit": {"default": 0.75},
            "deviation_angle_weight": {"default": 0.0},
            "max_deviation_distance_for_penalty": {"default": 4.0},
            "curb_clearance_distance": {"default": 0.0},
            "curb_clearance_weight": {"default": 1.0},
            "static_speed_weight": {"default": 1.0},
            "over_speed_threshold": {"default": 2.23},
            "over_speed_scale": {"default": 1.0},
            "width_buffer": {"default": 0.0, "calculator_default": 0.0},
            "position_weight": {"default": 1.0},
            "velocity_weight": {"default": 1.0},
            "heading_weight": {"default": 1.0},
            "position_threshold": {"default": 2.0},
            "velocity_threshold": {"default": 2.0},
            "heading_threshold": {"default": 0.5},
        },
    },
    "dataloader_config": {
        "dataset_type": "webdataset",
        "dataset_files": {"train": ["unused"]},
    },
}


class _TargetRolloutEngine(Engine):
    @staticmethod
    def _route_goal_positions(positions, velocities, scenario_data):
        goals = [
            route_goal_positions(
                positions,
                velocities,
                scenario_data.route_points_array,
                scenario_data.route_points_mask,
                horizon_s=horizon,
            )
            for horizon in ROUTE_GOAL_HORIZONS_S
        ]
        return torch.cat(goals, dim=-1)

    def _calculate_rewards(self, scenario_data, log_scenario_data):
        scenario_data.update_occ_surface_points(self.config, self._occ_ray_cache)
        scenario_data.occ_surface_points_all = scenario_data.occ_surface_points_all[
            :, :, -1:
        ].contiguous()
        rewards_and_infos = {}
        for name, calculator in self._reward_calculators:
            if name == "GoalReaching":
                calculator.forward(scenario_data, log_scenario_data, rewards_and_infos)
                break
        num_envs, num_agents = scenario_data.agent_positions_all.shape[:2]
        info = torch.zeros(
            (num_envs, num_agents, InfoDimension.size()), device=self.device
        )
        if "GoalReaching" in rewards_and_infos:
            info[:, :, InfoDimension.GOAL_REACHED_FLAG.value] = rewards_and_infos[
                "GoalReaching"
            ]["info"]
        elif not self.config.is_pretrain:
            raise RuntimeError("GoalReaching calculator is required for Teacher rollout.")
        dones = torch.zeros(
            (num_envs, num_agents), device=self.device, dtype=torch.bool
        )
        reward_components = torch.empty(0, device=self.device)
        return torch.empty(0, device=self.device), dones, info, reward_components


class Teacher:
    num_goal_positions = 2
    no_goal_allowed = True
    expected_hermes_commit = EXPECTED_HERMES_COMMIT
    fixed_agent_config = FIXED_AGENT_CONFIG
    fixed_env_config = FIXED_ENV_CONFIG

    def __init__(
        self,
        *,
        loader,
        checkpoint_path,
        device,
        init_steps,
        num_steps,
        compile_model,
        goal_sampling_mode,
        goal_sampling_gamma,
        goal_count_probs,
        goal_snap_keep_prob,
        goal_snap_k,
        goal_snap_max_distance,
    ):
        self.checkpoint_path = checkpoint_path
        self.loader = loader
        self.device = str(device)
        self.env_config = self._build_env_config(
            init_steps=int(init_steps),
            num_steps=int(num_steps),
            goal_sampling_mode=str(goal_sampling_mode),
            goal_sampling_gamma=float(goal_sampling_gamma),
            goal_count_probs=list(goal_count_probs),
            goal_snap_keep_prob=goal_snap_keep_prob,
            goal_snap_k=int(goal_snap_k),
            goal_snap_max_distance=float(goal_snap_max_distance),
        )
        self.env = HermesEnv(config=self.env_config, scene_data_loader=loader)
        self.env.engine.__class__ = _TargetRolloutEngine
        self.agent = self._build_agent()
        load_model_checkpoint(self.agent, self.checkpoint_path, self.device, strict=True)
        self.agent.eval()
        for parameter in self.agent.parameters():
            parameter.requires_grad_(False)
        if bool(compile_model):
            torch.set_float32_matmul_precision("high")
            self.agent.model = torch.compile(self.agent.model, dynamic=False)

    def _build_env_config(self, **runtime) -> EnvConfig:
        config = copy.deepcopy(self.fixed_env_config)
        config.update(runtime)
        config["device"] = self.device
        config["compile"] = False
        config["dataloader_config"]["batch_size"] = self.loader.batch_size
        env_config = EnvConfig.from_dict(config, allow_extra=True)
        engine_runtime = EngineRuntimeConfig(
            batch_size=self.loader.batch_size,
            max_agents=self.loader.max_agents,
            frame_time_interval=self.loader.frame_time_interval,
            dynamics_model=env_config.dynamics_model,
            device=self.device,
            num_steps=env_config.num_steps,
            enable_occupancy_grid=env_config.enable_occupancy_grid,
            occ_grid_xmin=self.loader.metadata.occ_grid_xmin,
            occ_grid_xmax=self.loader.metadata.occ_grid_xmax,
            occ_grid_ymin=self.loader.metadata.occ_grid_ymin,
            occ_grid_ymax=self.loader.metadata.occ_grid_ymax,
            occ_grid_resolution=self.loader.metadata.occ_grid_resolution,
        )
        env_config.engine_config = EngineMergedConfig.from_configs(
            env_config.engine_config, engine_runtime
        )
        return env_config

    def _build_agent(self):
        agent_config = copy.deepcopy(self.fixed_agent_config)
        agent_config["device"] = self.device
        agent_config["compile"] = False
        config = AgentConfig.from_dict(agent_config)
        engine = self.fixed_env_config["engine_config"]
        steering_rate = engine["nuplan_bicycle_max_steering_rate"]
        action_space = Box(
            low=np.asarray([engine["min_jerk_long"], -steering_rate], dtype=np.float32),
            high=np.asarray([engine["max_jerk_long"], steering_rate], dtype=np.float32),
            dtype=np.float32,
        )
        return BaseAgent.agent_factory(
            agent_name=config.agent_name,
            config=config,
            action_space=action_space,
            batch_size=self.loader.batch_size,
            action_key_to_values=None,
            frame_time_interval=self.loader.frame_time_interval,
            no_goal_allowed=self.no_goal_allowed,
            domain_randomization_config=DomainRandomizationConfig(enabled=False),
        )

    @torch.no_grad()
    def prepare_batch(self, batch, *, horizon_k, long_trajectory_additional_steps):
        scenario = self._build_replay_scenario(
            batch,
            route_points=batch.get("intent_route_points"),
            route_points_mask=batch.get("intent_route_mask"),
        )
        self.reset_scenario(scenario)
        features = self.agent.preprocess(self.env.scenario_data)
        rollout = self._build_targets(horizon_k + long_trajectory_additional_steps)
        return {
            "kinematics_features": features[2],
            "trajectory_target": rollout[:, :horizon_k],
            "trajectory_long_target": drivor_long_trajectory_target(
                rollout, horizon_k, long_trajectory_additional_steps
            ),
            "student_command": batch.get("intent_command", batch["command"]),
        }

    @torch.no_grad()
    def reset_scenario(self, scenario_full) -> None:
        scenario_live = ScenarioData.from_log(
            scenario_full, self.env.config, self.env.engine.config, time_idx=0
        )
        scenario_live.agent_control_manager.set_controlled(
            scenario_full.agent_control_manager.controlled_mask
        )
        self.env.reset(scenario_data=scenario_live, log_scenario_data=scenario_full)

        live = self.env.scenario_data
        live.goal_positions[:, :1, -1] = self.env.engine._route_goal_positions(
            live.agent_positions_all[:, :1, -1],
            live.agent_velocity_all[:, :1, -1],
            live,
        )

    def _build_targets(self, horizon_k: int):
        if horizon_k < 1:
            raise ValueError(f"horizon_k must be >= 1, got {horizon_k}.")
        num_agents = self.env.scenario_data.agent_positions_all.shape[1]
        live = self.env.scenario_data
        initial_xy = live.agent_positions_all[:, 0, -1].clone()
        initial_yaw = live.agent_orientation_all[:, 0, -1].clone()
        poses = []
        for step in range(horizon_k):
            features = self.agent.preprocess(self.env.scenario_data)
            action_params, _ = self.agent.model(*features)
            action_dim = action_params.shape[-1] // 2
            alpha = action_params[..., :action_dim]
            beta = action_params[..., action_dim:]
            drive = self.agent._rescale_action(alpha / (alpha + beta))
            self.env.step_dynamics(drive.expand(-1, num_agents, -1).float())
            live = self.env.scenario_data
            delta = live.agent_positions_all[:, 0, -1] - initial_xy
            c, s = torch.cos(initial_yaw), torch.sin(initial_yaw)
            xy = torch.stack(
                (delta[:, 0] * c + delta[:, 1] * s, -delta[:, 0] * s + delta[:, 1] * c),
                dim=-1,
            )
            yaw = live.agent_orientation_all[:, 0, -1] - initial_yaw
            yaw = torch.atan2(torch.sin(yaw), torch.cos(yaw))
            poses.append(torch.cat((xy, yaw[:, None]), dim=-1))
        return torch.stack(poses, dim=1).float()

    def _build_replay_scenario(
        self,
        batch: dict[str, Any],
        *,
        route_points=None,
        route_points_mask=None,
    ):
        required = (
            "objects_history_states",
            "objects_history_masks",
            "objects_future_states",
            "objects_future_masks",
            "lanes_centers_points",
            "lanes_centers_masks",
            "lanes_centers_attributes",
            "lanes_centers_lines",
            "lanes_centers_lines_mask",
            "other_lanes_points",
            "other_lanes_masks",
            "other_lanes_attributes",
            "route_points",
            "occupancy_grid",
        )
        missing = [field for field in required if field not in batch]
        if missing:
            raise KeyError(f"NuPlan Teacher batch is missing fields: {missing}")
        config = self.env_config.engine_config
        device = torch.device(self.device)
        history_states = batch["objects_history_states"].to(device)
        history_masks = batch["objects_history_masks"].to(device, dtype=torch.bool)
        future_states = batch["objects_future_states"].to(device)
        future_masks = batch["objects_future_masks"].to(device, dtype=torch.bool)
        states = torch.cat([history_states, future_states], dim=2)
        masks = torch.cat([history_masks, future_masks], dim=2)
        (
            positions,
            velocities,
            sizes,
            orientations,
            agent_types,
            accelerations,
            yaw_rates,
            steering_angles,
            masks,
        ) = HermesWebDataset.parse_agent_states(states, masks)
        lane_points, lane_masks = HermesWebDataset.parse_lanes_points(
            batch["other_lanes_points"].to(device, dtype=torch.float32),
            batch["other_lanes_masks"].to(device, dtype=torch.bool),
            batch["other_lanes_attributes"].to(device, dtype=torch.float32),
        )
        center_points, center_masks = HermesWebDataset.parse_lanes_points(
            batch["lanes_centers_points"].to(device, dtype=torch.float32),
            batch["lanes_centers_masks"].to(device, dtype=torch.bool),
            batch["lanes_centers_attributes"].to(device, dtype=torch.float32),
        )
        lane_points, center_points = HermesEnv._override_lane_speed_with_default(
            lane_points, lane_masks, center_points, center_masks
        )
        batch_size, max_agents, total_steps, _ = positions.shape
        init_steps = self.env_config.init_steps
        if init_steps >= total_steps - 1:
            raise ValueError(
                f"init_steps={init_steps} leaves no future in packed timeline {total_steps}."
            )
        accelerations = accelerations.clamp(config.min_a_long, config.max_a_long)
        steering_angles = steering_angles.clamp(MIN_STEERING_ANGLE, MAX_STEERING_ANGLE)
        wheelbases = get_wheelbase_from_length(sizes[..., 0])
        max_yaw_rate = (
            torch.linalg.norm(velocities, dim=-1)
            * math.tan(MAX_STEERING_ANGLE)
            / wheelbases
            * 1.05
        )
        yaw_rates = yaw_rates.clamp(min=-max_yaw_rate, max=max_yaw_rate)
        last_valid, sampled_goal = sample_valid_positions(
            positions[:, :, init_steps + 1 :],
            masks[:, :, init_steps + 1 :],
            mode="last",
            gamma=self.env_config.goal_sampling_gamma,
            orientations=orientations[:, :, init_steps + 1 :],
            offset_sigma_major=self.env_config.goal_sampling_offset_sigma_major,
            offset_sigma_minor=self.env_config.goal_sampling_offset_sigma_minor,
            lanes_centers_points=center_points,
            lanes_centers_mask=center_masks,
            goal_snap_keep_prob=self.env_config.goal_snap_keep_prob,
            goal_snap_k=self.env_config.goal_snap_k,
            goal_snap_max_distance=self.env_config.goal_snap_max_distance,
            goal_count_probs=self.env_config.goal_count_probs,
            num_goal_positions=self.num_goal_positions,
        )
        goals = sampled_goal.unsqueeze(2).repeat(1, 1, total_steps, 1)
        valid_agents = masks.any(dim=2)
        npc_ids = torch.arange(max_agents, device=device).unsqueeze(0).expand(batch_size, -1).clone()
        npc_ids.masked_fill_(~valid_agents, -1)
        control_manager = AgentControlManager(batch_size, max_agents, device=device)
        controlled_mask = torch.zeros_like(masks[:, :, init_steps])
        controlled_mask[:, 0] = masks[:, 0, init_steps]
        control_manager.set_controlled(controlled_mask)
        domain_config = copy.deepcopy(config.domain_randomization)
        domain_config.enabled = False
        randomized = sample_randomized_features(
            batch_size, max_agents, domain_config, device
        )
        scenario_route_points = (
            batch["route_points"] if route_points is None else route_points
        ).to(device, dtype=torch.float32)
        scenario_route_mask = (
            torch.any(scenario_route_points.abs() > 0, dim=-1)
            if route_points_mask is None
            else route_points_mask.to(device, dtype=torch.bool)
        )
        scenario = ScenarioData(
            npc_id_array=npc_ids,
            lanes_points=lane_points,
            lanes_points_mask=lane_masks,
            lanes_centers_points=center_points,
            lanes_centers_mask=center_masks,
            lanes_centers_lines=batch["lanes_centers_lines"].to(device, dtype=torch.float32),
            lanes_centers_lines_mask=batch["lanes_centers_lines_mask"].to(device, dtype=torch.bool),
            route_points_array=scenario_route_points,
            route_points_mask=scenario_route_mask,
            occupancy_grid=batch["occupancy_grid"].to(device, dtype=torch.uint8),
            agent_positions_all=positions,
            agent_velocity_all=velocities,
            agent_size_all=sizes,
            agent_orientation_all=orientations,
            agent_type_all=agent_types,
            npc_mask_all=masks,
            agent_acceleration_state_all=accelerations,
            agent_acceleration_control_all=accelerations.clone(),
            agent_steering_state_all=steering_angles,
            agent_steering_control_all=steering_angles.clone(),
            agent_jerk_lat_all=torch.zeros_like(steering_angles),
            agent_jerk_long_all=torch.zeros_like(steering_angles),
            agent_yaw_rate_all=yaw_rates,
            agent_goal_reached_all=torch.zeros(
                (batch_size, max_agents, total_steps), dtype=torch.bool, device=device
            ),
            goal_positions=goals,
            last_valid_positions=last_valid,
            route_progress=torch.zeros((batch_size, max_agents), device=device),
            agent_not_blind_mask=torch.ones(
                (batch_size, max_agents), dtype=torch.bool, device=device
            ),
            frame_drop_mask_all=torch.ones(
                (batch_size, max_agents, total_steps), dtype=torch.bool, device=device
            ),
            _agent_control_manager=control_manager,
            _randomized_features=randomized,
        )
        scenario.occ_surface_points_all = self._build_occ_surface_points(scenario)
        scenario.lanes_centers_tl_states = batch["lanes_centers_tl_states"].to(
            device, dtype=torch.float32
        )
        scenario.lanes_centers_tl_masks = batch["lanes_centers_tl_masks"].to(
            device, dtype=torch.bool
        )
        scenario.lanes_tl_states = batch["other_lanes_tl_states"].to(
            device, dtype=torch.float32
        )
        scenario.lanes_tl_masks = batch["other_lanes_tl_masks"].to(
            device, dtype=torch.bool
        )
        return scenario

    def _build_occ_surface_points(self, scenario):
        config = self.env_config.engine_config
        total_steps = scenario.agent_positions_all.shape[2]
        return torch.full(
            (
                scenario.agent_positions_all.shape[0],
                1,
                total_steps - self.env_config.init_steps,
                config.occ_num_rays,
                2,
            ),
            -1.0,
            device=self.device,
        )
