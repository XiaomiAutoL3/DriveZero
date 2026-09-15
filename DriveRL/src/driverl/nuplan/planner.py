"""nuPlan planner adapter for DriveRL actions."""

from __future__ import annotations

import math
from typing import Any

import torch

from driverl.nuplan.agent_loader import LoadedDriveRLAgent, load_driverl_agent_for_nuplan
from driverl.nuplan.config import (
    DriveRLNuPlanFeatureBuilderConfig,
    DriveRLNuPlanPlannerConfig,
)
from driverl.nuplan.feature_builder import DriveRLNuPlanFeatureBuilder
from driverl.nuplan.trajectory import (
    DriveRLActionTrajectory,
    build_lqr_action_matching_trajectory,
    nuplan_bicycle_action_targets,
)

try:
    from nuplan.planning.simulation.observation.observation_type import (
        DetectionsTracks,
        Observation,
    )
    from nuplan.planning.simulation.planner.abstract_planner import (
        AbstractPlanner,
        PlannerInitialization,
        PlannerInput,
    )
    from nuplan.planning.simulation.trajectory.abstract_trajectory import (
        AbstractTrajectory,
    )
    from nuplan.planning.simulation.trajectory.trajectory_sampling import (
        TrajectorySampling,
    )
except ImportError:  # pragma: no cover - lightweight local smoke fallback.
    AbstractPlanner = object  # type: ignore[misc,assignment]
    PlannerInitialization = Any  # type: ignore[misc,assignment]
    PlannerInput = Any  # type: ignore[misc,assignment]
    AbstractTrajectory = Any  # type: ignore[misc,assignment]
    Observation = Any  # type: ignore[misc,assignment]
    DetectionsTracks = object  # type: ignore[misc,assignment]
    from nuplan.planning.simulation.trajectory.trajectory_sampling import (
        TrajectorySampling,
    )


class DriveRLNuPlanPlanner(AbstractPlanner):
    """Planner that turns a DriveRL policy action into a nuPlan trajectory."""

    requires_scenario: bool = True

    def __init__(
        self,
        config: DriveRLNuPlanPlannerConfig | None = None,
        feature_builder_config: DriveRLNuPlanFeatureBuilderConfig | None = None,
        agent: Any | None = None,
        action_keys_tensor: torch.Tensor | None = None,
        feature_builder: DriveRLNuPlanFeatureBuilder | None = None,
        scenario: Any | None = None,
        **config_overrides: Any,
    ) -> None:
        self.config = config or DriveRLNuPlanPlannerConfig()
        for key, value in config_overrides.items():
            if not hasattr(self.config, key):
                raise TypeError(f"Unknown DriveRLNuPlanPlanner config key: {key}")
            setattr(self.config, key, value)
        self._device = torch.device(self.config.device)
        if (
            isinstance(self.config.tts_seed, bool)
            or not isinstance(self.config.tts_seed, int)
            or not 0 <= self.config.tts_seed < 2**63
        ):
            raise ValueError("tts_seed must be an integer in [0, 2^63-1].")
        if (
            isinstance(self.config.tts_num_candidates, bool)
            or not isinstance(self.config.tts_num_candidates, int)
            or self.config.tts_num_candidates < 1
        ):
            raise ValueError("tts_num_candidates must be a positive integer.")
        self._agent = agent
        self._action_keys_tensor = action_keys_tensor
        self._loaded_agent: LoadedDriveRLAgent | None = None
        self._feature_builder = feature_builder or DriveRLNuPlanFeatureBuilder(
            feature_builder_config
        )
        self._scenario = scenario
        for name in ("route_goal_horizon_s", "route_goal_min_speed_mps"):
            value = float(getattr(self.config, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive, got {value}.")
            setattr(self._feature_builder.config, name, value)
        goal_pair_mode = self.config.route_goal_pair_mode
        if goal_pair_mode not in {
            "legacy",
            "duplicate_earlier",
            "duplicate_later",
        }:
            raise ValueError(
                "route_goal_pair_mode must be legacy, duplicate_earlier, or "
                f"duplicate_later, got {goal_pair_mode!r}."
            )
        self._feature_builder.config.route_goal_pair_mode = goal_pair_mode
        self._initialization: PlannerInitialization | None = None
        self._trajectory_sampling = TrajectorySampling(
            num_poses=self.config.trajectory_num_poses,
            interval_length=self.config.trajectory_interval_length,
        )
        self._last_policy_time_us: int | None = None
        self._last_action: torch.Tensor | None = None
        self._last_goal_points: list[tuple[float, float]] | None = None
        self._last_model_input_debug: dict[str, Any] | None = None
        self._cached_policy_acceleration_target: float | None = None
        self._tts_selector: Any | None = None

    def name(self) -> str:
        return self.__class__.__name__

    def observation_type(self) -> type[Observation]:
        return DetectionsTracks

    def initialize(self, initialization: PlannerInitialization) -> None:
        self._initialization = initialization
        # Planner instances may be reused across scenarios.
        self._feature_builder._corrected_route_roadblock_ids = None
        if self._agent is None:
            self._loaded_agent = load_driverl_agent_for_nuplan(
                config_path=self.config.config_path,
                checkpoint_path=self.config.checkpoint_path,
                device=str(self._device),
                strict_checkpoint=self.config.strict_checkpoint,
                compile_agent=self.config.compile_agent,
            )
            self._agent = self._loaded_agent.agent
            self._action_keys_tensor = self._loaded_agent.action_keys_tensor
            self.config.min_jerk_long = self._loaded_agent.engine_config.min_jerk_long
            self.config.max_jerk_long = self._loaded_agent.engine_config.max_jerk_long
            self.config.positive_jerk_limit_when_nonnegative_acc = float(
                getattr(
                    self._loaded_agent.engine_config,
                    "positive_jerk_limit_when_nonnegative_acc",
                    1.0,
                )
            )
            self.config.nuplan_bicycle_accel_time_constant = float(
                getattr(
                    self._loaded_agent.engine_config,
                    "nuplan_bicycle_accel_time_constant",
                    0.2,
                )
            )
            self.config.nuplan_bicycle_max_acceleration = float(
                getattr(
                    self._loaded_agent.engine_config,
                    "nuplan_bicycle_max_acceleration",
                    3.0,
                )
            )
            if self._loaded_agent.env_config.dynamics_model == "nuplan_bicycle_model":
                max_steering_rate = float(
                    getattr(
                        self._loaded_agent.engine_config,
                        "nuplan_bicycle_max_steering_rate",
                        0.5,
                    )
                )
                self.config.nuplan_bicycle_max_steering_rate = max_steering_rate
                self.config.min_jerk_lat = -max_steering_rate
                self.config.max_jerk_lat = max_steering_rate
            else:
                self.config.min_jerk_lat = self._loaded_agent.engine_config.min_jerk_lat
                self.config.max_jerk_lat = self._loaded_agent.engine_config.max_jerk_lat
            self._feature_builder.config.device = str(self._device)
            self._feature_builder.config.history_steps = (
                self._loaded_agent.agent_config.history_steps
            )
            self._feature_builder.config.history_sample_interval = (
                self._loaded_agent.frame_time_interval
            )
            self._feature_builder.config.num_goal_positions = (
                self._loaded_agent.env_config.num_goal_positions
            )
            self._feature_builder.config.enable_occupancy_grid = (
                self._loaded_agent.env_config.enable_occupancy_grid
            )
            self._feature_builder.config.domain_randomization = (
                self._loaded_agent.engine_config.domain_randomization
            )
            self._feature_builder.device = self._device
            if self.config.tts_enabled:
                from driverl.nuplan.tts import NuPlanTTSSelector

                if self.config.compile_agent:
                    raise ValueError("nuPlan TTS does not support compile_agent yet.")
                self._tts_selector = NuPlanTTSSelector(
                    agent=self._agent,
                    env_config=self._loaded_agent.env_config,
                    engine_config=self._loaded_agent.engine_config,
                    frame_time_interval=self._loaded_agent.frame_time_interval,
                    max_agents=self._feature_builder.config.max_agents,
                    device=self._device,
                    num_candidates=self.config.tts_num_candidates,
                    gamma=self._loaded_agent.tts_gamma,
                    route_goal_horizon_s=self.config.route_goal_horizon_s,
                    route_goal_min_speed_mps=self.config.route_goal_min_speed_mps,
                    route_goal_pair_mode=self.config.route_goal_pair_mode,
                )
        if hasattr(self._agent, "eval"):
            self._agent.eval()

    def compute_planner_trajectory(
        self, current_input: PlannerInput
    ) -> AbstractTrajectory:
        if self._agent is None:
            raise RuntimeError(
                "DriveRLNuPlanPlanner has no agent. Call initialize() before "
                "compute_planner_trajectory()."
            )
        ego_state, _ = current_input.history.current_state
        should_forward_policy = self._should_forward_policy(ego_state.time_us)
        if should_forward_policy:
            scenario_data = self._feature_builder.build(
                current_input, self._initialization
            )
            log_scenario_data = None
            future_steps = int(getattr(getattr(self._agent, "config", None), "future_steps", 0))
            if future_steps:
                if self._scenario is None:
                    raise RuntimeError("Oracle future input requires a nuPlan scenario.")
                log_scenario_data = self._feature_builder.build_log_scenario_data(
                    scenario_data,
                    self._scenario,
                    current_input.iteration.index,
                    future_steps,
                    anchor_state=ego_state,
                )
            features = None
            with torch.no_grad():
                tts_debug = None
                if self._tts_selector is not None:
                    from driverl.nuplan.tts import (
                        TTS_HORIZON,
                        TTS_TOTAL_RETURN_SWITCH_MARGIN,
                        TTS_USE_BF16,
                        derive_tts_sample_seed,
                    )

                    if log_scenario_data is not None:
                        raise ValueError(
                            "nuPlan TTS does not support oracle future input."
                        )
                    scenario_token = getattr(self._scenario, "token", "")
                    if not isinstance(scenario_token, (str, bytes)):
                        raise TypeError("nuPlan scenario token must be str or bytes.")
                    sample_seed = derive_tts_sample_seed(
                        self.config.tts_seed,
                        scenario_token,
                        ego_state.time_us,
                    )
                    tts_output = self._tts_selector.select_action(
                        scenario_data, sample_seed=sample_seed
                    )
                    action = tts_output.selected_action
                    tts_debug = {
                        "candidate_sources": self._tts_selector.candidate_sources,
                        "candidate_actions": tts_output.candidate_actions.detach()
                        .cpu()
                        .clone(),
                        "candidate_values": tts_output.candidate_values.detach()
                        .cpu()
                        .clone(),
                        "candidate_scores": tts_output.candidate_scores.detach()
                        .cpu()
                        .clone(),
                        "selected_candidate": int(tts_output.selected_candidate.item()),
                        "num_candidates": self._tts_selector.num_candidates,
                        "horizon": TTS_HORIZON,
                        "use_bf16": TTS_USE_BF16,
                        "total_return_switch_margin": TTS_TOTAL_RETURN_SWITCH_MARGIN,
                        "candidate_total_rewards": tts_output.candidate_total_rewards.detach()
                        .cpu()
                        .clone(),
                        "candidate_dones": tts_output.candidate_dones.detach()
                        .cpu()
                        .clone(),
                        "candidate_terminal_steps": tts_output.candidate_terminal_steps.detach()
                        .cpu()
                        .clone(),
                        "candidate_leaf_total_values": tts_output.candidate_leaf_total_values.detach()
                        .cpu()
                        .clone(),
                        "candidate_bootstrap_mask": tts_output.candidate_bootstrap_mask.detach()
                        .cpu()
                        .clone(),
                        "candidate_valid": tts_output.candidate_valid.detach()
                        .cpu()
                        .clone(),
                        "gamma": self._tts_selector.gamma,
                        "tts_seed": self.config.tts_seed,
                        "sample_seed": tts_output.sample_seed,
                    }
                    output = action
                elif hasattr(self._agent, "preprocess") and hasattr(
                    self._agent, "forward_with_features"
                ):
                    features = self._agent.preprocess(
                        scenario_data, log_scenario_data=log_scenario_data
                    )
                    current_speed = torch.linalg.norm(
                        scenario_data.agent_velocity_all[:, :, -1, :], dim=-1
                    )
                    output = self._agent.forward_with_features(
                        features,
                        sampling_method=self.config.sampling_method,
                        action_keys_tensor=self._action_keys_tensor,
                        current_speed=current_speed,
                    )
                else:
                    output = self._agent(
                        scenario_data,
                        log_scenario_data=log_scenario_data,
                        sampling_method=self.config.sampling_method,
                        action_keys_tensor=self._action_keys_tensor,
                    )
            action = _extract_action(output).detach()
            self._last_goal_points = _extract_ego_goal_points(scenario_data)
            model_input_debug = _extract_model_input_debug(
                self._agent, features, ego_state
            )
            self._last_model_input_debug = model_input_debug
            if tts_debug is not None:
                if model_input_debug is None:
                    model_input_debug = {}
                model_input_debug["test_time_scaling"] = tts_debug
                self._last_model_input_debug = model_input_debug
            self._last_action = action
            self._last_policy_time_us = ego_state.time_us
        elif self._last_action is not None:
            action = self._last_action
            model_input_debug = self._last_model_input_debug
        else:
            raise RuntimeError(
                "DriveRL policy action cache is empty before first forward."
            )
        jerk_long, lat_command = self._decode_action(action)
        reference_trajectory = None
        acceleration_control = None
        steering_control = None
        action_matching_debug = None
        if self.config.trajectory_mode == "action_matching_lqr":
            (
                target_acceleration,
                target_steering_rate,
                acceleration_control,
                next_cached,
            ) = nuplan_bicycle_action_targets(
                ego_state=ego_state,
                jerk_long=jerk_long,
                lat_command=lat_command,
                frame_dt=self.config.trajectory_interval_length,
                min_jerk_long=self.config.min_jerk_long,
                max_jerk_long=self.config.max_jerk_long,
                max_acceleration=self.config.nuplan_bicycle_max_acceleration,
                max_steering_rate=self.config.nuplan_bicycle_max_steering_rate,
                positive_jerk_limit_when_nonnegative_acc=(
                    self.config.positive_jerk_limit_when_nonnegative_acc
                ),
                accel_time_constant=self.config.nuplan_bicycle_accel_time_constant,
                policy_interval_s=self.config.policy_interval_s,
                policy_forward=should_forward_policy,
                cached_acceleration_target=self._cached_policy_acceleration_target,
            )
            self._cached_policy_acceleration_target = next_cached
            reference_trajectory, action_matching_debug = (
                build_lqr_action_matching_trajectory(
                    ego_state=ego_state,
                    target_acceleration=target_acceleration,
                    target_steering_rate=target_steering_rate,
                    trajectory_sampling=self._trajectory_sampling,
                    heading_span=self.config.lqr_action_match_heading_span,
                    lateral_span=self.config.lqr_action_match_lateral_span,
                    heading_samples=self.config.lqr_action_match_heading_samples,
                    lateral_samples=self.config.lqr_action_match_lateral_samples,
                    discretization_time=self.config.lqr_discretization_time,
                    tracking_horizon=self.config.lqr_tracking_horizon,
                )
            )
            steering_control = target_steering_rate
        elif self.config.trajectory_mode != "action":
            raise ValueError(f"Unknown trajectory_mode={self.config.trajectory_mode!r}")
        return DriveRLActionTrajectory(
            ego_state=ego_state,
            jerk_long=jerk_long,
            lat_command=lat_command,
            raw_action=action.detach().cpu().clone(),
            acceleration_control=acceleration_control,
            steering_control=steering_control,
            goal_points=self._last_goal_points,
            trajectory_sampling=self._trajectory_sampling,
            reference_trajectory=reference_trajectory,
            debug_info={
                "sampling_method": self.config.sampling_method,
                "trajectory_mode": self.config.trajectory_mode,
                "policy_interval_s": self.config.policy_interval_s,
                "policy_forward": should_forward_policy,
                "engine_action_config": {
                    "min_jerk_long": self.config.min_jerk_long,
                    "max_jerk_long": self.config.max_jerk_long,
                    "positive_jerk_limit_when_nonnegative_acc": (
                        self.config.positive_jerk_limit_when_nonnegative_acc
                    ),
                    "nuplan_bicycle_accel_time_constant": (
                        self.config.nuplan_bicycle_accel_time_constant
                    ),
                    "nuplan_bicycle_max_acceleration": (
                        self.config.nuplan_bicycle_max_acceleration
                    ),
                    "nuplan_bicycle_max_steering_rate": (
                        self.config.nuplan_bicycle_max_steering_rate
                    ),
                },
                "goal_points": self._last_goal_points,
                "action_matching": action_matching_debug,
                "model_input": model_input_debug,
                "temporary_limitations": self._feature_builder.temporary_limitations,
                "checkpoint_update": (
                    None
                    if self._loaded_agent is None
                    else self._loaded_agent.checkpoint_update
                ),
            },
        )

    def _should_forward_policy(self, current_time_us: int) -> bool:
        if self._last_action is None or self._last_policy_time_us is None:
            return True
        policy_interval_us = int(round(max(self.config.policy_interval_s, 0.0) * 1e6))
        if policy_interval_us <= 0:
            return True
        return current_time_us - self._last_policy_time_us >= policy_interval_us

    def _decode_action(self, action: torch.Tensor) -> tuple[float, float]:
        action = action.detach().to(self._device)
        if action.is_floating_point() and action.shape[-1] == 2:
            action_values = action.reshape(-1, 2)[0]
        else:
            if self._action_keys_tensor is None:
                raise ValueError(
                    "Discrete DriveRL action requires action_keys_tensor for decoding."
                )
            action_idx = action.reshape(-1)[0].long()
            action_values = self._action_keys_tensor.to(self._device)[action_idx]
        jerk_long = float(
            action_values[0]
            .clamp(self.config.min_jerk_long, self.config.max_jerk_long)
            .item()
        )
        lat_command = float(
            action_values[1]
            .clamp(self.config.min_jerk_lat, self.config.max_jerk_lat)
            .item()
        )
        return jerk_long, lat_command


def _extract_action(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, tuple) and output:
        action = output[0]
        if isinstance(action, torch.Tensor):
            return action
    raise TypeError(
        "DriveRL agent must return a Tensor action or a tuple whose first item "
        "is a Tensor."
    )


def _extract_ego_goal_points(scenario_data: Any) -> list[tuple[float, float]]:
    goal_positions = getattr(scenario_data, "goal_positions", None)
    if goal_positions is None or goal_positions.numel() == 0:
        return []
    goals = goal_positions[0, 0, -1].detach().float().cpu().reshape(-1, 2)
    return [(float(x), float(y)) for x, y in goals.tolist()]


def _extract_model_input_debug(
    agent: Any, features: Any, ego_state: Any
) -> dict[str, Any] | None:
    if features is None:
        return None
    if not hasattr(agent, "features_to_vis_inputs"):
        return None
    try:
        model_inputs, _ = agent.features_to_vis_inputs(features)
        lane_features = model_inputs["lane_features"][0].detach().float().cpu()
        lane_mask = model_inputs["lane_mask"][0].detach().bool().cpu()
        raw_lane_types = model_inputs.get("lane_types_are_raw", False)
        if isinstance(raw_lane_types, torch.Tensor):
            raw_lane_types = bool(raw_lane_types.reshape(-1)[0].item())
        else:
            raw_lane_types = bool(raw_lane_types)
        vehicle_features = (
            model_inputs["vehicle_state_features"][0].detach().float().cpu()
        )
        other_features = model_inputs["other_agents_features"][0].detach().float().cpu()
        other_mask = model_inputs["other_agents_mask"][0].detach().bool().cpu()
    except Exception:
        return None

    lanes = []
    if lane_features.ndim == 2 and lane_features.shape[-1] >= 4:
        for lane, valid in zip(lane_features, lane_mask):
            if not bool(valid):
                continue
            lane_type = int(lane[6].item()) if lane.shape[-1] > 6 else 0
            lanes.append(
                {
                    "points": [
                        _local_to_global_xy(ego_state, lane[0].item(), lane[1].item()),
                        _local_to_global_xy(ego_state, lane[2].item(), lane[3].item()),
                    ],
                    "lane_type": lane_type,
                    "is_curb": lane_type == (0 if raw_lane_types else 1),
                }
            )

    other_agents = []
    for agent_features, valid in zip(other_features, other_mask):
        if not bool(valid):
            continue
        other_agents.append(_agent_feature_to_debug(ego_state, agent_features))

    return {
        "frame": "global",
        "ego": _agent_feature_to_debug(ego_state, vehicle_features, is_ego=True),
        "road_graph": {
            "lanes": lanes,
            "lane_types_are_raw": raw_lane_types,
        },
        "other_agents": other_agents,
    }


def _agent_feature_to_debug(
    ego_state: Any, feature: torch.Tensor, is_ego: bool = False
) -> dict[str, Any]:
    local_position = feature[0:2].tolist()
    polygon = feature[7:15].reshape(4, 2) if feature.shape[-1] >= 15 else None
    if polygon is None or not bool(torch.any(polygon)):
        polygon = _box_polygon(
            x=float(feature[0]),
            y=float(feature[1]),
            heading=float(feature[4]),
            length=float(feature[5]),
            width=float(feature[6]),
            dtype=feature.dtype,
        )
    if is_ego:
        polygon = _shift_polygon_longitudinally(
            polygon,
            heading=float(feature[4]),
            distance=float(ego_state.car_footprint.rear_axle_to_center_dist),
        )
    polygon_points = [
        _local_to_global_xy(ego_state, float(point[0]), float(point[1]))
        for point in polygon
    ]
    if polygon_points:
        polygon_points.append(polygon_points[0])
    return {
        "position": _local_to_global_xy(
            ego_state, float(local_position[0]), float(local_position[1])
        ),
        "orientation": _wrap_angle(float(feature[4]) + ego_state.rear_axle.heading),
        "size": feature[5:7].tolist(),
        "velocity": feature[2:4].tolist(),
        "polygon": polygon_points,
    }


def _box_polygon(
    *,
    x: float,
    y: float,
    heading: float,
    length: float,
    width: float,
    dtype: torch.dtype,
) -> torch.Tensor:
    half_l = length * 0.5
    half_w = width * 0.5
    corners = torch.tensor(
        [[half_l, half_w], [half_l, -half_w], [-half_l, -half_w], [-half_l, half_w]],
        dtype=dtype,
    )
    cos_h = math.cos(heading)
    sin_h = math.sin(heading)
    rot = torch.tensor([[cos_h, -sin_h], [sin_h, cos_h]], dtype=dtype)
    return corners @ rot.T + torch.tensor([x, y], dtype=dtype)


def _shift_polygon_longitudinally(
    polygon: torch.Tensor, *, heading: float, distance: float
) -> torch.Tensor:
    shifted = polygon.clone()
    shifted[:, 0] += distance * math.cos(heading)
    shifted[:, 1] += distance * math.sin(heading)
    return shifted


def _local_to_global_xy(ego_state: Any, x_local: float, y_local: float) -> list[float]:
    anchor = ego_state.rear_axle
    cos_h = math.cos(anchor.heading)
    sin_h = math.sin(anchor.heading)
    return [
        anchor.x + cos_h * x_local - sin_h * y_local,
        anchor.y + sin_h * x_local + cos_h * y_local,
    ]


def _wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi
