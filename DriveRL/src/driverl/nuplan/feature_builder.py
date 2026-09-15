"""Runtime nuPlan PlannerInput to DriveRL ScenarioData conversion."""

from __future__ import annotations

import copy
import math
from typing import Any

import numpy as np
import torch

from driverl.datatypes.data_enums import AgentControlManager
from driverl.datatypes.goal_position_utils import (
    apply_goal_pair_mode,
    route_goal_positions,
)
from driverl.datatypes.scenario_data import ScenarioData
from driverl.env import constants
from driverl.env.dataloader.lane_stitching import (
    INVALID_GROUP_ID,
    stitch_lane_segments,
)
from driverl.env.domain_randomization.config import DomainRandomizationConfig
from driverl.env.domain_randomization.domain_randomization import (
    sample_randomized_features,
)
from driverl.nuplan.config import DriveRLNuPlanFeatureBuilderConfig
from driverl.nuplan.route_correction import correct_route_roadblock_ids
from driverl.nuplan.state_conversion import (
    DRIVERL_EGO_TYPE,
    global_heading_to_anchor,
    global_to_anchor_xy,
    local_vector_to_anchor_xy,
    vector_to_anchor_xy,
)

try:
    from nuplan.common.actor_state.ego_state import EgoState
    from nuplan.planning.script.driverl_runtime_map_features import (
        DEFAULT_MAP_RADIUS_M,
        LANE_TYPE_LANE_CENTER_LINE,
        FrameRow,
        _build_map_arrays,
        _empty_map_arrays,
        _sample_route_points_from_map,
    )
except ImportError as exc:  # pragma: no cover - import guard for non-nuPlan envs.
    raise ImportError(
        "driverl.nuplan.feature_builder requires nuPlan to be installed."
    ) from exc


class DriveRLNuPlanFeatureBuilder:
    """Build DriveRL ``ScenarioData`` from nuPlan official planner inputs."""

    temporary_limitations = (
        "runtime map query uses route-centered anchors at fixed spacing rather "
        "than offline WebDataset expert future ego poses; "
        "traffic-light state is folded into the 10D lane feature expected by "
        "the current VanillaNet checkpoint rather than passed as a separate "
        "model input."
    )

    def __init__(
        self,
        config: DriveRLNuPlanFeatureBuilderConfig | None = None,
        **config_overrides: Any,
    ) -> None:
        self.config = config or DriveRLNuPlanFeatureBuilderConfig()
        for key, value in config_overrides.items():
            if not hasattr(self.config, key):
                raise TypeError(f"Unknown DriveRLNuPlanFeatureBuilder config key: {key}")
            setattr(self.config, key, value)
        self.device = torch.device(self.config.device)
        self._corrected_route_roadblock_ids: list[str] | None = None

    def build(
        self, current_input: Any, initialization: Any | None = None
    ) -> ScenarioData:
        """Build a batch-size-1 ``ScenarioData`` from a nuPlan planner input."""
        target_device = self.device
        build_device = torch.device("cpu")
        self.device = build_device
        try:
            scenario_data = self._build_on_current_device(current_input, initialization)
        finally:
            self.device = target_device
        scenario_data = scenario_data.make_contiguous()
        if target_device.type != "cpu":
            _move_scenario_data_to_device(scenario_data, target_device)
        return scenario_data

    def _build_on_current_device(
        self, current_input: Any, initialization: Any | None = None
    ) -> ScenarioData:
        history = current_input.history
        ego_states: list[EgoState] = list(history.ego_states)
        observations = list(history.observations)
        if not ego_states:
            raise ValueError("PlannerInput history has no ego states.")

        anchor_state = ego_states[-1]
        history_steps = self.config.history_steps
        max_agents = self.config.max_agents
        selected_ego_states, selected_observations, selected_valid = (
            _sample_history_by_time(
                ego_states,
                observations,
                history_steps,
                self.config.history_sample_interval,
            )
        )
        current_agents = _observation_agents(selected_observations[-1])
        tracked_slots = self._select_current_agents(
            current_agents,
            selected_observations,
            selected_valid,
            anchor_state,
        )
        slot_keys = [_EGO_KEY] + [_track_key(agent) for agent in tracked_slots]
        self._last_slot_keys = slot_keys

        positions = np.zeros((1, max_agents, history_steps, 2), dtype=np.float32)
        velocities = np.zeros_like(positions)
        sizes = np.zeros_like(positions)
        orientations = np.zeros((1, max_agents, history_steps), dtype=np.float32)
        agent_types = np.zeros_like(orientations)
        masks = np.zeros((1, max_agents, history_steps), dtype=np.bool_)
        accelerations = np.zeros_like(orientations)
        yaw_rates = np.zeros_like(orientations)
        steering = np.zeros_like(orientations)

        for time_idx, (ego_state, observation, is_valid) in enumerate(
            zip(selected_ego_states, selected_observations, selected_valid)
        ):
            if not is_valid:
                continue
            self._write_ego_slot(
                slot=0,
                time_idx=time_idx,
                ego_state=ego_state,
                anchor_state=anchor_state,
                positions=positions,
                velocities=velocities,
                sizes=sizes,
                orientations=orientations,
                agent_types=agent_types,
                masks=masks,
                accelerations=accelerations,
                yaw_rates=yaw_rates,
                steering=steering,
            )
            agents_by_key = {
                _track_key(agent): agent for agent in _observation_agents(observation)
            }
            for slot, key in enumerate(slot_keys[1:], start=1):
                agent = agents_by_key.get(key)
                if agent is None:
                    continue
                self._write_agent_slot(
                    slot=slot,
                    time_idx=time_idx,
                    agent=agent,
                    anchor_state=anchor_state,
                    positions=positions,
                    velocities=velocities,
                    sizes=sizes,
                    orientations=orientations,
                    agent_types=agent_types,
                    masks=masks,
                    yaw_rates=yaw_rates,
                )

        self._fill_agent_derivatives_from_history(
            velocities=velocities,
            orientations=orientations,
            masks=masks,
            accelerations=accelerations,
            yaw_rates=yaw_rates,
            timestamps_s=np.asarray(
                [
                    (state.time_us - anchor_state.time_us) * 1e-6
                    for state in selected_ego_states
                ],
                dtype=np.float32,
            ),
            ego_slot_count=1,
        )
        positions = torch.as_tensor(positions, dtype=torch.float32, device=self.device)
        velocities = torch.as_tensor(
            velocities, dtype=torch.float32, device=self.device
        )
        sizes = torch.as_tensor(sizes, dtype=torch.float32, device=self.device)
        orientations = torch.as_tensor(
            orientations, dtype=torch.float32, device=self.device
        )
        agent_types = torch.as_tensor(
            agent_types, dtype=torch.float32, device=self.device
        )
        masks = torch.as_tensor(masks, dtype=torch.bool, device=self.device)
        accelerations = torch.as_tensor(
            accelerations, dtype=torch.float32, device=self.device
        )
        yaw_rates = torch.as_tensor(yaw_rates, dtype=torch.float32, device=self.device)
        steering = torch.as_tensor(steering, dtype=torch.float32, device=self.device)
        map_features = self._build_map_features(
            anchor_state=anchor_state,
            initialization=initialization,
            current_input=current_input,
        )
        goal_positions = self._build_goal_positions(
            current_position=positions[:, :1, -1, :],
            current_velocity=velocities[:, :1, -1, :],
            route_points=map_features["route_points_array"],
            route_mask=map_features["route_points_mask"],
            max_agents=max_agents,
            history_steps=history_steps,
        )
        frame_valid_mask = torch.as_tensor(
            selected_valid,
            dtype=torch.bool,
            device=self.device,
        ).view(1, 1, history_steps)
        frame_valid_mask = frame_valid_mask.expand(1, max_agents, history_steps)

        scenario_data = ScenarioData(
            npc_id_array=self._npc_id_array(masks),
            lanes_points=map_features["lanes_points"],
            lanes_points_mask=map_features["lanes_points_mask"],
            lanes_centers_points=map_features["lanes_centers_points"],
            lanes_centers_mask=map_features["lanes_centers_mask"],
            lanes_centers_lines=map_features["lanes_centers_lines"],
            lanes_centers_lines_mask=map_features["lanes_centers_lines_mask"],
            route_points_array=map_features["route_points_array"],
            route_points_mask=map_features["route_points_mask"],
            occupancy_grid=(
                torch.zeros((1, 0, 0), dtype=torch.uint8, device=self.device)
                if self.config.enable_occupancy_grid
                else torch.empty(0, dtype=torch.uint8, device=self.device)
            ),
            agent_positions_all=positions,
            agent_velocity_all=velocities,
            agent_size_all=sizes,
            agent_orientation_all=orientations,
            agent_type_all=agent_types,
            npc_mask_all=masks,
            agent_acceleration_state_all=accelerations,
            agent_acceleration_control_all=accelerations.clone(),
            agent_yaw_rate_all=yaw_rates,
            agent_steering_state_all=steering,
            agent_steering_control_all=steering.clone(),
            agent_jerk_lat_all=torch.zeros_like(steering),
            agent_jerk_long_all=torch.zeros_like(steering),
            goal_positions=goal_positions,
            agent_goal_reached_all=torch.zeros_like(masks, dtype=torch.bool),
            frame_drop_mask_all=frame_valid_mask,
            agent_not_blind_mask=masks.any(dim=2),
            occ_surface_points_all=torch.full(
                (1, 1, 1, 512, 2),
                -1.0,
                dtype=torch.float32,
                device=self.device,
            ),
            _randomized_features=sample_randomized_features(
                1,
                max_agents,
                self.config.domain_randomization
                or DomainRandomizationConfig.default_preset(),
                self.device,
            ),
        )
        scenario_data._agent_control_manager = AgentControlManager(
            1, max_agents, self.device
        )
        controlled_mask = torch.zeros(
            (1, max_agents), dtype=torch.bool, device=self.device
        )
        controlled_mask[:, 0] = True
        scenario_data._agent_control_manager.set_controlled(controlled_mask)
        return scenario_data

    def build_log_scenario_data(
        self,
        scenario_data: ScenarioData,
        scenario: Any,
        iteration: int,
        future_steps: int,
        anchor_state: EgoState | None = None,
    ) -> ScenarioData:
        """Append nuPlan ground-truth future frames for oracle evaluation."""
        if future_steps <= 0:
            return scenario_data
        interval = float(self.config.history_sample_interval)
        if anchor_state is None:
            anchor_state = scenario.get_ego_state_at_iteration(iteration)
        ego_future = list(
            scenario.get_ego_future_trajectory(
                iteration, future_steps * interval, future_steps
            )
        )[:future_steps]
        future_observations = list(
            scenario.get_future_tracked_objects(
                iteration, future_steps * interval, future_steps
            )
        )[:future_steps]
        max_agents = scenario_data.agent_positions_all.shape[1]
        future_count = min(len(ego_future), len(future_observations))
        shape = (1, max_agents, future_count)
        positions = np.zeros((*shape, 2), dtype=np.float32)
        velocities = np.zeros_like(positions)
        sizes = np.zeros_like(positions)
        orientations = np.zeros(shape, dtype=np.float32)
        types = np.zeros(shape, dtype=np.float32)
        masks = np.zeros(shape, dtype=np.bool_)
        yaw_rates = np.zeros(shape, dtype=np.float32)
        slot_keys = list(getattr(self, "_last_slot_keys", [_EGO_KEY]))
        slot_by_key = {key: idx for idx, key in enumerate(slot_keys)}
        free_slots = iter(
            idx for idx in range(max_agents) if idx not in slot_by_key.values()
        )
        for time_idx in range(future_count):
            self._write_ego_slot(
                slot=0,
                time_idx=time_idx,
                ego_state=ego_future[time_idx],
                anchor_state=anchor_state,
                positions=positions,
                velocities=velocities,
                sizes=sizes,
                orientations=orientations,
                agent_types=types,
                masks=masks,
                accelerations=np.zeros(shape, dtype=np.float32),
                yaw_rates=yaw_rates,
                steering=np.zeros(shape, dtype=np.float32),
            )
            for agent in _observation_agents(future_observations[time_idx]):
                key = _track_key(agent)
                if key not in slot_by_key:
                    try:
                        slot_by_key[key] = next(free_slots)
                    except StopIteration:
                        continue
                self._write_agent_slot(
                    slot=slot_by_key[key],
                    time_idx=time_idx,
                    agent=agent,
                    anchor_state=anchor_state,
                    positions=positions,
                    velocities=velocities,
                    sizes=sizes,
                    orientations=orientations,
                    agent_types=types,
                    masks=masks,
                    yaw_rates=yaw_rates,
                )
        future_tensors = {
            "agent_positions_all": positions,
            "agent_velocity_all": velocities,
            "agent_size_all": sizes,
            "agent_orientation_all": orientations,
            "agent_type_all": types,
            "npc_mask_all": masks,
            "agent_yaw_rate_all": yaw_rates,
        }
        log_data = copy.deepcopy(scenario_data)
        for name, value in future_tensors.items():
            current = getattr(log_data, name).detach().cpu().numpy()
            combined = np.concatenate([current, value], axis=2)
            setattr(log_data, name, torch.as_tensor(combined, device=self.device))
        for name in (
            "agent_acceleration_state_all",
            "agent_acceleration_control_all",
            "agent_steering_state_all",
            "agent_steering_control_all",
        ):
            current = getattr(log_data, name)
            zeros = torch.zeros((*current.shape[:2], future_count), device=current.device)
            setattr(log_data, name, torch.cat([current, zeros], dim=2))
        log_data.frame_drop_mask_all = torch.ones_like(log_data.npc_mask_all)
        return log_data.make_contiguous()

    def _build_map_features(
        self,
        *,
        anchor_state: EgoState,
        initialization: Any | None,
        current_input: Any,
    ) -> dict[str, torch.Tensor]:
        if initialization is None:
            arrays = _empty_map_arrays(
                1,
                max_lanes_centers=self.config.max_lanes_centers,
                max_lanes_other=self.config.max_lanes_other,
            )
            route_points = np.zeros((self.config.max_route_points, 2), dtype=np.float32)
        else:
            anchor_frame = _frame_from_ego_state(anchor_state)
            traffic_lights = _traffic_lights_by_current_token(
                anchor_frame,
                getattr(current_input, "traffic_light_data", None),
            )
            map_api = getattr(initialization, "map_api", None)
            route_roadblock_ids = [
                str(item) for item in getattr(initialization, "route_roadblock_ids", [])
            ]
            # Public DriveRL inference always consumes corrected route IDs.
            if self._corrected_route_roadblock_ids is None:
                self._corrected_route_roadblock_ids = correct_route_roadblock_ids(
                    anchor_state, map_api, route_roadblock_ids
                )
            route_roadblock_ids = self._corrected_route_roadblock_ids
            route_points = _sample_route_points_from_map(
                map_api=map_api,
                route_roadblock_ids=route_roadblock_ids,
                anchor=anchor_frame,
            )
            map_query_frames = (
                [anchor_frame]
                if self.config.use_nuplan_current_ego_map_query
                else _route_map_query_frames(
                    anchor_frame=anchor_frame,
                    route_points=route_points,
                    num_points=self.config.route_map_query_points,
                    spacing_m=self.config.route_map_query_spacing_m,
                    max_distance_m=self.config.route_map_query_distance_m,
                )
            )
            if not map_query_frames:
                map_query_frames = [anchor_frame]
            arrays, _stats = _build_map_arrays(
                map_api=map_api,
                anchor=anchor_frame,
                total_frames=1,
                sampled_frames=[anchor_frame],
                map_query_frames=map_query_frames,
                tl_by_token=traffic_lights,
                route_roadblock_ids=route_roadblock_ids,
                radius_m=self.config.map_radius_m or DEFAULT_MAP_RADIUS_M,
                max_center_segments=self.config.max_lanes_centers,
                max_boundary_segments=self.config.max_lanes_other,
            )
            if route_points is None:
                route_points = self._route_points_from_current_map_arrays(arrays)

        lanes_points, lanes_points_mask = self._parse_lanes_points(
            arrays["other_lanes_points"],
            arrays["other_lanes_masks"],
            arrays["other_lanes_attributes"],
        )
        lanes_centers_points, lanes_centers_mask = self._parse_lanes_points(
            arrays["lanes_centers_points"],
            arrays["lanes_centers_masks"],
            arrays["lanes_centers_attributes"],
        )
        lanes_points, lanes_centers_points = self._override_lane_speed_with_default(
            lanes_points,
            lanes_points_mask,
            lanes_centers_points,
            lanes_centers_mask,
        )
        lanes_points = self._append_traffic_light_features(
            lanes_points,
            arrays.get("other_lanes_tl_states"),
        )
        lines, lines_mask = stitch_lane_segments(
            arrays["lanes_centers_points"],
            arrays["lanes_centers_masks"].astype(np.bool_),
            arrays.get(
                "lanes_centers_groups",
                np.full(
                    arrays["lanes_centers_points"].shape[0],
                    INVALID_GROUP_ID,
                    dtype=np.uint32,
                ),
            ),
            arrays.get("lanes_centers_next_groups"),
        )
        route_points = _fit_route_points(route_points, self.config.max_route_points)
        route_mask = np.any(np.abs(route_points) > 0.0, axis=-1)
        return {
            "lanes_points": torch.as_tensor(
                lanes_points[None], dtype=torch.float32, device=self.device
            ),
            "lanes_points_mask": torch.as_tensor(
                lanes_points_mask[None], dtype=torch.bool, device=self.device
            ),
            "lanes_centers_points": torch.as_tensor(
                lanes_centers_points[None], dtype=torch.float32, device=self.device
            ),
            "lanes_centers_mask": torch.as_tensor(
                lanes_centers_mask[None], dtype=torch.bool, device=self.device
            ),
            "lanes_centers_lines": torch.as_tensor(
                lines[None], dtype=torch.float32, device=self.device
            ),
            "lanes_centers_lines_mask": torch.as_tensor(
                lines_mask[None], dtype=torch.bool, device=self.device
            ),
            "route_points_array": torch.as_tensor(
                route_points[None], dtype=torch.float32, device=self.device
            ),
            "route_points_mask": torch.as_tensor(
                route_mask[None], dtype=torch.bool, device=self.device
            ),
        }

    def _route_points_from_current_map_arrays(
        self, arrays: dict[str, np.ndarray]
    ) -> np.ndarray:
        route = np.zeros((self.config.max_route_points, 2), dtype=np.float32)
        centers = arrays.get("lanes_centers_points")
        masks = arrays.get("lanes_centers_masks")
        attributes = arrays.get("lanes_centers_attributes")
        if centers is None or masks is None or attributes is None:
            return route
        route_mask = masks & (attributes[:, 0] == LANE_TYPE_LANE_CENTER_LINE)
        points = centers[route_mask].reshape(-1, 2)
        if len(points) == 0:
            return route
        if len(points) <= self.config.max_route_points:
            route[: len(points)] = points
        else:
            idx = (
                np.linspace(0, len(points) - 1, self.config.max_route_points)
                .round()
                .astype(np.int64)
            )
            route[:] = points[idx]
        return route

    def _build_goal_positions(
        self,
        *,
        current_position: torch.Tensor,
        current_velocity: torch.Tensor,
        route_points: torch.Tensor,
        route_mask: torch.Tensor,
        max_agents: int,
        history_steps: int,
    ) -> torch.Tensor:
        goal = torch.zeros(
            (1, max_agents, history_steps, self.config.num_goal_positions * 2),
            dtype=torch.float32,
            device=self.device,
        )
        local_goal = _route_goal_positions(
            current_position,
            current_velocity,
            route_points,
            route_mask,
            horizon_s=self.config.route_goal_horizon_s,
            min_speed_mps=self.config.route_goal_min_speed_mps,
            num_goal_positions=self.config.num_goal_positions,
        )
        local_goal = apply_goal_pair_mode(local_goal, self.config.route_goal_pair_mode)
        for goal_idx in range(self.config.num_goal_positions):
            goal[:, 0, :, goal_idx * 2 : goal_idx * 2 + 2] = local_goal[
                :, 0, goal_idx * 2 : goal_idx * 2 + 2
            ].unsqueeze(1)
        return goal

    @staticmethod
    def _parse_lanes_points(
        points: np.ndarray,
        masks: np.ndarray,
        attributes: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        lane_type = np.broadcast_to(
            attributes[:, 0:1, None],
            (points.shape[0], points.shape[1], 1),
        )
        speed_raw = attributes[:, 1]
        lane_speed = np.clip(speed_raw, 0.0, constants.MAX_SPEED_LIMIT).astype(
            np.float32
        )
        lane_speed = np.where(masks.astype(bool), lane_speed, 0.0)
        lane_speed = np.broadcast_to(
            lane_speed[:, None, None], (points.shape[0], points.shape[1], 1)
        )
        return np.concatenate(
            [points.astype(np.float32), lane_type.astype(np.float32), lane_speed],
            axis=-1,
        ), masks.astype(np.bool_)

    @staticmethod
    def _override_lane_speed_with_default(
        lanes_points: np.ndarray,
        lanes_points_mask: np.ndarray,
        lanes_centers_points: np.ndarray,
        lanes_centers_mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        lanes_points = lanes_points.copy()
        lanes_centers_points = lanes_centers_points.copy()
        lanes_points[..., 3] = np.where(
            lanes_points_mask[..., None] & (lanes_points[..., 3] <= 0.0),
            constants.DEFAULT_SPEED_LIMIT,
            lanes_points[..., 3],
        )
        lanes_centers_points[..., 3] = np.where(
            lanes_centers_mask[..., None] & (lanes_centers_points[..., 3] <= 0.0),
            constants.DEFAULT_SPEED_LIMIT,
            lanes_centers_points[..., 3],
        )
        return lanes_points, lanes_centers_points

    @staticmethod
    def _append_traffic_light_features(
        lanes_points: np.ndarray,
        tl_states: np.ndarray | None,
    ) -> np.ndarray:
        if tl_states is None:
            tl = np.zeros(
                (lanes_points.shape[0], lanes_points.shape[1], 4), dtype=np.float32
            )
            tl[..., 0] = 1.0
        else:
            tl_current = np.asarray(tl_states[:, -1, :], dtype=np.float32)
            tl = np.repeat(tl_current[:, None, :], lanes_points.shape[1], axis=1)
        return np.concatenate([lanes_points, tl], axis=-1)

    def _fill_agent_derivatives_from_history(
        self,
        *,
        velocities: np.ndarray,
        orientations: np.ndarray,
        masks: np.ndarray,
        accelerations: np.ndarray,
        yaw_rates: np.ndarray,
        timestamps_s: np.ndarray,
        ego_slot_count: int,
    ) -> None:
        time_steps = velocities.shape[2]
        if time_steps < 2:
            return
        agent_slice = slice(ego_slot_count, velocities.shape[1])
        valid = masks[0, agent_slice]
        if not bool(valid.any()):
            return

        slot_velocities = velocities[0, agent_slice]
        slot_orientations = orientations[0, agent_slice]
        slot_accelerations = accelerations[0, agent_slice]
        slot_yaw_rates = yaw_rates[0, agent_slice]

        for t in range(time_steps):
            valid_t = valid[:, t]
            has_prev = t > 0 and valid[:, t - 1]
            has_next = t + 1 < time_steps and valid[:, t + 1]
            if isinstance(has_prev, bool):
                has_prev = np.zeros_like(valid_t)
            if isinstance(has_next, bool):
                has_next = np.zeros_like(valid_t)

            central = valid_t & has_prev & has_next
            forward = valid_t & ~has_prev & has_next
            backward = valid_t & has_prev & ~has_next

            if central.any():
                denom = timestamps_s[t + 1] - timestamps_s[t - 1]
                self._write_agent_derivative_slice(
                    mask=central,
                    target_accelerations=slot_accelerations[:, t],
                    target_yaw_rates=slot_yaw_rates[:, t],
                    velocities_from=slot_velocities[:, t - 1],
                    velocities_to=slot_velocities[:, t + 1],
                    orientations_from=slot_orientations[:, t - 1],
                    orientations_to=slot_orientations[:, t + 1],
                    headings=slot_orientations[:, t],
                    denom=denom,
                )
            if forward.any():
                denom = timestamps_s[t + 1] - timestamps_s[t]
                self._write_agent_derivative_slice(
                    mask=forward,
                    target_accelerations=slot_accelerations[:, t],
                    target_yaw_rates=slot_yaw_rates[:, t],
                    velocities_from=slot_velocities[:, t],
                    velocities_to=slot_velocities[:, t + 1],
                    orientations_from=slot_orientations[:, t],
                    orientations_to=slot_orientations[:, t + 1],
                    headings=slot_orientations[:, t],
                    denom=denom,
                )
            if backward.any():
                denom = timestamps_s[t] - timestamps_s[t - 1]
                self._write_agent_derivative_slice(
                    mask=backward,
                    target_accelerations=slot_accelerations[:, t],
                    target_yaw_rates=slot_yaw_rates[:, t],
                    velocities_from=slot_velocities[:, t - 1],
                    velocities_to=slot_velocities[:, t],
                    orientations_from=slot_orientations[:, t - 1],
                    orientations_to=slot_orientations[:, t],
                    headings=slot_orientations[:, t],
                    denom=denom,
                )

    @staticmethod
    def _write_agent_derivative_slice(
        *,
        mask: np.ndarray,
        target_accelerations: np.ndarray,
        target_yaw_rates: np.ndarray,
        velocities_from: np.ndarray,
        velocities_to: np.ndarray,
        orientations_from: np.ndarray,
        orientations_to: np.ndarray,
        headings: np.ndarray,
        denom: float,
    ) -> None:
        if float(denom) <= 0.0:
            return
        dv = (velocities_to - velocities_from) / denom
        dyaw = _wrap_angle_np(orientations_to - orientations_from) / denom
        acceleration = dv[:, 0] * np.cos(headings) + dv[:, 1] * np.sin(headings)
        target_accelerations[mask] = acceleration[mask]
        target_yaw_rates[mask] = dyaw[mask]

    def _select_current_agents(
        self,
        agents: list[Any],
        observations: list[Any],
        valid_frames: list[bool],
        anchor_state: EgoState,
    ) -> list[Any]:
        anchor = anchor_state.rear_axle
        candidates = {_track_key(agent): agent for agent in agents}
        distances = {key: float("inf") for key in candidates}
        for observation, is_valid in zip(observations, valid_frames):
            if not is_valid:
                continue
            for agent in _observation_agents(observation):
                key = _track_key(agent)
                if key not in candidates:
                    continue
                dx = float(agent.center.x) - float(anchor.x)
                dy = float(agent.center.y) - float(anchor.y)
                distances[key] = min(distances[key], dx * dx + dy * dy)

        ordered = sorted(candidates, key=lambda key: (distances[key], key))
        capacity = max(self.config.max_agents - 1, 0)
        reserve_count = max(0, min(self.config.min_vehicle_agents, capacity))
        vehicle_keys = [
            key for key in ordered if _agent_type_code(candidates[key]) == 6.0
        ][:reserve_count]
        vehicle_key_set = set(vehicle_keys)
        selected_keys = (
            vehicle_keys
            + [key for key in ordered if key not in vehicle_key_set][
                : capacity - len(vehicle_keys)
            ]
        )
        return [candidates[key] for key in selected_keys]

    def _write_ego_slot(
        self,
        *,
        slot: int,
        time_idx: int,
        ego_state: EgoState,
        anchor_state: EgoState,
        positions: np.ndarray,
        velocities: np.ndarray,
        sizes: np.ndarray,
        orientations: np.ndarray,
        agent_types: np.ndarray,
        masks: np.ndarray,
        accelerations: np.ndarray,
        yaw_rates: np.ndarray,
        steering: np.ndarray,
    ) -> None:
        anchor = anchor_state.rear_axle
        pose = ego_state.rear_axle
        pos_x, pos_y = global_to_anchor_xy(
            pose.x, pose.y, anchor.x, anchor.y, anchor.heading
        )
        positions[0, slot, time_idx, 0] = pos_x
        positions[0, slot, time_idx, 1] = pos_y
        vel_x, vel_y = local_vector_to_anchor_xy(
            ego_state.dynamic_car_state.rear_axle_velocity_2d.x,
            ego_state.dynamic_car_state.rear_axle_velocity_2d.y,
            pose.heading,
            anchor.heading,
        )
        velocities[0, slot, time_idx, 0] = vel_x
        velocities[0, slot, time_idx, 1] = vel_y
        sizes[0, slot, time_idx, 0] = ego_state.car_footprint.vehicle_parameters.length
        sizes[0, slot, time_idx, 1] = ego_state.car_footprint.vehicle_parameters.width
        orientations[0, slot, time_idx] = global_heading_to_anchor(
            pose.heading,
            anchor.heading,
        )
        agent_types[0, slot, time_idx] = DRIVERL_EGO_TYPE
        accelerations[0, slot, time_idx] = float(
            ego_state.dynamic_car_state.rear_axle_acceleration_2d.x
        )
        yaw_rates[0, slot, time_idx] = float(
            ego_state.dynamic_car_state.angular_velocity
        )
        steering[0, slot, time_idx] = float(ego_state.tire_steering_angle)
        masks[0, slot, time_idx] = True

    def _write_agent_slot(
        self,
        *,
        slot: int,
        time_idx: int,
        agent: Any,
        anchor_state: EgoState,
        positions: np.ndarray,
        velocities: np.ndarray,
        sizes: np.ndarray,
        orientations: np.ndarray,
        agent_types: np.ndarray,
        masks: np.ndarray,
        yaw_rates: np.ndarray,
    ) -> None:
        anchor = anchor_state.rear_axle
        pos_x, pos_y = global_to_anchor_xy(
            agent.center.x,
            agent.center.y,
            anchor.x,
            anchor.y,
            anchor.heading,
        )
        positions[0, slot, time_idx, 0] = pos_x
        positions[0, slot, time_idx, 1] = pos_y
        velocity = getattr(agent, "velocity", None)
        if velocity is not None:
            vel_x, vel_y = vector_to_anchor_xy(velocity.x, velocity.y, anchor.heading)
            velocities[0, slot, time_idx, 0] = vel_x
            velocities[0, slot, time_idx, 1] = vel_y
        sizes[0, slot, time_idx, 0] = float(
            getattr(agent.box, "length", self.config.default_agent_length)
        )
        sizes[0, slot, time_idx, 1] = float(
            getattr(agent.box, "width", self.config.default_agent_width)
        )
        orientations[0, slot, time_idx] = global_heading_to_anchor(
            agent.center.heading,
            anchor.heading,
        )
        agent_types[0, slot, time_idx] = _agent_type_code(agent)
        yaw_rate = getattr(agent, "angular_velocity", None)
        yaw_rates[0, slot, time_idx] = 0.0 if yaw_rate is None else float(yaw_rate)
        masks[0, slot, time_idx] = True

    def _npc_id_array(self, masks: torch.Tensor) -> torch.Tensor:
        ids = torch.arange(
            masks.shape[1],
            dtype=torch.long,
            device=masks.device,
        ).unsqueeze(0)
        return ids.masked_fill(~masks.any(dim=2), -1)


_EGO_KEY = "__ego__"


def _left_pad(items: list[Any], target_length: int) -> list[Any]:
    if len(items) >= target_length:
        return list(items[-target_length:])
    if not items:
        raise ValueError("Cannot pad an empty sequence.")
    return [items[0]] * (target_length - len(items)) + list(items)


def _sample_history_by_time(
    ego_states: list[EgoState],
    observations: list[Any],
    target_length: int,
    sample_interval: float,
) -> tuple[list[EgoState], list[Any], list[bool]]:
    if len(ego_states) != len(observations):
        raise ValueError("Ego and observation histories must have the same length.")
    if not ego_states:
        raise ValueError("Cannot sample an empty history.")

    selected_ego_states: list[EgoState] = []
    selected_observations: list[Any] = []
    selected_valid: list[bool] = []
    earliest_time_s = ego_states[0].time_us * 1e-6
    anchor_time_s = ego_states[-1].time_us * 1e-6
    cursor = 0
    for sample_idx in range(target_length):
        target_time_s = (
            anchor_time_s - (target_length - 1 - sample_idx) * sample_interval
        )
        while cursor + 1 < len(ego_states) and abs(
            ego_states[cursor + 1].time_us * 1e-6 - target_time_s
        ) <= abs(ego_states[cursor].time_us * 1e-6 - target_time_s):
            cursor += 1
        selected_ego_states.append(ego_states[cursor])
        selected_observations.append(observations[cursor])
        selected_valid.append(target_time_s >= earliest_time_s - 1e-9)
    return selected_ego_states, selected_observations, selected_valid


def _route_goal_positions(
    positions: torch.Tensor,
    velocities: torch.Tensor,
    route_points: torch.Tensor,
    route_mask: torch.Tensor,
    *,
    horizon_s: float = 12.0,
    min_speed_mps: float = 5.0,
    num_goal_positions: int,
) -> torch.Tensor:
    return route_goal_positions(
        positions,
        velocities,
        route_points,
        route_mask,
        horizon_s=horizon_s,
        min_speed_mps=min_speed_mps,
        num_goal_positions=num_goal_positions,
    )


def _observation_agents(observation: Any) -> list[Any]:
    tracked_objects = getattr(observation, "tracked_objects", None)
    if tracked_objects is None:
        return []
    if hasattr(tracked_objects, "get_agents") or hasattr(
        tracked_objects, "get_static_objects"
    ):
        agents = (
            list(tracked_objects.get_agents())
            if hasattr(tracked_objects, "get_agents")
            else []
        )
        static_objects = (
            list(tracked_objects.get_static_objects())
            if hasattr(tracked_objects, "get_static_objects")
            else []
        )
        return agents + static_objects
    return list(tracked_objects)


def _track_key(agent: Any) -> str:
    return str(
        getattr(agent, "track_token", None)
        or getattr(agent, "token", None)
        or id(agent)
    )


def _agent_type_code(agent: Any) -> float:
    tracked_type = getattr(agent, "tracked_object_type", None)
    name = getattr(tracked_type, "name", "")
    if name == "PEDESTRIAN":
        return 4.0
    if name in {"BICYCLE", "MOTORCYCLE"}:
        return 5.0
    if name in {"VEHICLE", "EGO"}:
        return 6.0
    return 1.0


def _frame_from_ego_state(ego_state: EgoState) -> FrameRow:
    return FrameRow(
        token=b"current",
        token_hex="current",
        timestamp_us=int(ego_state.time_us),
        scene_token=None,
        x=float(ego_state.rear_axle.x),
        y=float(ego_state.rear_axle.y),
        yaw=float(ego_state.rear_axle.heading),
        vx=float(ego_state.dynamic_car_state.rear_axle_velocity_2d.x),
        vy=float(ego_state.dynamic_car_state.rear_axle_velocity_2d.y),
        ax=float(ego_state.dynamic_car_state.rear_axle_acceleration_2d.x),
        ay=float(ego_state.dynamic_car_state.rear_axle_acceleration_2d.y),
        yaw_rate=float(ego_state.dynamic_car_state.angular_velocity),
    )


def _traffic_lights_by_current_token(
    anchor_frame: FrameRow,
    traffic_light_data: Any,
) -> dict[bytes, list[dict[str, object]]]:
    if traffic_light_data is None:
        return {anchor_frame.token: []}
    states = []
    for item in traffic_light_data:
        raw_status = getattr(item, "status", "unknown")
        status = getattr(raw_status, "name", str(raw_status)).lower().split(".")[-1]
        states.append(
            {
                "lane_connector_id": int(getattr(item, "lane_connector_id")),
                "status": status,
            }
        )
    return {anchor_frame.token: states}


def _fit_route_points(route_points: np.ndarray, max_route_points: int) -> np.ndarray:
    result = np.zeros((max_route_points, 2), dtype=np.float32)
    if route_points.size == 0:
        return result
    route_points = np.asarray(route_points, dtype=np.float32).reshape(-1, 2)
    if len(route_points) <= max_route_points:
        result[: len(route_points)] = route_points
    else:
        idx = (
            np.linspace(0, len(route_points) - 1, max_route_points)
            .round()
            .astype(np.int64)
        )
        result[:] = route_points[idx]
    return result


def _route_map_query_frames(
    *,
    anchor_frame: FrameRow,
    route_points: np.ndarray | None,
    num_points: int,
    spacing_m: float,
    max_distance_m: float,
) -> list[FrameRow]:
    if (
        route_points is None
        or num_points <= 0
        or spacing_m <= 0.0
        or max_distance_m < 0.0
    ):
        return []
    points = np.asarray(route_points, dtype=np.float32).reshape(-1, 2)
    valid = np.any(np.abs(points) > 0.0, axis=1)
    if len(valid) > 0:
        valid[0] = True
    points = points[valid]
    if len(points) == 0:
        return []

    distances_to_ego = np.linalg.norm(points, axis=1)
    start_idx = int(np.argmin(distances_to_ego))
    points = points[start_idx:]
    if len(points) == 0:
        return []

    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    progress = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    keep = progress <= float(max_distance_m)
    points = points[keep]
    progress = progress[keep]
    if len(points) == 0:
        return []

    max_route_distance = min(float(max_distance_m), float(progress[-1]))
    target_progress = np.arange(0.0, max_route_distance + 1e-6, float(spacing_m))
    if len(target_progress) == 0:
        target_progress = np.asarray([0.0], dtype=np.float32)
    if len(target_progress) > num_points:
        target_progress = target_progress[:num_points]
    indices = np.searchsorted(progress, target_progress, side="left")
    indices = np.clip(indices, 0, len(points) - 1)
    points = points[indices]

    query_frames: list[FrameRow] = []
    previous_idx = -1
    for idx, point in enumerate(points):
        if previous_idx >= 0 and np.allclose(point, points[previous_idx]):
            continue
        previous_idx = idx
        if idx + 1 < len(points):
            direction = points[idx + 1] - point
        elif idx > 0:
            direction = point - points[idx - 1]
        else:
            direction = np.array([1.0, 0.0], dtype=np.float32)
        local_heading = math.atan2(float(direction[1]), float(direction[0]))
        x_local, y_local = float(point[0]), float(point[1])
        cos_h = math.cos(anchor_frame.yaw)
        sin_h = math.sin(anchor_frame.yaw)
        x_global = anchor_frame.x + cos_h * x_local - sin_h * y_local
        y_global = anchor_frame.y + sin_h * x_local + cos_h * y_local
        query_frames.append(
            FrameRow(
                token=anchor_frame.token,
                token_hex=anchor_frame.token_hex,
                timestamp_us=anchor_frame.timestamp_us,
                scene_token=anchor_frame.scene_token,
                x=x_global,
                y=y_global,
                yaw=anchor_frame.yaw + local_heading,
                vx=anchor_frame.vx,
                vy=anchor_frame.vy,
                ax=anchor_frame.ax,
                ay=anchor_frame.ay,
                yaw_rate=anchor_frame.yaw_rate,
            )
        )
    return query_frames


def _wrap_angle_np(angle: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(angle), np.cos(angle))


def _move_scenario_data_to_device(
    scenario_data: ScenarioData, device: torch.device
) -> None:
    if scenario_data._agent_control_manager is not None:
        manager = scenario_data._agent_control_manager
        manager._control_types = manager._control_types.to(device)
        manager.device = device
    for name, value in vars(scenario_data).items():
        if isinstance(value, torch.Tensor):
            setattr(scenario_data, name, value.to(device))
        elif hasattr(value, "values") and isinstance(value.values, dict):
            value.values = {
                key: tensor.to(device) if isinstance(tensor, torch.Tensor) else tensor
                for key, tensor in value.values.items()
            }
