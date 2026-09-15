"""
This module defines a model-based agent that uses a simple neural network.
"""

import torch

from driverl.agents.base_agent import AGENT_REGISTER, LearningAgent, _history_slice
from driverl.agents.config import AgentConfig
from driverl.agents.learning.networks.base_network import BaseNetwork
from driverl.datatypes.data_enums import LaneType
from driverl.datatypes.goal_position_utils import normalize_goal_positions
from driverl.datatypes.scenario_data import ScenarioData
from driverl.env.constants import MAX_STEERING_RATE, MIN_STEERING_RATE
from driverl.env.domain_randomization.config import DomainRandomizationConfig
from driverl.utils.geometry import translate_and_rotate_2d_points, vehicle_to_polygon
from driverl.utils.gym_compat import Box, Discrete, Space


def _select_agent_indices(
    agent_distances: torch.Tensor,
    agent_types: torch.Tensor,
    agent_mask: torch.Tensor,
    max_agents: int,
    min_vehicle_agents: int,
    min_pedestrian_agents: int,
) -> torch.Tensor:
    batch_size, num_agents, _ = agent_distances.shape
    ego_index = torch.zeros(
        batch_size, 1, dtype=torch.long, device=agent_distances.device
    )
    other_k = min(max_agents - 1, num_agents - 1)
    if other_k <= 0:
        return ego_index

    other_mask = agent_mask[:, 1:].bool()
    latest_indices = other_mask.shape[-1] - 1 - other_mask.flip(-1).long().argmax(-1)
    other_distances = torch.gather(
        agent_distances[:, 1:], dim=-1, index=latest_indices.unsqueeze(-1)
    ).squeeze(-1)
    other_distances = other_distances.masked_fill(~other_mask.any(dim=-1), torch.inf)
    vehicle_mask = ((agent_types[:, 1:] == 6) & other_mask).any(dim=-1)
    pedestrian_mask = ((agent_types[:, 1:] == 4) & other_mask).any(dim=-1)
    generic_mask = ((agent_types[:, 1:] == 1) & other_mask).any(dim=-1)

    distance_order = other_distances.argsort(dim=1)
    ordered_vehicles = vehicle_mask.gather(1, distance_order)
    ordered_pedestrians = pedestrian_mask.gather(1, distance_order)
    reserved_vehicles = ordered_vehicles & (
        ordered_vehicles.cumsum(dim=1) <= min_vehicle_agents
    )
    reserved_pedestrians = ordered_pedestrians & (
        ordered_pedestrians.cumsum(dim=1) <= min_pedestrian_agents
    )
    vehicle_reserve_mask = torch.zeros_like(vehicle_mask).scatter(
        1, distance_order, reserved_vehicles
    )
    pedestrian_reserve_mask = torch.zeros_like(pedestrian_mask).scatter(
        1, distance_order, reserved_pedestrians
    )

    finite_max = torch.where(
        other_distances.isfinite(), other_distances, torch.zeros_like(other_distances)
    ).amax(dim=1, keepdim=True)
    selection_scores = other_distances + generic_mask * (finite_max + 1)
    selection_scores = torch.where(
        vehicle_reserve_mask,
        torch.full_like(selection_scores, -2),
        selection_scores,
    )
    selection_scores = torch.where(
        pedestrian_reserve_mask,
        torch.full_like(selection_scores, -1),
        selection_scores,
    )

    _, other_indices = torch.topk(selection_scores, k=other_k, dim=1, largest=False)
    return torch.cat([ego_index, other_indices + 1], dim=1)


@AGENT_REGISTER.register_module
class VanillaNetAgent(LearningAgent):
    """
    A model-based agent that uses a Transformer-style neural network to make decisions.

    The agent processes ego-centric agent data and augments it with lane-boundary
    context distilled from the high-quality lane geometry.
    """

    # Feature dimensions
    AGENT_FEATURE_SIZE = 15  # pos(2), vel(2), yaw(1), size(2), polygon corners(8)
    GOAL_FEATURE_SIZE = 2  # num_goal_positions * goal_pos(2)
    KINEMATICS_FEATURE_SIZE = (
        8  # vel(2), yaw(1), size(2), steering(1), acceleration(1), yaw_rate(1)
    )
    LANE_BOUND_FEATURE_SIZE = 6  # x_start, y_start, x_end, y_end, type, speed_limit
    OCC_NUM_RAYS = 512
    OCC_K = 1

    def __init__(
        self,
        config: AgentConfig,
        action_space: Space,
        batch_size: int,
        action_key_to_values: dict,
        frame_time_interval: float,
        no_goal_allowed: bool,
        domain_randomization_config: DomainRandomizationConfig,
    ):
        super().__init__(config)
        if self.config is None:
            raise ValueError("VanillaNetAgent requires a configuration object.")

        self.action_space = action_space
        self.batch_size = batch_size
        self.action_key_to_values = action_key_to_values
        self.frame_time_interval = frame_time_interval
        self.no_goal_allowed = no_goal_allowed
        self.domain_randomization_config = domain_randomization_config
        self.num_goal_positions = max(1, int(getattr(config, "num_goal_positions", 1)))
        self.GOAL_FEATURE_SIZE = self.num_goal_positions * 2
        self.use_steering_rate = bool(getattr(config, "use_steering_rate", False))
        self.KINEMATICS_FEATURE_SIZE = 9 if self.use_steering_rate else 8
        self.enable_traffic_light_features = bool(
            getattr(config, "enable_traffic_light_features", False)
        )
        self.LANE_BOUND_FEATURE_SIZE = 10 if self.enable_traffic_light_features else 6

        if no_goal_allowed:
            self.GOAL_FEATURE_SIZE += 1

        # Determine output size based on action space type
        if isinstance(action_space, Discrete):
            output_action_size = self.action_space.n
        elif isinstance(action_space, Box):
            action_dim = int(action_space.shape[0])  # 2 for [jerk_long, lat_command]
            # Network outputs [alpha, beta] → need action_dim * 2 raw params
            output_action_size = action_dim
            self.is_continuous = True
            # Register action bounds for linear rescaling of Beta distribution samples.
            self.register_buffer(
                "_action_low",
                torch.as_tensor(
                    action_space.low, dtype=torch.float32, device=self.config.device
                ),
                persistent=False,
            )
            self.register_buffer(
                "_action_high",
                torch.as_tensor(
                    action_space.high, dtype=torch.float32, device=self.config.device
                ),
                persistent=False,
            )
        else:
            raise ValueError(
                f"VanillaNetAgent supports Discrete or Box action spaces, got {type(action_space)}."
            )

        # Radius for filtering entities around the ego agent
        self.radius = getattr(self.config, "agent_filter_radius", 200.0)
        # Asymmetric longitudinal lane window. Defaults (200/200) preserve the
        # original circular behaviour; override via agent config to shrink
        # either side (e.g. 100m behind).
        self.lane_forward_range = getattr(self.config, "lane_forward_range", 200.0)
        self.lane_backward_range = getattr(self.config, "lane_backward_range", 200.0)

        # Build network configuration from agent config
        network_name = getattr(self.config, "network_name", "VanillaNet")
        network_kwargs = {
            "agent_feature_size": self.AGENT_FEATURE_SIZE,
            "goal_feature_size": self.GOAL_FEATURE_SIZE,
            "kinematics_feature_size": self.KINEMATICS_FEATURE_SIZE,
            "lane_bound_feature_size": self.LANE_BOUND_FEATURE_SIZE,
            "output_size": output_action_size,
            "embed_dim": self.config.embed_dim,
            "num_heads": self.config.num_heads,
            "no_goal_allowed": no_goal_allowed,
            "max_agents": self.config.max_agents,
            "history_steps": self.config.history_steps + self.config.future_steps,
            "future_steps": self.config.future_steps,
            "enable_agent_frame_masking": self.config.enable_agent_frame_masking,
            "enable_occupancy_grid": self.config.enable_occupancy_grid,
            "is_continuous": self.is_continuous,
            "enable_value_decomposition": self.config.enable_value_decomposition,
            "num_goal_positions": self.num_goal_positions,
            "enable_ordered_goal_encoding": self.config.enable_ordered_goal_encoding,
            "num_agent_attention_layers": self.config.num_agent_attention_layers,
            "enable_lane_attention": self.config.enable_lane_attention,
            "num_lane_attention_layers": self.config.num_lane_attention_layers,
            "num_heads_lane": self.config.num_heads_lane,
        }

        self.model = BaseNetwork.network_factory(network_name, **network_kwargs)
        self.model.to(self.config.device)

    def _prepare_lane_boundaries(
        self,
        lane_boundaries: torch.Tensor,
        lane_boundaries_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Normalize lane boundary features (type/speed) for downstream models."""
        if lane_boundaries.numel() == 0:
            return lane_boundaries, lane_boundaries_mask

        lane_features = lane_boundaries.clone()
        if getattr(self.config, "use_raw_lane_types", False):
            return lane_features, lane_boundaries_mask

        original_types = lane_features[..., 2]
        hard_boundary_mask = original_types == float(LaneType.CURB.code)
        solid_mask = (original_types == float(LaneType.SOLID_LINE.code)) | (
            original_types == float(LaneType.DOUBLE_SOLID_LINE.code)
        )
        # if getattr(self.config, "mode", "train") == "export":
        #     # Hack: reverse curb point order for curbs on the ego left side.
        #     curb_lane = torch.any(hard_boundary_mask, dim=2)
        #     left_lane = lane_features[..., 1].mean(dim=2) > 0
        #     reverse_mask = curb_lane & left_lane & lane_boundaries_mask
        #     reversed_features = torch.flip(lane_features, dims=[2])
        #     lane_features = torch.where(
        #         reverse_mask.unsqueeze(-1).unsqueeze(-1),
        #         reversed_features,
        #         lane_features,
        #     )

        new_types = torch.zeros_like(original_types)
        new_types = torch.where(
            hard_boundary_mask, torch.ones_like(new_types), new_types
        )
        new_types = torch.where(solid_mask, torch.full_like(new_types, 2.0), new_types)
        type_update_mask = lane_boundaries_mask.unsqueeze(-1) & (original_types >= 0)
        lane_features[..., 2:3] = torch.where(
            type_update_mask.unsqueeze(-1),
            new_types.unsqueeze(-1),
            lane_features[..., 2:3],
        )

        return lane_features, lane_boundaries_mask

    def _select_nearest_lanes(
        self,
        lanes: torch.Tensor,
        lanes_mask: torch.Tensor,
        max_lanes: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply the shared spatial window and nearest-lane selection."""
        if lanes.shape[1] == 0:
            return lanes, lanes_mask

        lane_x = lanes[..., 0]
        lane_y = lanes[..., 1]
        in_window = (
            (lane_x >= -self.lane_backward_range)
            & (lane_x <= self.lane_forward_range)
            & (lane_y.abs() <= self.radius)
        )
        lane_points_xy = lanes[..., :2].clone()
        lane_points_xy[..., 0] *= 0.5
        point_distances = torch.norm(lane_points_xy, dim=-1).masked_fill(
            ~in_window, torch.inf
        )
        lane_distances = point_distances.amin(dim=2)
        lanes_mask = lanes_mask.bool() & torch.isfinite(lane_distances)
        scores = torch.where(
            (in_window & (lane_x > 0.0)).any(dim=2),
            lane_distances * 0.5,
            lane_distances,
        ).masked_fill(~lanes_mask, torch.inf)

        k = max(1, min(max_lanes, lanes.shape[1]))
        topk_indices = torch.topk(scores, k=k, dim=1, largest=False).indices
        lanes = torch.gather(
            lanes,
            dim=1,
            index=topk_indices.unsqueeze(-1)
            .unsqueeze(-1)
            .expand(-1, -1, lanes.shape[2], lanes.shape[3]),
        )
        return lanes, torch.gather(lanes_mask, dim=1, index=topk_indices)

    def _preprocess_deploy(
        self,
        positions,
        velocities,
        sizes,
        orientations,
        agent_types,
        accelerations,
        yaw_rates,
        steering_angles,
        agent_masks,
        lanes_points,
        lanes_points_mask,
        goal_positions,
        frame_drop_mask,
        not_blind_mask,
        randomized_features,
        training_mask,
        goal_reached_scalar,
        occ_points,
        history_steps=5,
    ):
        ## 1. slice history
        env_idx = slice(None)
        step_idx = _history_slice(
            history_steps, getattr(self.config, "history_frame_gap", 1)
        )
        # Extract data using the determined indices.
        positions = positions[env_idx, :, step_idx, :]
        velocities = velocities[env_idx, :, step_idx, :]
        orientations = orientations[env_idx, :, step_idx]
        sizes = sizes[env_idx, :, step_idx, :]
        types = agent_types[env_idx, :, step_idx]
        agent_mask = agent_masks[env_idx, :, step_idx]
        steering = steering_angles[env_idx, :, step_idx]
        accelerations = accelerations[env_idx, :, step_idx]
        goal_reached = torch.full_like(
            positions[:, :, 0, 0], goal_reached_scalar, dtype=torch.bool
        )
        visible_mask = None

        # Extract angular velocity (yaw rate) if available
        yaw_rates = yaw_rates[env_idx, :, step_idx]
        frame_drop_mask = frame_drop_mask[env_idx, :, step_idx]

        ## 2. coordinate transform
        base_positions = positions[:, 0:1, -1, :].clone()  # [B, 1, 2]
        ego_velocities = velocities[:, 0:1, -1, :].clone()  # [B, 1, 2]
        ego_orientations = orientations[:, 0:1, -1].clone()  # [B, 1]
        # Calculate yaw from velocity, which is more stable at higher speeds.
        ego_velocity_yaws = torch.atan2(
            ego_velocities[..., 1], ego_velocities[..., 0]
        )  # [B, 1]
        base_orientations = torch.where(
            ego_velocities.norm(dim=-1) < 0.5, ego_orientations, ego_velocity_yaws
        )  # [B, 1]
        (
            positions,
            velocities,
            orientations,
            goal_positions,
            lanes_points,
        ) = self.coordinates_transformation(
            base_positions,
            base_orientations,
            positions,
            velocities,
            orientations,
            goal_positions,
            lanes_points,
        )

        return self.tensor_to_model_input(
            positions,  # [B, A, history_steps, 2]
            velocities,  # [B, A, history_steps, 2]
            orientations,  # [B, A, history_steps]
            sizes,  # [B, A, history_steps, 2]
            types,
            agent_mask,  # [B, A, history_steps]
            goal_positions,  # [B, A, 2 * num_goal_positions]
            goal_reached,  # [B, A]
            steering,  # [B, A, history_steps]
            accelerations,  # [B, A, history_steps]
            yaw_rates,  # [B, A, history_steps]
            lanes_points,
            lanes_points_mask,
            visible_mask,  # [B, A, history_steps, A] or None
            frame_drop_mask,  # [B, A, history_steps]
            occ_points,  # [B, R, 2]
        )

    def features_to_vis_inputs(
        self,
        features: tuple,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """Convert VanillaNet 9-tuple into the unified visualizer dict.

        The returned dict uses the **same keys** as AgentCentricNet so that
        ``calculate_model_input`` in *calculate.py* can handle both without
        branching.  Tensors are packed to ``(N, ...)`` where ``N = B``
        (one controlled ego per world).  The returned ``training_mask`` keeps
        the original scenario agent dimension even when model preprocessing
        top-k filters nearby agents, so visualizer scatter can index by the
        real scene agent id.

        Key mapping (VanillaNet → unified):
        - ``vehicle_state_features (N, 18/19)`` – ego current agent_features[15]
          concatenated with steer/(optional steer_rate)/accel/yaw_rate.
        - ``other_agents_features (N, A-1, 15)`` – all non-ego agents now.
        - ``other_agents_mask (N, A-1)`` – current-frame presence mask.
        - ``lane_features (N, M, 8/12)`` – lane start(2)+end(2)+zeros(2)+type(1)+speed(1)
          plus optional traffic-light one-hot.
        - ``lane_mask (N, M)``.
        """
        (
            agent_features,  # [B, A, T, 15]
            goal_features,  # [B, A, 2or3]
            kinematics_features,  # [B, A, 8 or 9]
            agent_mask,  # [B, A, T]
            lane_boundaries,  # [B, M, 6 or 10]
            lane_boundaries_mask,  # [B, M]
            occ_points,  # [B, R, 2]
            occ_mask,  # [B, R]
            visible_mask,  # [B, A, T, A] or None
        ) = features

        B, A, T, _ = agent_features.shape
        current_idx = T - int(getattr(self.config, "future_steps", 0)) - 1

        # --- training_mask: only ego (col 0) is controlled ---
        original_agent_count = int(getattr(self, "_original_agent_count", A))
        training_mask = torch.zeros(
            B, original_agent_count, dtype=torch.bool, device=agent_features.device
        )
        training_mask[:, 0] = True  # N = B

        # --- vehicle_state_features (N=B, 18 or 19) ---
        ego_last = agent_features[:, 0, current_idx, :]  # [B, 15]
        ego_kin = kinematics_features[:, 0, 5:]
        vehicle_state_features = torch.cat([ego_last, ego_kin], dim=-1)

        # --- other_agents_features (N=B, A-1, 15) at current frame ---
        other_agents_features = agent_features[:, 1:, current_idx, :]
        other_agents_mask = agent_mask[:, 1:, current_idx]

        # --- lane_features (N=B, M, 8/12): start(2)+end(2)+pad(2)+type(1)+speed(1)+tl(4) ---
        lb = lane_boundaries  # [B, M, 6 or 10]: x_s, y_s, x_e, y_e, type, speed, optional tl
        pad = torch.zeros(B, lb.shape[1], 2, dtype=lb.dtype, device=lb.device)
        lane_feats = torch.cat(
            [lb[..., 0:4], pad, lb[..., 4:6], lb[..., 6:]], dim=-1
        )  # [B, M, 8/12]: start(2)+end(2)+zeros(2)+type(1)+speed(1)+tl(4)
        #  Remap indices so calculate_road_graph_ego reads type@[6], speed@[7]

        return {
            "vehicle_state_features": vehicle_state_features,
            "other_agents_features": other_agents_features,
            "other_agents_mask": other_agents_mask,
            "model_input_agent_features": agent_features,
            "model_input_agent_mask": agent_mask,
            "lane_features": lane_feats,
            "lane_mask": lane_boundaries_mask,
            "lane_types_are_raw": torch.full(
                (B,),
                getattr(self.config, "use_raw_lane_types", False),
                dtype=torch.bool,
                device=lb.device,
            ),
            "goal_positions": goal_features[:, 0, : self.GOAL_FEATURE_SIZE],
        }, training_mask

    def forward_policy_params(self, features) -> tuple[torch.Tensor, torch.Tensor]:
        """Return batched policy parameters and values."""
        if self.model is None:
            raise ValueError("LearningAgent requires 'self.model' to be defined.")

        # VanillaNet returns ego-only [B, 1, output_size] and [B, 1, value_dim].
        action_logits, values = self.model(*features)
        A = getattr(
            self, "_original_agent_count", features[3].shape[1]
        )  # agent_mask may be top-k filtered before the model.
        action_logits = action_logits.expand(-1, A, -1)  # [B, A, output_size]
        if not self.config.enable_value_decomposition:
            values = values.squeeze(-1).expand(-1, A)  # [B, A]
        else:
            values = values.expand(-1, A, -1)  # [B, A, value_dim]

        return action_logits, values

    def forward_with_features(
        self,
        features,
        action: torch.Tensor | None = None,
        sampling_method: str = "sample",
        action_keys_tensor: torch.Tensor | None = None,
        current_speed: torch.Tensor | None = None,
    ):
        """Expand ego-only network output to the configured agent dimension."""
        action_logits, values = self.forward_policy_params(features)

        act, log_prob, entropy, val = self._policy_outputs(
            action_logits,
            values,
            action,
            sampling_method,
            action_keys_tensor,
            current_speed=current_speed,
        )
        return act, log_prob, entropy, val, action_logits

    def _preprocess(
        self,
        scenario_data: ScenarioData,
        env_indices=None,
        step_indices=None,
        log_scenario_data: ScenarioData | None = None,
    ):
        """
        Preprocesses raw scenario data into feature tensors for the model.
        """
        lane_centers = None
        lane_centers_mask = None
        if getattr(self.config, "max_lane_centers", 0) > 0:
            if env_indices is None or step_indices is None:
                env_idx = slice(None)
                ego_positions = scenario_data.agent_positions_all[:, 0, -1]
                ego_orientations = scenario_data.agent_orientation_all[:, 0, -1]
            else:
                device = scenario_data.agent_positions_all.device
                env_idx = torch.as_tensor(env_indices, device=device)
                step_idx = torch.as_tensor(step_indices, device=device)
                ego_positions = scenario_data.agent_positions_all[env_idx, 0, step_idx]
                ego_orientations = scenario_data.agent_orientation_all[
                    env_idx, 0, step_idx
                ]
            lane_centers = scenario_data.lanes_centers_points[env_idx].clone()
            lane_centers[..., :2] = translate_and_rotate_2d_points(
                lane_centers[..., :2],
                ego_positions[:, None, None],
                ego_orientations[:, None, None],
            )
            lane_centers_mask = scenario_data.lanes_centers_mask[env_idx]

        (
            positions,  # [B, A, history_steps, 2]
            velocities,  # [B, A, history_steps, 2]
            orientations,  # [B, A, history_steps]
            sizes,  # [B, A, history_steps, 2]
            types,
            agent_mask,  # [B, A, history_steps]
            goal_positions,  # [B, A, 2 * num_goal_positions]
            goal_reached,  # [B, A]
            steering_state,  # [B, A, history_steps]
            steering_control,  # [B, A, history_steps]
            acceleration_state,  # [B, A, history_steps]
            acceleration_control,  # [B, A, history_steps]
            yaw_rates,  # [B, A, history_steps]
            lane_boundaries,
            lane_boundaries_mask,
            visible_mask,  # [B, A, history_steps, A]
            frame_drop_mask,  # [B, A, history_steps]
            advantage_filtering_mask,  # [B, A, history_steps]
            not_blind_mask,
            randomized_features,
            occ_points,
        ) = self.transform_scenario_coordinates(
            scenario_data,
            env_indices=env_indices,
            step_indices=step_indices,
            category="ego",
            log_scenario_data=log_scenario_data,
        )
        return self.tensor_to_model_input(
            positions,  # [B, A, history_steps, 2]
            velocities,  # [B, A, history_steps, 2]
            orientations,  # [B, A, history_steps]
            sizes,  # [B, A, history_steps, 2]
            types,
            agent_mask,  # [B, A, history_steps]
            goal_positions,  # [B, A, 2 * num_goal_positions]
            goal_reached,  # [B, A]
            steering_state,  # [B, A, history_steps]
            acceleration_state,  # [B, A, history_steps]
            yaw_rates,  # [B, A, history_steps]
            lane_boundaries,
            lane_boundaries_mask,
            visible_mask,  # [B, A, history_steps, A]
            frame_drop_mask,  # [B, A, history_steps]
            occ_points,  # [B, R, 2]
            lane_centers=lane_centers,
            lane_centers_mask=lane_centers_mask,
        )

    def tensor_to_model_input(
        self,
        positions,  # [B, A, history_steps, 2]
        velocities,  # [B, A, history_steps, 2]
        orientations,  # [B, A, history_steps]
        sizes,  # [B, A, history_steps, 2]
        types,
        agent_mask,  # [B, A, history_steps]
        goal_positions,  # [B, A, 2 * num_goal_positions]
        goal_reached,  # [B, A]
        steering,  # [B, A, history_steps]
        acceleration,  # [B, A, history_steps]
        yaw_rates,  # [B, A, history_steps]
        lane_boundaries,
        lane_boundaries_mask,
        visible_mask,  # [B, A, history_steps, A]
        frame_drop_mask,  # [B, A, history_steps]
        occ_points,  # [B, R, 2]
        lane_centers: torch.Tensor | None = None,
        lane_centers_mask: torch.Tensor | None = None,
    ):
        # Filter agents outside the specified radius from the ego
        goal_positions = normalize_goal_positions(
            goal_positions, self.num_goal_positions
        )
        future_steps = int(getattr(self.config, "future_steps", 0))
        current_idx = agent_mask.shape[2] - future_steps - 1
        previous_idx = max(current_idx - 1, 0)
        steering_rate_valid_mask = (
            agent_mask[:, :, current_idx : current_idx + 1].bool()
            & agent_mask[:, :, previous_idx : previous_idx + 1].bool()
        )
        if current_idx == 0:
            steering_rate_valid_mask.zero_()
        self._original_agent_count = agent_mask.shape[1]
        agent_distances = torch.norm(positions, dim=-1)  # [B, A, history_steps]
        radius_mask = agent_distances <= self.radius  # [B, A, history_steps]
        agent_mask = agent_mask * radius_mask  # [B, A, history_steps]
        agent_mask *= frame_drop_mask
        # Preserve the legacy feature layout expected by the policy head.
        agent_mask[:, 0, current_idx] = True  # Ensure ego current is never masked
        if not getattr(self.config, "use_ego_history", False):
            agent_mask[:, 0, :current_idx] = False  # Don't use ego history
        agent_mask[:, 0, current_idx + 1 :] = False  # Never expose ego log future
        # agent_mask[:, 1:, -1] = False  # TODO: mock perception latency of current frame

        max_agents = (
            self.config.max_agents
            if self.config.max_agents > 0
            else agent_mask.shape[1]
        )
        if max_agents < agent_mask.shape[1]:
            selection_end = agent_mask.shape[2] if future_steps else current_idx + 1
            topk_agent_indices = _select_agent_indices(
                agent_distances[:, :, :selection_end],
                types[:, :, :selection_end],
                agent_mask[:, :, :selection_end],
                max_agents,
                getattr(self.config, "min_vehicle_agents", 24),
                getattr(self.config, "min_pedestrian_agents", 32),
            )

            def gather_agents(tensor: torch.Tensor) -> torch.Tensor:
                index = topk_agent_indices.reshape(
                    *topk_agent_indices.shape, *([1] * (tensor.dim() - 2))
                ).expand(-1, -1, *tensor.shape[2:])
                return torch.gather(tensor, dim=1, index=index)

            positions = gather_agents(positions)
            velocities = gather_agents(velocities)
            sizes = gather_agents(sizes)
            orientations = gather_agents(orientations)
            types = gather_agents(types)
            agent_mask = gather_agents(agent_mask)
            steering = gather_agents(steering)
            acceleration = gather_agents(acceleration)
            yaw_rates = gather_agents(yaw_rates)
            steering_rate_valid_mask = gather_agents(steering_rate_valid_mask)
            goal_positions = gather_agents(goal_positions)
            goal_reached = gather_agents(goal_reached)

            if visible_mask is not None:
                visible_mask = gather_agents(visible_mask)
                visible_mask = torch.gather(
                    visible_mask,
                    dim=3,
                    index=topk_agent_indices.unsqueeze(1)
                    .unsqueeze(2)
                    .expand(-1, visible_mask.shape[1], visible_mask.shape[2], -1),
                )

        lane_boundaries, lane_boundaries_mask = self._prepare_lane_boundaries(
            lane_boundaries, lane_boundaries_mask
        )
        configured_max_lanes = getattr(self.config, "max_lanes", 0)
        total_lane_limit = (
            configured_max_lanes
            if configured_max_lanes > 0
            else max(1, lane_boundaries.shape[1] // 8)
        )
        max_lane_centers = min(
            getattr(self.config, "max_lane_centers", 0), total_lane_limit
        )
        if (
            max_lane_centers > 0
            and lane_centers is not None
            and lane_centers_mask is not None
        ):
            lane_centers = lane_centers.clone()
            lane_centers[..., 2] = (
                float(LaneType.LANE_CENTER_LINE.code)
                if getattr(self.config, "use_raw_lane_types", False)
                else 3.0
            )
            lane_centers, lane_centers_mask = self._select_nearest_lanes(
                lane_centers,
                lane_centers_mask,
                max_lane_centers,
            )
            if lane_centers.shape[-1] < lane_boundaries.shape[-1]:
                lane_centers = torch.cat(
                    [
                        lane_centers,
                        lane_centers.new_zeros(
                            *lane_centers.shape[:-1],
                            lane_boundaries.shape[-1] - lane_centers.shape[-1],
                        ),
                    ],
                    dim=-1,
                )
            lane_boundaries = torch.cat([lane_boundaries, lane_centers], dim=1)
            lane_boundaries_mask = torch.cat(
                [lane_boundaries_mask, lane_centers_mask], dim=1
            )
        lane_boundaries, lane_boundaries_mask = self._select_nearest_lanes(
            lane_boundaries,
            lane_boundaries_mask,
            total_lane_limit,
        )

        # Assemble final feature tensors, including polygon corners (4 corners * 2 coords)
        polygons = vehicle_to_polygon(
            length=sizes[..., 0],
            width=sizes[..., 1],
            posx=positions[..., 0],
            posy=positions[..., 1],
            theta=orientations,
        )  # [B, A, history_steps, 4, 2]
        polygon_features = polygons.reshape(*polygons.shape[:3], -1)

        agent_features = torch.cat(
            [
                positions,
                velocities,
                orientations.unsqueeze(-1),
                sizes,
                polygon_features,
            ],
            dim=-1,
        )  # [B, A, history_steps, _]

        if self.no_goal_allowed:
            if getattr(
                self, "training", getattr(self.config, "mode", "train") == "train"
            ):
                goal_positions = goal_positions * ~goal_reached.unsqueeze(-1)
            goal_features = torch.cat(
                [goal_positions, goal_reached.unsqueeze(-1)], dim=-1
            )
        else:
            goal_features = goal_positions.clone()

        # Zero out features for masked entities. This is important for MLPs.
        agent_features *= agent_mask.unsqueeze(-1)
        goal_features *= agent_mask[:, :, current_idx : current_idx + 1]
        lane_boundaries *= lane_boundaries_mask.unsqueeze(-1).unsqueeze(
            -1
        )  # Match shape (B, M, P, C)

        kinematics_parts = [
            velocities[:, :, current_idx, :],
            orientations[:, :, current_idx : current_idx + 1],
            sizes[:, :, current_idx, :],
            steering[:, :, current_idx : current_idx + 1],
        ]
        if getattr(self, "use_steering_rate", False):
            steering_rate = (
                steering[:, :, current_idx : current_idx + 1]
                - steering[:, :, previous_idx : previous_idx + 1]
            ) / (
                self.frame_time_interval * getattr(self.config, "history_frame_gap", 1)
            )
            steering_rate = torch.where(
                steering_rate_valid_mask,
                steering_rate,
                torch.zeros_like(steering_rate),
            ).clamp(MIN_STEERING_RATE, MAX_STEERING_RATE)
            kinematics_parts.append(steering_rate)
        kinematics_parts.extend(
            [
                acceleration[:, :, current_idx : current_idx + 1],
                yaw_rates[:, :, current_idx : current_idx + 1],
            ]
        )
        kinematics_features = torch.cat(kinematics_parts, dim=-1)

        raw_lane_boundaries = lane_boundaries
        point_feature_dim = raw_lane_boundaries.shape[-1]
        lane_boundaries = lane_boundaries.reshape(*lane_boundaries.shape[:-2], -1)[
            ..., [0, 1, point_feature_dim, point_feature_dim + 1, 2, 3]
        ]
        enable_traffic_light_features = bool(
            getattr(
                self,
                "enable_traffic_light_features",
                getattr(self.config, "enable_traffic_light_features", False),
            )
        )
        if enable_traffic_light_features:
            lane_boundaries = torch.cat(
                [lane_boundaries, raw_lane_boundaries[:, :, 0, 4:8]], dim=-1
            )

        occ_mask = occ_points[..., 0] >= 0
        return (
            agent_features,
            goal_features,
            kinematics_features,
            agent_mask,
            lane_boundaries,
            lane_boundaries_mask,
            occ_points,
            occ_mask,
            visible_mask,
        )
