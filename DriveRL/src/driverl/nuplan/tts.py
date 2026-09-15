"""PyTorch test-time scaling action selection for nuPlan closed-loop eval."""

from __future__ import annotations

import copy
import hashlib
import math
import struct
from dataclasses import dataclass

import torch

from driverl.datatypes.data_enums import AgentControlManager
from driverl.datatypes.goal_position_utils import (
    apply_goal_pair_mode,
    route_goal_positions,
)
from driverl.datatypes.scenario_data import ScenarioData
from driverl.env.domain_randomization.domain_randomization import RandomizedFeatureDict
from driverl.env.engine.config import (
    EngineConfig,
    EngineRuntimeConfig,
)
from driverl.env.engine.engine import Engine
from driverl.env.engine.reward_calculator import InfoDimension
from driverl.env.engine.reward_decomposition import DECOMPOSED_VALUE_DIM
from driverl.nuplan.tensor_rollout import (
    TensorRolloutState,
    advance_tensor_rollout_state,
)
from driverl.utils.geometry import get_wheelbase_from_length, wrap_angle

DEFAULT_TTS_NUM_CANDIDATES = 8
TTS_HORIZON = 5
TTS_TOTAL_RETURN_SWITCH_MARGIN = 0.03
TTS_USE_BF16 = False
TTS_BASELINE_CANDIDATE_INDEX = 0


def derive_tts_sample_seed(
    base_seed: int, scenario_token: str | bytes, time_us: int
) -> int:
    """Derive a stable per-scenario, per-decision seed for TTS sampling."""
    token_bytes = (
        scenario_token.encode("utf-8")
        if isinstance(scenario_token, str)
        else bytes(scenario_token)
    )
    payload = struct.pack(">Qq", base_seed, time_us) + token_bytes
    digest = hashlib.blake2b(payload, digest_size=8, person=b"driverl-tts-v1").digest()
    return int.from_bytes(digest, "big") & ((1 << 63) - 1)

_PACKED_ROUTE_CONNECTOR_FIELDS = frozenset(
    {
        "route_lane_connector_polygon_vertices",
        "route_lane_connector_polygon_offsets",
        "route_lane_connector_bboxes",
        "route_lane_connector_tl_status_codes",
        "route_lane_connector_ids",
        "route_lane_connector_batch_indices",
        "route_lane_connector_world_offsets",
        "route_lane_connector_data_available",
    }
)


@dataclass(frozen=True)
class NuPlanTTSResult:
    selected_action: torch.Tensor
    candidate_actions: torch.Tensor
    candidate_values: torch.Tensor
    candidate_scores: torch.Tensor
    selected_candidate: torch.Tensor
    candidate_total_rewards: torch.Tensor
    candidate_dones: torch.Tensor
    candidate_terminal_steps: torch.Tensor
    candidate_leaf_total_values: torch.Tensor
    candidate_bootstrap_mask: torch.Tensor
    candidate_valid: torch.Tensor
    sample_seed: int | None


class NuPlanTTSSelector:
    """Select among an argmax baseline and configurable Beta samples."""

    def __init__(
        self,
        *,
        agent,
        env_config,
        engine_config: EngineConfig,
        frame_time_interval: float,
        max_agents: int,
        device: torch.device,
        num_candidates: int = DEFAULT_TTS_NUM_CANDIDATES,
        gamma: float = 0.99,
        route_goal_horizon_s: float = 12.0,
        route_goal_min_speed_mps: float = 5.0,
        route_goal_pair_mode: str = "legacy",
    ) -> None:
        if (
            isinstance(num_candidates, bool)
            or not isinstance(num_candidates, int)
            or num_candidates < 1
        ):
            raise ValueError("TTS num_candidates must be a positive integer.")
        if not math.isfinite(gamma) or not 0.0 <= gamma <= 1.0:
            raise ValueError("TTS gamma must be finite and in [0, 1].")
        reward_horizon_steps = int(getattr(env_config, "num_steps", 0))
        if reward_horizon_steps < 1:
            raise ValueError(
                "TTS requires env_config.num_steps to be a positive rollout "
                f"horizon, got {reward_horizon_steps}."
            )
        history_steps = int(getattr(agent.config, "history_steps", 0))
        if history_steps < 1:
            raise ValueError(
                "TTS requires agent.config.history_steps to be positive, got "
                f"{history_steps}."
            )
        # Engine init_steps is the index of the final observed frame, whereas
        # agent.history_steps is the number of observed frames.
        rollout_init_steps = history_steps - 1
        if not math.isfinite(route_goal_horizon_s) or route_goal_horizon_s <= 0.0:
            raise ValueError("TTS route_goal_horizon_s must be finite and positive.")
        if (
            not math.isfinite(route_goal_min_speed_mps)
            or route_goal_min_speed_mps <= 0.0
        ):
            raise ValueError(
                "TTS route_goal_min_speed_mps must be finite and positive."
            )
        if route_goal_pair_mode not in {
            "legacy",
            "ordered_sequential",
            "duplicate_earlier",
            "duplicate_later",
        }:
            raise ValueError(
                "TTS route_goal_pair_mode must be legacy, ordered_sequential, "
                "duplicate_earlier, or duplicate_later."
            )
        if not bool(getattr(agent.config, "enable_value_decomposition", False)):
            raise ValueError("nuPlan TTS requires enable_value_decomposition=true.")
        if env_config.dynamics_model != "nuplan_bicycle_model":
            raise ValueError(
                "nuPlan TTS currently supports dynamics_model=nuplan_bicycle_model, "
                f"got {env_config.dynamics_model!r}."
            )
        if (
            engine_config.enable_dynamics_noise
            or engine_config.control_delay_frame != 0
        ):
            raise ValueError("TTS rollout requires deterministic, delay-free dynamics.")

        runtime = EngineRuntimeConfig(
            batch_size=num_candidates,
            max_agents=max_agents,
            frame_time_interval=frame_time_interval,
            dynamics_model=env_config.dynamics_model,
            device=str(device),
            # Reward normalization uses the release horizon. The TTS rollout length is
            # tracked independently by ``self.horizon`` below.
            num_steps=reward_horizon_steps,
            # Reward calculators must exclude the observed planner history and
            # start their temporal windows at the first imagined frame.
            init_steps=rollout_init_steps,
            enable_occupancy_grid=env_config.enable_occupancy_grid,
        )
        self.agent = agent
        tts_engine_config = copy.deepcopy(engine_config)
        self.reward_engine = Engine(tts_engine_config, runtime)
        self.dynamics_model = self.reward_engine._dynamics_model
        self.device = device
        self.dt = float(frame_time_interval)
        self.max_acceleration = float(engine_config.nuplan_bicycle_max_acceleration)
        self.max_steering_angle = float(engine_config.nuplan_bicycle_max_steering_angle)
        self.horizon = TTS_HORIZON
        self.num_candidates = num_candidates
        self.num_random_samples = num_candidates - 1
        self.total_return_switch_margin = TTS_TOTAL_RETURN_SWITCH_MARGIN
        self.gamma = float(gamma)
        self.reward_horizon_steps = reward_horizon_steps
        self.rollout_init_steps = rollout_init_steps
        self.route_goal_horizon_s = float(route_goal_horizon_s)
        self.route_goal_min_speed_mps = float(route_goal_min_speed_mps)
        self.route_goal_pair_mode = route_goal_pair_mode
        self.candidate_sources = ("argmax",) + tuple(
            f"sample_{index}" for index in range(1, num_candidates)
        )

    @torch.no_grad()
    def propose_candidate_actions(
        self, scenario_data: ScenarioData, *, sample_seed: int | None = None
    ) -> tuple[torch.Tensor, ScenarioData]:
        """Return the argmax baseline followed by stochastic Beta samples."""
        features = self.agent.preprocess(scenario_data)
        action_params, _ = self._forward_policy_params(features)
        if not bool(getattr(self.agent, "is_continuous", False)):
            raise ValueError("nuPlan TTS requires a continuous Beta policy.")
        if action_params.ndim != 3 or action_params.shape[0] != 1:
            raise ValueError(
                "TTS action params must have shape [1, A, 2D], got "
                f"{tuple(action_params.shape)}."
            )
        action_dim = action_params.shape[-1] // 2
        if action_params.shape[-1] % 2 != 0 or action_dim != 2:
            raise ValueError("nuPlan TTS requires 2D continuous Beta parameters.")
        action_low = getattr(self.agent, "_action_low", None)
        action_high = getattr(self.agent, "_action_high", None)
        if action_low is None or action_high is None:
            raise ValueError("nuPlan TTS requires continuous action bounds.")

        default_actions = self._argmax_actions_from_params(action_params)
        if self.num_random_samples == 0:
            return default_actions[0, 0].unsqueeze(0).float(), scenario_data
        ego_action_params = action_params[:, :1]
        expanded_ego_params = ego_action_params.expand(
            self.num_random_samples, -1, -1
        )
        if sample_seed is None:
            sampled_actions = self.agent.select_action_from_params(
                expanded_ego_params,
                sampling_method="sample",
                action_keys_tensor=None,
                current_speed=None,
            )
        else:
            devices = []
            if expanded_ego_params.device.type == "cuda":
                device_index = expanded_ego_params.device.index
                if device_index is None:
                    device_index = torch.cuda.current_device()
                devices = [device_index]
            with torch.random.fork_rng(devices=devices):
                if expanded_ego_params.device.type == "cuda":
                    with torch.cuda.device(expanded_ego_params.device):
                        torch.cuda.manual_seed(sample_seed)
                else:
                    torch.manual_seed(sample_seed)
                sampled_actions = self.agent.select_action_from_params(
                    expanded_ego_params,
                    sampling_method="sample",
                    action_keys_tensor=None,
                    current_speed=None,
                )
        candidate_actions = torch.cat(
            [default_actions[0, 0].unsqueeze(0), sampled_actions[:, 0]], dim=0
        ).float()
        return candidate_actions, scenario_data

    def _forward_policy_params(self, features):
        return self.agent.forward_policy_params(features)

    def _argmax_actions_from_params(self, action_params: torch.Tensor) -> torch.Tensor:
        return self.agent.select_action_from_params(
            action_params,
            sampling_method="argmax",
            action_keys_tensor=None,
            current_speed=None,
        )

    @torch.no_grad()
    def select_action(
        self, scenario_data: ScenarioData, *, sample_seed: int | None = None
    ) -> NuPlanTTSResult:
        candidate_actions, _ = self.propose_candidate_actions(
            scenario_data, sample_seed=sample_seed
        )
        (
            _,
            leaf_scenario,
            candidate_total_rewards,
            candidate_dones,
            rollout_action_valid,
        ) = (
            self._rollout_candidates(scenario_data, candidate_actions)
        )
        bootstrap_mask = ~candidate_dones.any(dim=1)
        if bool(bootstrap_mask.any()):
            features = self.agent.preprocess(leaf_scenario)
            _, values = self.agent.forward_policy_params(features)
        else:
            values = torch.zeros(
                self.num_candidates,
                scenario_data.agent_positions_all.shape[1],
                DECOMPOSED_VALUE_DIM,
                device=self.device,
            )
        if values.ndim != 3 or values.shape != (
            self.num_candidates,
            scenario_data.agent_positions_all.shape[1],
            DECOMPOSED_VALUE_DIM,
        ):
            raise ValueError(
                "TTS decomposed critic values must have shape "
                f"[{self.num_candidates}, A, {DECOMPOSED_VALUE_DIM}], got "
                f"{tuple(values.shape)}."
            )
        candidate_values = values[:, 0].float()
        candidate_leaf_total_values = candidate_values.sum(dim=-1)
        discounts = self.gamma ** torch.arange(
            self.horizon,
            device=self.device,
            dtype=candidate_total_rewards.dtype,
        )
        candidate_scores = (candidate_total_rewards * discounts).sum(dim=1)
        bootstrap_values = torch.where(
            bootstrap_mask,
            candidate_leaf_total_values,
            torch.zeros_like(candidate_leaf_total_values),
        )
        candidate_scores = (
            candidate_scores + self.gamma**self.horizon * bootstrap_values
        )
        candidate_valid = (
            torch.isfinite(candidate_actions).all(dim=1)
            & rollout_action_valid
            & torch.isfinite(candidate_total_rewards).all(dim=1)
            & (~bootstrap_mask | torch.isfinite(candidate_values).all(dim=1))
            & torch.isfinite(candidate_scores)
        )
        step_indices = torch.arange(self.horizon, device=self.device).expand(
            self.num_candidates, -1
        )
        candidate_terminal_steps = torch.where(
            candidate_dones,
            step_indices,
            torch.full_like(step_indices, self.horizon),
        ).amin(dim=1)
        candidate_terminal_steps = torch.where(
            bootstrap_mask,
            torch.full_like(candidate_terminal_steps, -1),
            candidate_terminal_steps,
        )
        selected_candidate = self._select_baseline_priority_by_total_return(
            candidate_scores, candidate_valid
        )
        return NuPlanTTSResult(
            selected_action=candidate_actions[selected_candidate].unsqueeze(0),
            candidate_actions=candidate_actions,
            candidate_values=candidate_values,
            candidate_scores=candidate_scores,
            selected_candidate=selected_candidate,
            candidate_total_rewards=candidate_total_rewards,
            candidate_dones=candidate_dones,
            candidate_terminal_steps=candidate_terminal_steps,
            candidate_leaf_total_values=candidate_leaf_total_values,
            candidate_bootstrap_mask=bootstrap_mask,
            candidate_valid=candidate_valid,
            sample_seed=sample_seed,
        )

    def _select_baseline_priority_by_total_return(
        self,
        candidate_total_returns: torch.Tensor,
        candidate_valid: torch.Tensor,
    ) -> torch.Tensor:
        if candidate_total_returns.ndim != 1 or candidate_total_returns.numel() < 1:
            raise ValueError("TTS total returns must have shape [N] with N >= 1.")
        if candidate_valid.shape != candidate_total_returns.shape:
            raise ValueError(
                "TTS candidate validity mask must match total returns, got "
                f"{tuple(candidate_valid.shape)} and "
                f"{tuple(candidate_total_returns.shape)}."
            )
        if candidate_valid.dtype != torch.bool:
            raise ValueError("TTS candidate validity mask must be boolean.")
        if not bool(candidate_valid[0]):
            raise RuntimeError(
                "TTS baseline candidate is invalid; refusing to hide a non-finite "
                "action, rollout reward, bootstrap value, or score."
            )
        if candidate_total_returns.numel() == 1:
            return torch.zeros((), device=self.device, dtype=torch.long)
        other_values = candidate_total_returns[1:]
        valid_others = candidate_valid[1:]
        if not bool(valid_others.any()):
            return torch.zeros((), device=self.device, dtype=torch.long)
        comparable_others = torch.where(
            valid_others,
            other_values,
            torch.full_like(other_values, -torch.inf),
        )
        best_other = comparable_others.argmax() + 1
        advantage = candidate_total_returns[best_other] - candidate_total_returns[0]
        if bool(advantage > self.total_return_switch_margin):
            return best_other
        return torch.zeros((), device=self.device, dtype=torch.long)

    def _rollout_candidates(
        self, scenario_data: ScenarioData, actions: torch.Tensor
    ) -> tuple[
        TensorRolloutState,
        ScenarioData,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        state = self._state_from_scenario(scenario_data)
        ego = TensorRolloutState(
            **{
                name: getattr(state, name)[:, :1].expand(
                    self.num_candidates,
                    -1,
                    *([-1] * (getattr(state, name).ndim - 2)),
                )
                for name in state.__dataclass_fields__
            }
        )
        background = state
        reward_scenario = self._repeat_scenario(scenario_data)
        alive = torch.ones(self.num_candidates, device=self.device, dtype=torch.bool)
        total_rewards = torch.zeros(
            self.num_candidates, self.horizon, device=self.device
        )
        dones = torch.zeros(
            self.num_candidates,
            self.horizon,
            device=self.device,
            dtype=torch.bool,
        )
        rollout_action_valid = torch.isfinite(actions).all(dim=1)
        for step in range(self.horizon):
            if not bool(alive.any()):
                break
            step_actions = actions
            if step > 0:
                features = self.agent.preprocess(reward_scenario)
                action_params, _ = self._forward_policy_params(features)
                expected_shape = (
                    self.num_candidates,
                    reward_scenario.agent_positions_all.shape[1],
                    4,
                )
                if action_params.shape != expected_shape:
                    raise ValueError(
                        "TTS continuation policy params must have shape "
                        f"{expected_shape}, got {tuple(action_params.shape)}."
                    )
                policy_actions = self._argmax_actions_from_params(action_params)
                expected_action_shape = expected_shape[:-1] + (2,)
                if policy_actions.shape != expected_action_shape:
                    raise ValueError(
                        "TTS continuation policy actions must have shape "
                        f"{expected_action_shape}, got {tuple(policy_actions.shape)}."
                    )
                step_actions = policy_actions[:, 0].float()
                rollout_action_valid &= (~alive) | torch.isfinite(step_actions).all(
                    dim=1
                )
            ego = advance_tensor_rollout_state(
                ego,
                step_actions.unsqueeze(1),
                self.dynamics_model,
                agent_mask_next=ego.agent_masks[:, :, -1:],
                frame_drop_next=torch.ones_like(ego.frame_drop_mask[:, :, -1:]),
            )
            background = self._advance_background_ctra(background)
            combined = self._combine_ego_and_background(ego, background)
            self._update_scenario_from_state(reward_scenario, combined)
            step_total_reward, step_done = self._evaluate_rollout_step(reward_scenario)
            total_rewards[:, step] = torch.where(
                alive, step_total_reward, torch.zeros_like(step_total_reward)
            )
            newly_done = alive & step_done
            dones[:, step] = newly_done
            alive &= ~newly_done
        leaf_state = self._combine_ego_and_background(ego, background)
        return leaf_state, reward_scenario, total_rewards, dones, rollout_action_valid

    def _combine_ego_and_background(
        self, ego: TensorRolloutState, background: TensorRolloutState
    ) -> TensorRolloutState:
        if background.positions.shape[1] == 1:
            return ego

        def combine(name: str) -> torch.Tensor:
            ego_value = getattr(ego, name)
            bg_value = getattr(background, name)[:, 1:].expand(
                self.num_candidates,
                -1,
                *([-1] * (getattr(background, name).ndim - 2)),
            )
            return torch.cat([ego_value, bg_value], dim=1)

        return TensorRolloutState(
            **{name: combine(name) for name in ego.__dataclass_fields__}
        )

    def _evaluate_rollout_step(
        self, scenario: ScenarioData
    ) -> tuple[torch.Tensor, torch.Tensor]:
        total_rewards, dones, infos, _ = self.reward_engine._calculate_rewards(
            scenario, scenario
        )
        if getattr(self.reward_engine, "_next_goal_stage", None) is not None:
            scenario.goal_stage = self.reward_engine._next_goal_stage
        goal_reached = infos[..., InfoDimension.GOAL_REACHED_FLAG.value].bool()
        scenario.update_goal_reached(goal_reached)
        return total_rewards[:, 0].float(), dones[:, 0]

    def _advance_background_ctra(self, state: TensorRolloutState) -> TensorRolloutState:
        position = state.positions[:, :, -1]
        raw_velocity = state.velocities[:, :, -1]
        velocity = torch.where(
            torch.isfinite(raw_velocity).all(dim=-1, keepdim=True),
            raw_velocity,
            torch.zeros_like(raw_velocity),
        )
        raw_yaw = state.orientations[:, :, -1]
        yaw = torch.where(torch.isfinite(raw_yaw), raw_yaw, torch.zeros_like(raw_yaw))
        speed = torch.linalg.vector_norm(velocity, dim=-1)
        raw_yaw_rate = state.yaw_rates[:, :, -1]
        yaw_rate = torch.where(
            torch.isfinite(raw_yaw_rate), raw_yaw_rate, torch.zeros_like(raw_yaw_rate)
        )
        wheelbase = get_wheelbase_from_length(state.sizes[:, :, -1, 0])
        max_yaw_rate = speed * math.tan(self.max_steering_angle) / wheelbase
        yaw_rate = torch.clamp(yaw_rate, min=-max_yaw_rate, max=max_yaw_rate)
        if state.velocities.shape[2] >= 2:
            previous_velocity = state.velocities[:, :, -2]
            finite_history = torch.isfinite(raw_velocity).all(dim=-1) & torch.isfinite(
                previous_velocity
            ).all(dim=-1)
            previous_speed = torch.linalg.vector_norm(previous_velocity, dim=-1)
            acceleration = (speed - previous_speed) / self.dt
            acceleration = torch.where(
                finite_history & torch.isfinite(acceleration),
                acceleration,
                torch.zeros_like(acceleration),
            ).clamp(-self.max_acceleration, self.max_acceleration)
        else:
            acceleration = torch.zeros_like(speed)

        moving = speed >= 0.2
        next_speed_raw = speed + acceleration * self.dt
        stopping = moving & (acceleration < 0.0) & (next_speed_raw < 0.0)
        stop_dt = speed / torch.clamp(-acceleration, min=torch.finfo(speed.dtype).eps)
        motion_dt = torch.where(stopping, stop_dt, torch.full_like(speed, self.dt))
        next_speed = torch.where(
            stopping,
            torch.zeros_like(speed),
            next_speed_raw.clamp(min=0.0),
        )
        heading = torch.stack([torch.cos(yaw), torch.sin(yaw)], dim=-1)
        distance = (speed * motion_dt + 0.5 * acceleration * motion_dt.square()).clamp(
            min=0.0
        )
        next_yaw = wrap_angle(yaw + yaw_rate * motion_dt)
        next_heading = torch.stack([torch.cos(next_yaw), torch.sin(next_yaw)], dim=-1)
        straight_position = position + heading * distance.unsqueeze(-1)

        safe_yaw_rate = torch.where(
            yaw_rate.abs() >= 1.0e-4, yaw_rate, torch.ones_like(yaw_rate)
        )
        inverse_yaw_rate = safe_yaw_rate.reciprocal()
        inverse_yaw_rate_sq = inverse_yaw_rate.square()
        sin_yaw = torch.sin(yaw)
        cos_yaw = torch.cos(yaw)
        sin_next_yaw = torch.sin(next_yaw)
        cos_next_yaw = torch.cos(next_yaw)
        curved_dx = speed * (
            sin_next_yaw - sin_yaw
        ) * inverse_yaw_rate + acceleration * (
            motion_dt * sin_next_yaw * inverse_yaw_rate
            + (cos_next_yaw - cos_yaw) * inverse_yaw_rate_sq
        )
        curved_dy = speed * (
            cos_yaw - cos_next_yaw
        ) * inverse_yaw_rate + acceleration * (
            -motion_dt * cos_next_yaw * inverse_yaw_rate
            + (sin_next_yaw - sin_yaw) * inverse_yaw_rate_sq
        )
        curved_position = position + torch.stack([curved_dx, curved_dy], dim=-1)
        turning = yaw_rate.abs() >= 1.0e-4
        accelerated_position = torch.where(
            turning.unsqueeze(-1), curved_position, straight_position
        )
        accelerated_velocity = next_heading * next_speed.unsqueeze(-1)
        zero_acceleration = acceleration.abs() < 1.0e-6
        preserve_linear_velocity = zero_acceleration & ~turning
        next_position = torch.where(
            preserve_linear_velocity.unsqueeze(-1),
            position + velocity * self.dt,
            accelerated_position,
        )
        next_velocity = torch.where(
            preserve_linear_velocity.unsqueeze(-1), velocity, accelerated_velocity
        )
        next_position = torch.where(moving.unsqueeze(-1), next_position, position)
        next_velocity = torch.where(
            moving.unsqueeze(-1), next_velocity, torch.zeros_like(next_velocity)
        )
        next_yaw = torch.where(moving, next_yaw, yaw)
        next_yaw_rate = torch.where(
            moving & (next_speed >= 0.2), yaw_rate, torch.zeros_like(yaw_rate)
        )
        active = state.agent_masks[:, :, -1].clone()
        active[:, 0] = False
        next_position = torch.where(active.unsqueeze(-1), next_position, position)
        next_velocity = torch.where(active.unsqueeze(-1), next_velocity, velocity)
        next_yaw = torch.where(active, next_yaw, yaw)
        next_acceleration = torch.where(
            active, acceleration, state.accelerations[:, :, -1]
        )
        next_yaw_rate = torch.where(active, next_yaw_rate, state.yaw_rates[:, :, -1])

        def append(value: torch.Tensor, latest: torch.Tensor) -> torch.Tensor:
            return torch.cat([value, latest.unsqueeze(2)], dim=2)

        return TensorRolloutState(
            positions=append(state.positions, next_position),
            velocities=append(state.velocities, next_velocity),
            sizes=append(state.sizes, state.sizes[:, :, -1]),
            orientations=append(state.orientations, next_yaw),
            agent_types=append(state.agent_types, state.agent_types[:, :, -1]),
            accelerations=append(state.accelerations, next_acceleration),
            acceleration_controls=append(
                state.acceleration_controls, next_acceleration
            ),
            yaw_rates=append(state.yaw_rates, next_yaw_rate),
            steering_angles=append(
                state.steering_angles, state.steering_angles[:, :, -1]
            ),
            steering_controls=append(
                state.steering_controls, state.steering_controls[:, :, -1]
            ),
            jerk_lat=append(state.jerk_lat, torch.zeros_like(next_yaw_rate)),
            jerk_long=append(
                state.jerk_long,
                (next_acceleration - state.accelerations[:, :, -1]) / self.dt,
            ),
            agent_masks=append(state.agent_masks, state.agent_masks[:, :, -1]),
            frame_drop_mask=append(
                state.frame_drop_mask,
                torch.ones_like(state.frame_drop_mask[:, :, -1]),
            ),
        )

    def _update_scenario_from_state(
        self, scenario: ScenarioData, state: TensorRolloutState
    ) -> None:
        previous_steps = scenario.agent_positions_all.shape[2]
        mapping = {
            "agent_positions_all": state.positions,
            "agent_velocity_all": state.velocities,
            "agent_size_all": state.sizes,
            "agent_orientation_all": state.orientations,
            "agent_type_all": state.agent_types,
            "agent_acceleration_state_all": state.accelerations,
            "agent_acceleration_control_all": state.acceleration_controls,
            "agent_yaw_rate_all": state.yaw_rates,
            "agent_steering_state_all": state.steering_angles,
            "agent_steering_control_all": state.steering_controls,
            "agent_jerk_lat_all": state.jerk_lat,
            "agent_jerk_long_all": state.jerk_long,
            "npc_mask_all": state.agent_masks,
            "frame_drop_mask_all": state.frame_drop_mask,
        }
        for name, value in mapping.items():
            setattr(scenario, name, value)

        current_steps = state.positions.shape[2]
        if current_steps == previous_steps:
            return
        if current_steps != previous_steps + 1:
            raise ValueError(
                "TTS rollout state must advance by exactly one frame, got "
                f"{previous_steps} -> {current_steps}."
            )
        if scenario.goal_positions.ndim != 4 or (
            scenario.goal_positions.shape[2] != previous_steps
        ):
            raise ValueError(
                "TTS goal history must align with state history before advancing, got "
                f"{tuple(scenario.goal_positions.shape)} and {previous_steps} steps."
            )

        num_goal_positions = scenario.goal_positions.shape[-1] // 2
        if num_goal_positions < 1:
            raise ValueError("TTS rollout requires at least one route goal position.")
        ego_goal = route_goal_positions(
            state.positions[:, :1, -1],
            state.velocities[:, :1, -1],
            scenario.route_points_array,
            scenario.route_points_mask,
            horizon_s=self.route_goal_horizon_s,
            min_speed_mps=self.route_goal_min_speed_mps,
            num_goal_positions=num_goal_positions,
        )
        ego_goal = apply_goal_pair_mode(ego_goal, self.route_goal_pair_mode)
        next_goal = scenario.goal_positions[:, :, -1:].clone()
        next_goal[:, :1, 0] = ego_goal
        scenario.goal_positions = torch.cat([scenario.goal_positions, next_goal], dim=2)

    @staticmethod
    def _state_from_scenario(source: ScenarioData) -> TensorRolloutState:
        return TensorRolloutState(
            positions=source.agent_positions_all,
            velocities=source.agent_velocity_all,
            sizes=source.agent_size_all,
            orientations=source.agent_orientation_all,
            agent_types=source.agent_type_all,
            accelerations=source.agent_acceleration_state_all,
            acceleration_controls=source.agent_acceleration_control_all,
            yaw_rates=source.agent_yaw_rate_all,
            steering_angles=source.agent_steering_state_all,
            steering_controls=source.agent_steering_control_all,
            jerk_lat=source.agent_jerk_lat_all,
            jerk_long=source.agent_jerk_long_all,
            agent_masks=source.npc_mask_all,
            frame_drop_mask=source.frame_drop_mask_all,
        )

    def _repeat_scenario(self, source: ScenarioData) -> ScenarioData:
        result = copy.copy(source)
        batch = source.agent_positions_all.shape[0]
        if batch != 1:
            raise ValueError(f"nuPlan TTS expects batch size 1, got {batch}.")
        for name, value in vars(source).items():
            if (
                name not in _PACKED_ROUTE_CONNECTOR_FIELDS
                and name != "_agent_control_manager"
                and isinstance(value, torch.Tensor)
                and value.ndim > 0
                and value.shape[0] == 1
            ):
                setattr(
                    result,
                    name,
                    value.repeat_interleave(self.num_candidates, dim=0),
                )
        self._repeat_agent_control_manager(source, result)
        self._repeat_packed_route_connectors(source, result)
        randomized = getattr(source, "_randomized_features", None)
        if randomized is not None:
            result._randomized_features = RandomizedFeatureDict(
                randomized.config,
                {
                    name: value.repeat_interleave(self.num_candidates, dim=0)
                    for name, value in randomized.values.items()
                },
            )
        return result

    def _repeat_agent_control_manager(
        self, source: ScenarioData, result: ScenarioData
    ) -> None:
        manager = getattr(source, "_agent_control_manager", None)
        if manager is None:
            return
        if manager._control_types.shape[0] != 1:
            raise ValueError(
                "nuPlan TTS expects AgentControlManager batch size 1, got "
                f"{manager._control_types.shape[0]}."
            )
        repeated = AgentControlManager(
            self.num_candidates,
            manager.max_agents,
            manager.device,
        )
        repeated._control_types = manager._control_types.repeat_interleave(
            self.num_candidates, dim=0
        )
        result._agent_control_manager = repeated

    def _repeat_packed_route_connectors(
        self, source: ScenarioData, result: ScenarioData
    ) -> None:
        required = _PACKED_ROUTE_CONNECTOR_FIELDS - {
            "route_lane_connector_data_available"
        }
        if not all(hasattr(source, name) for name in required):
            return

        vertices = source.route_lane_connector_polygon_vertices
        offsets = source.route_lane_connector_polygon_offsets
        bboxes = source.route_lane_connector_bboxes
        status_codes = source.route_lane_connector_tl_status_codes
        connector_ids = source.route_lane_connector_ids
        batch_indices = source.route_lane_connector_batch_indices
        world_offsets = source.route_lane_connector_world_offsets
        connector_count = connector_ids.numel()
        vertex_count = vertices.shape[0]
        if (
            connector_count == 0
            and vertex_count == 0
            and offsets.numel() == 0
            and world_offsets.numel() == 0
        ):
            return
        if offsets.ndim != 1 or offsets.numel() != connector_count + 1:
            raise ValueError("Invalid packed route connector polygon offsets for TTS.")
        if int(offsets[-1].item()) != vertex_count:
            raise ValueError(
                "Packed route connector offsets do not cover all vertices."
            )
        if batch_indices.shape != (connector_count,) or bool(
            (batch_indices != 0).any()
        ):
            raise ValueError(
                "nuPlan TTS expects packed route connectors for batch 0 only."
            )
        if world_offsets.shape != (2,) or not torch.equal(
            world_offsets.to(torch.int64),
            torch.tensor(
                [0, connector_count], device=world_offsets.device, dtype=torch.int64
            ),
        ):
            raise ValueError("Invalid packed route connector world offsets for TTS.")

        result.route_lane_connector_polygon_vertices = vertices.repeat(
            self.num_candidates, 1
        )
        result.route_lane_connector_polygon_offsets = torch.cat(
            [
                offsets[:1],
                *[
                    offsets[1:] + repeat_index * vertex_count
                    for repeat_index in range(self.num_candidates)
                ],
            ]
        )
        result.route_lane_connector_bboxes = bboxes.repeat(self.num_candidates, 1)
        result.route_lane_connector_tl_status_codes = status_codes.repeat(
            self.num_candidates, 1
        )
        result.route_lane_connector_ids = connector_ids.repeat(self.num_candidates)
        result.route_lane_connector_batch_indices = torch.arange(
            self.num_candidates,
            device=batch_indices.device,
            dtype=batch_indices.dtype,
        ).repeat_interleave(connector_count)
        result.route_lane_connector_world_offsets = (
            torch.arange(
                self.num_candidates + 1,
                device=world_offsets.device,
                dtype=world_offsets.dtype,
            )
            * connector_count
        )
        available = getattr(source, "route_lane_connector_data_available", None)
        if isinstance(available, torch.Tensor):
            if available.shape != (1,):
                raise ValueError(
                    "nuPlan TTS expects route connector availability shape [1]."
                )
            result.route_lane_connector_data_available = available.repeat(
                self.num_candidates
            )
