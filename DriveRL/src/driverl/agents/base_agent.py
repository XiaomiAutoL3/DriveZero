"""
This module defines the base classes for all agents.
"""

import abc
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Beta, Categorical

from driverl.agents.config import AgentConfig
from driverl.datatypes.goal_position_utils import normalize_goal_positions
from driverl.datatypes.scenario_data import ScenarioData
from driverl.env.constants import EXPECTATION_SPEED_THRESHOLD
from driverl.utils.geometry import (
    rotate_2d_points,
    translate_and_rotate_2d_points,
    wrap_angle,
)
from driverl.utils.logging import logger
from driverl.utils.registry import Registry

__all__ = ["BaseAgent", "LearningAgent", "AGENT_REGISTER"]


AGENT_REGISTER = Registry("agent")

AgentForwardOutput = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


def _history_slice(history_steps: int, history_frame_gap: int) -> slice:
    return slice(
        -((history_steps - 1) * history_frame_gap + 1),
        None,
        history_frame_gap,
    )


def _history_indices(
    step_indices: torch.Tensor, history_steps: int, history_frame_gap: int
) -> torch.Tensor:
    offsets = (
        torch.arange(history_steps, device=step_indices.device) * history_frame_gap
    )
    return step_indices[:, None] - (history_steps - 1) * history_frame_gap + offsets


class BaseAgent(abc.ABC):
    """Base class for all agents."""

    def __init__(self, config: AgentConfig):
        """Initialize the agent."""
        self.config = config

    @staticmethod
    def agent_factory(agent_name: str, *args, **kwargs) -> "BaseAgent":
        """Factory method for agents."""
        agent_name = Registry.get_model_instance_name(agent_name)
        supported_agent_names = AGENT_REGISTER.module_keys
        assert agent_name in supported_agent_names, (
            f"Currently only support {supported_agent_names}, but got {agent_name}."
        )
        return AGENT_REGISTER.get(agent_name)(*args, **kwargs)

    @abc.abstractmethod
    def get_action(self, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    @abc.abstractmethod
    def __call__(
        self,
        scenario_data: ScenarioData,
        action: Optional[torch.Tensor] = None,
        env_indices: Optional[Union[np.ndarray, torch.Tensor]] = None,
        step_indices: Optional[Union[np.ndarray, torch.Tensor]] = None,
        sampling_method: str = "sample",
        log_scenario_data: ScenarioData | None = None,
    ) -> AgentForwardOutput:
        """Run the policy and return (action, log_prob, entropy, value)."""
        raise NotImplementedError

    @staticmethod
    def features_to_vis_inputs(
        features: tuple,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """Convert model features tuple into a visualizer-friendly dict.

        Returns ``(inputs_dict, training_mask)``.
        ``inputs_dict`` must contain the keys expected by
        ``calculate_model_input`` in *calculate.py*:
        ``vehicle_state_features``, ``other_agents_features``,
        ``other_agents_mask``, ``lane_features``, ``lane_mask``.
        All value tensors are packed as ``(N, ...)``.
        Subclasses must override this to match their feature tuple layout.
        """
        raise NotImplementedError(
            "Subclass must implement features_to_vis_inputs for visualizer support."
        )


class LearningAgent(BaseAgent, nn.Module):
    """Base class for learning-based agents that use neural networks."""

    def __init__(self, config: AgentConfig):
        BaseAgent.__init__(self, config)
        nn.Module.__init__(self)
        self.model: Optional[nn.Module] = None
        # Flag indicating whether the agent uses a continuous action space.
        # Subclasses should set this to True for continuous (Beta) policies.
        self.is_continuous: bool = False
        # Action bounds for continuous policies (linear rescaling of Beta samples).
        # Subclasses should register these via register_buffer when using Box spaces.
        self.register_buffer("_action_low", None, persistent=False)
        self.register_buffer("_action_high", None, persistent=False)

    def forward(self, *args, **kwargs) -> AgentForwardOutput:
        return self.__call__(*args, **kwargs)

    def compile_agent(
        self,
        *,
        fullgraph: bool = True,
        dynamic: bool = False,
    ):
        """
        Optionally compile preprocess/model paths. Keep preprocessing separate to
        tolerate dynamic masking while still compiling the stable forward.
        Call after self.model is initialized.
        """
        self.forward = torch.compile(  # type: ignore[method-assign]
            self.forward,
            fullgraph=fullgraph,
            dynamic=dynamic,
        )

    @abc.abstractmethod
    def _preprocess(
        self,
        scenario_data: ScenarioData,
        env_indices: Optional[Union[np.ndarray, torch.Tensor]] = None,
        step_indices: Optional[Union[np.ndarray, torch.Tensor]] = None,
        log_scenario_data: ScenarioData | None = None,
    ):
        """Prepare raw scenario data into model-ready tensors."""
        raise NotImplementedError

    def preprocess(
        self,
        scenario_data: ScenarioData,
        env_indices: Optional[Union[np.ndarray, torch.Tensor]] = None,
        step_indices: Optional[Union[np.ndarray, torch.Tensor]] = None,
        log_scenario_data: ScenarioData | None = None,
    ):
        """Public preprocessing entry point wrapping the subclass hook in no_grad."""
        with torch.no_grad():
            if log_scenario_data is None:
                return self._preprocess(scenario_data, env_indices, step_indices)
            return self._preprocess(
                scenario_data,
                log_scenario_data=log_scenario_data,
                env_indices=env_indices,
                step_indices=step_indices,
            )

    def forward_with_features(
        self,
        features,
        action: torch.Tensor | None = None,
        sampling_method: str = "sample",
        action_keys_tensor: torch.Tensor | None = None,
        current_speed: torch.Tensor | None = None,
    ):
        """Forward pass through the model given preprocessed features.

        Returns:
            ``(action, log_prob, entropy, value, action_params)`` where
            ``action_params`` is the (expanded) raw distribution parameters
            (logits for discrete, Beta params for continuous) with shape
            ``[B, A, ...]``.
        """
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

    def forward_policy_params(self, features) -> tuple[torch.Tensor, torch.Tensor]:
        """Run only the neural network and return action parameters and values."""
        if self.model is None:
            raise ValueError("LearningAgent requires 'self.model' to be defined.")
        return self.model(*features)

    def select_action_from_params(
        self,
        action_params: torch.Tensor,
        *,
        sampling_method: str,
        action_keys_tensor: torch.Tensor | None = None,
        current_speed: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Select an action from policy parameters without rerunning the model."""
        dummy_values = torch.zeros(
            action_params.shape[:-1], device=action_params.device
        )
        action, _, _, _ = self._policy_outputs(
            action_params,
            dummy_values,
            None,
            sampling_method,
            action_keys_tensor,
            current_speed=current_speed,
        )
        return action

    def _build_continuous_distribution(
        self, alpha: torch.Tensor, beta: torch.Tensor
    ) -> Beta:
        """Create a Beta distribution from the network outputs.

        The Beta distribution is naturally bounded on [0, 1] and does not
        require tanh squashing or Jacobian corrections.  Actions are linearly
        rescaled to [action_low, action_high] after sampling.

        Args:
            alpha: Concentration parameter α (> 0), shape ``[..., action_dim]``.
            beta:  Concentration parameter β (> 0), shape ``[..., action_dim]``.
        """
        return Beta(alpha, beta)

    def _build_action_distribution(self, action_params: torch.Tensor):
        """Create the action distribution.

        For continuous policies, ``action_params`` has shape ``[..., 2 * action_dim]``
        containing concatenated alpha and beta parameters for a Beta distribution.
        For discrete policies, ``action_params`` is logits ``[..., num_actions]``.
        """
        if self.is_continuous:
            action_dim = action_params.shape[-1] // 2
            alpha = action_params[..., :action_dim]
            beta = action_params[..., action_dim:]
            return self._build_continuous_distribution(alpha, beta)
        return Categorical(logits=action_params)

    def _policy_outputs(
        self,
        action_params: torch.Tensor,
        values: torch.Tensor,
        action: torch.Tensor | None,
        sampling_method: str,
        action_keys_tensor: torch.Tensor | None,
        current_speed: torch.Tensor | None = None,
    ):
        """Sample actions and compute log probs/entropy.

        Supports both continuous (Beta) and discrete (Categorical) policies.
        For both policy types, ``sampling_method="expectation"`` is only used
        below ``EXPECTATION_SPEED_THRESHOLD``; above that it falls back to
        ``argmax`` semantics.
        """
        if self.is_continuous:
            action_dim = action_params.shape[-1] // 2
            alpha = action_params[..., :action_dim]
            beta = action_params[..., action_dim:]
            return self._continuous_policy_outputs(
                alpha,
                beta,
                values,
                action,
                sampling_method,
                current_speed=current_speed,
            )
        return self._discrete_policy_outputs(
            action_params,
            values,
            action,
            sampling_method,
            action_keys_tensor,
            current_speed=current_speed,
        )

    # -- Beta distribution helpers for bounded continuous actions -------------

    def _rescale_action(self, unit_action: torch.Tensor) -> torch.Tensor:
        """Linearly rescale action from [0, 1] to [action_low, action_high]."""
        return self._action_low + unit_action * (self._action_high - self._action_low)

    def _normalize_action(self, action: torch.Tensor) -> torch.Tensor:
        """Linearly normalize action from [action_low, action_high] to (0, 1)."""
        normalized = (action - self._action_low) / (
            self._action_high - self._action_low
        )
        # Clamp to open interval (eps, 1-eps) to avoid log(0) in Beta log_prob
        return normalized.clamp(1e-6, 1.0 - 1e-6)

    # -----------------------------------------------------------------------

    def _continuous_policy_outputs(
        self,
        alpha: torch.Tensor,
        beta: torch.Tensor,
        values: torch.Tensor,
        action: torch.Tensor | None,
        sampling_method: str,
        current_speed: torch.Tensor | None = None,
    ):
        """Policy outputs for Beta distribution over bounded continuous actions.

        The Beta distribution is naturally supported on [0, 1]. Samples are
        linearly rescaled to [action_low, action_high]. Sampling semantics:
        - ``argmax``: use the Beta mode.
        - ``expectation``: use the Beta mean below the speed threshold and
          fall back to the Beta mode above it.
          This is applied jointly to every continuous action dimension
          (e.g. longitudinal and lateral are both taken from the Beta mean).
        - ``sample``: draw a stochastic sample.
        """
        # Guard against invalid parameters (e.g. zeros from masked_write_back
        # for non-controlled agents).  Beta requires α > 0 and β > 0.
        alpha = alpha.clamp(min=1e-2)
        beta = beta.clamp(min=1e-2)
        dist = self._build_continuous_distribution(alpha, beta)
        mean_action = alpha / (alpha + beta)
        raw_mode_action = (alpha - 1.0) / (alpha + beta - 2.0).clamp(min=1e-6)
        valid_mode = (alpha > 1.0) & (beta > 1.0)
        mode_action = torch.where(valid_mode, raw_mode_action, mean_action)

        if action is not None:
            # Re-evaluation: normalize bounded action back to (0, 1)
            unit_action = self._normalize_action(action)
        elif sampling_method == "argmax":
            unit_action = mode_action
        elif sampling_method == "expectation":
            unit_action = mean_action
            if current_speed is not None and current_speed.shape == alpha.shape[:-1]:
                use_expectation = (
                    current_speed < EXPECTATION_SPEED_THRESHOLD
                ).unsqueeze(-1)
                unit_action = torch.where(use_expectation, mean_action, mode_action)
        else:  # "sample"
            unit_action = dist.sample()

        unit_action = unit_action.clamp(1e-6, 1.0 - 1e-6)

        # Log-prob of the unit-interval sample.
        # The affine rescaling a = low + u * (high - low) has constant Jacobian
        # |da/du| = (high - low), so log_prob(a) = log_prob(u) - log(high - low).
        # Include the action-scale correction in the continuous log probability.
        # the constant cancels in the importance ratio.
        log_prob = dist.log_prob(unit_action).sum(dim=-1)

        entropy = dist.entropy().sum(dim=-1)

        if action is None:
            action = self._rescale_action(unit_action)

        return action, log_prob, entropy, values

    def _discrete_policy_outputs(
        self,
        action_logits: torch.Tensor,
        values: torch.Tensor,
        action: torch.Tensor | None,
        sampling_method: str,
        action_keys_tensor: torch.Tensor | None,
        current_speed: torch.Tensor | None = None,
    ):
        """Policy outputs for discrete Categorical distribution (legacy)."""
        dist = Categorical(logits=action_logits)

        if action is not None:
            log_prob = dist.log_prob(action)
            entropy = dist.entropy()
            return action, log_prob, entropy, values

        if sampling_method == "expectation":
            probs = dist.probs  # [..., N]
            if action_keys_tensor is None:
                raise ValueError(
                    "sampling_method='expectation' requires action_keys_tensor."
                )
            action_keys = action_keys_tensor.to(device=probs.device, dtype=probs.dtype)
            argmax_action = probs.argmax(dim=-1)  # [...]
            longitudinal = action_keys[argmax_action, 0]  # [...]
            # Discrete expectation is asymmetric by design:
            # keep longitudinal at argmax, but use a probability-weighted
            # expectation only for the lateral token, then snap back to the
            # nearest discrete action key.
            expected_lateral = (probs * action_keys[:, 1]).sum(dim=-1)  # [...]
            # Compose target and find nearest action key
            target = torch.stack([longitudinal, expected_lateral], dim=-1)  # [..., 2]
            diffs = target.unsqueeze(-2) - action_keys  # [..., N, 2]
            expectation_action = diffs.pow(2).sum(-1).argmin(-1)  # [...]

            action = expectation_action
            if (
                current_speed is not None
                and current_speed.shape == action_logits.shape[:-1]
            ):
                use_expectation = current_speed < EXPECTATION_SPEED_THRESHOLD
                action = torch.where(use_expectation, expectation_action, argmax_action)
        elif sampling_method == "argmax":
            action = torch.argmax(action_logits, dim=-1)
        else:  # "sample"
            action = dist.sample()

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        return action, log_prob, entropy, values

    def _filter_agents_behind_ego(
        self,
        positions: torch.Tensor,
        agent_mask: torch.Tensor,
        threshold: float = 1.5,
    ) -> torch.Tensor:
        """
        Filter agents that are exactly behind the ego agent.
        """
        lateral_dist = torch.abs(positions[..., 1])
        valid_mask = agent_mask.bool()
        lateral_and_behind = (lateral_dist < threshold) & (positions[..., 0] < -2.0)
        mask_out = (valid_mask[..., -2:] & lateral_and_behind[..., -2:]).any(dim=-1)
        return agent_mask & ~mask_out.unsqueeze(-1)

    def __call__(
        self,
        scenario_data: ScenarioData,
        action: torch.Tensor | None = None,
        env_indices: np.ndarray | torch.Tensor | None = None,
        step_indices: np.ndarray | torch.Tensor | None = None,
        sampling_method: str = "sample",
        action_keys_tensor: torch.Tensor | None = None,
        log_scenario_data: ScenarioData | None = None,
    ) -> AgentForwardOutput:
        """Run the policy and return (action, log_prob, entropy, value)."""
        features = self.preprocess(
            scenario_data,
            log_scenario_data=log_scenario_data,
            env_indices=env_indices,
            step_indices=step_indices,
        )
        current_speed = torch.linalg.norm(
            scenario_data.agent_velocity_all[:, :, -1, :], dim=-1
        )
        act, log_prob, entropy, val, _action_params = self.forward_with_features(
            features,
            action=action,
            sampling_method=sampling_method,
            action_keys_tensor=action_keys_tensor,
            current_speed=current_speed,
        )
        return act, log_prob, entropy, val

    def get_action(self, **kwargs) -> torch.Tensor:
        """Returns an action based on the model's output logits and sampling method."""
        scenario_data: ScenarioData | None = kwargs.get("scenario_data")
        if scenario_data is None:
            raise ValueError("LearningAgent.get_action requires 'scenario_data'.")

        sampling_method: str = kwargs.get("sampling_method", "sample")
        env_indices = kwargs.get("env_indices", None)
        step_indices = kwargs.get("step_indices", None)

        action, _, _, _ = self(
            scenario_data,
            log_scenario_data=kwargs.get("log_scenario_data"),
            env_indices=env_indices,
            step_indices=step_indices,
            sampling_method=sampling_method,
        )
        return action

    def _load_state_dict_flexible(
        self, state_dict: dict
    ) -> tuple[list[str], list[str]]:
        model_state = self.state_dict()
        filtered_state = {
            k: v
            for k, v in state_dict.items()
            if k in model_state and v.shape == model_state[k].shape
        }
        missing, unexpected = self.load_state_dict(filtered_state, strict=False)
        if missing or unexpected:
            logger.warning(
                "[load_checkpoint] Skipped loading mismatched keys. "
                "missing=%s, unexpected=%s",
                len(missing),
                len(unexpected),
            )
        return list(missing), list(unexpected)

    def slice_history(
        self,
        scenario_data: ScenarioData,
        env_indices,
        step_indices,
        log_scenario_data: ScenarioData | None = None,
    ):
        history_steps = self.config.history_steps
        history_frame_gap = getattr(self.config, "history_frame_gap", 1)
        advantage_filtering_mask = None
        agent_velocity_all = getattr(scenario_data, "agent_velocity_all", None)
        if agent_velocity_all is None:
            # Lightweight adapters may omit velocity when only coordinate
            # transforms are needed. A zero vector preserves the transform
            # contract while keeping the normal ScenarioData path unchanged.
            agent_velocity_all = torch.zeros_like(scenario_data.agent_positions_all)

        def pad_tensor_front(tensor, dim=2):
            """
            Padding with zeros along the dim (time) dimension until reaching history_steps length.
            tensor: shape = [B, A, T, ...]
            """
            current_len = tensor.shape[dim]
            if current_len >= history_steps:
                return tensor

            pad_shape = list(tensor.shape)
            pad_shape[dim] = history_steps - current_len
            padding = torch.zeros(pad_shape, dtype=tensor.dtype, device=tensor.device)

            return torch.cat([padding, tensor], dim=dim)

        if env_indices is None or step_indices is None:
            env_idx = slice(None)
            step_idx = _history_slice(history_steps, history_frame_gap)
            # Extract data using the determined indices.
            positions = scenario_data.agent_positions_all[env_idx, :, step_idx, :]
            velocities = agent_velocity_all[env_idx, :, step_idx, :]
            orientations = scenario_data.agent_orientation_all[env_idx, :, step_idx]
            sizes = scenario_data.agent_size_all[env_idx, :, step_idx, :].clone()
            types = scenario_data.agent_type_all[env_idx, :, step_idx]
            agent_mask = scenario_data.npc_mask_all[env_idx, :, step_idx]
            steering_state = scenario_data.agent_steering_state_all[
                env_idx, :, step_idx
            ]
            steering_control = scenario_data.agent_steering_control_all[
                env_idx, :, step_idx
            ]
            acceleration_state = scenario_data.agent_acceleration_state_all[
                env_idx, :, step_idx
            ]
            acceleration_control = scenario_data.agent_acceleration_control_all[
                env_idx, :, step_idx
            ]
            goal_reached = scenario_data.agent_goal_reached_all[env_idx, :, -1]
            occ_points = scenario_data.occ_surface_points_all[env_idx, 0, -1, :, :]

            if scenario_data.agent_visible_mask.numel() > 0:
                visible_mask = scenario_data.agent_visible_mask[env_idx, :, step_idx, :]
                visible_mask = pad_tensor_front(visible_mask)
            else:
                visible_mask = None

            # Extract angular velocity (yaw rate) if available
            if scenario_data.agent_yaw_rate_all.numel() > 0:
                yaw_rates = scenario_data.agent_yaw_rate_all[env_idx, :, step_idx]
            else:
                yaw_rates = torch.zeros_like(acceleration_state)
            frame_drop_mask = scenario_data.frame_drop_mask_all[env_idx, :, step_idx]

        else:
            env_idx = torch.as_tensor(env_indices, device=self.config.device)
            step_idx = torch.as_tensor(step_indices, device=self.config.device)

            B = len(env_idx)

            env_idx_2d = env_idx[:, None].expand(B, history_steps)  # [B, history_steps]
            history_idx_2d = _history_indices(
                step_idx, history_steps, history_frame_gap
            )  # [B, history_steps]

            valid_mask = history_idx_2d >= 0
            valid_mask_3d = valid_mask[:, None, :]  # [B, 1, history_steps]
            valid_mask_4d = valid_mask[:, None, :, None]  # [B, 1, history_steps, 1]

            clamped_history_idx_2d = history_idx_2d.clamp(min=0)

            # Extract data using the determined indices.
            positions = (
                scenario_data.agent_positions_all[
                    env_idx_2d, :, clamped_history_idx_2d, :
                ].permute(0, 2, 1, 3)
                * valid_mask_4d
            )

            velocities = (
                agent_velocity_all[
                    env_idx_2d, :, clamped_history_idx_2d, :
                ].permute(0, 2, 1, 3)
                * valid_mask_4d
            )

            orientations = (
                scenario_data.agent_orientation_all[
                    env_idx_2d, :, clamped_history_idx_2d
                ].permute(0, 2, 1)
                * valid_mask_3d
            )

            sizes = (
                scenario_data.agent_size_all[
                    env_idx_2d, :, clamped_history_idx_2d, :
                ].permute(0, 2, 1, 3)
                * valid_mask_4d
            ).clone()

            types = (
                scenario_data.agent_type_all[
                    env_idx_2d, :, clamped_history_idx_2d
                ].permute(0, 2, 1)
                * valid_mask_3d
            )

            agent_mask = (
                scenario_data.npc_mask_all[
                    env_idx_2d, :, clamped_history_idx_2d
                ].permute(0, 2, 1)
                * valid_mask_3d
            )

            steering_state = (
                scenario_data.agent_steering_state_all[
                    env_idx_2d, :, clamped_history_idx_2d
                ].permute(0, 2, 1)
                * valid_mask_3d
            )

            steering_control = (
                scenario_data.agent_steering_control_all[
                    env_idx_2d, :, clamped_history_idx_2d
                ].permute(0, 2, 1)
                * valid_mask_3d
            )

            acceleration_state = (
                scenario_data.agent_acceleration_state_all[
                    env_idx_2d, :, clamped_history_idx_2d
                ].permute(0, 2, 1)
                * valid_mask_3d
            )

            acceleration_control = (
                scenario_data.agent_acceleration_control_all[
                    env_idx_2d, :, clamped_history_idx_2d
                ].permute(0, 2, 1)
                * valid_mask_3d
            )

            goal_reached = scenario_data.agent_goal_reached_all[env_idx, :, step_idx]

            if scenario_data.agent_visible_mask.numel() > 0:
                visible_mask = (
                    scenario_data.agent_visible_mask[
                        env_idx_2d, :, clamped_history_idx_2d, :
                    ].permute(0, 2, 1, 3)
                    * valid_mask_4d
                )
            else:
                visible_mask = None

            # Extract angular velocity (yaw rate) if available
            if scenario_data.agent_yaw_rate_all.numel() > 0:
                yaw_rates = (
                    scenario_data.agent_yaw_rate_all[
                        env_idx_2d, :, clamped_history_idx_2d
                    ].permute(0, 2, 1)
                    * valid_mask_3d
                )
            else:
                yaw_rates = torch.zeros_like(acceleration_state)

            # Extract mask for history frames
            frame_drop_mask = (
                scenario_data.frame_drop_mask_all[
                    env_idx_2d, :, clamped_history_idx_2d
                ].permute(0, 2, 1)
                * valid_mask_3d
            )

            if scenario_data.advantage_filtering_mask is not None:
                advantage_filtering_mask = scenario_data.advantage_filtering_mask[
                    env_idx, :, step_idx
                ]
            init_steps = (
                scenario_data.agent_positions_all.shape[2]
                - scenario_data.occ_surface_points_all.shape[2]
            )
            occ_points = scenario_data.occ_surface_points_all[
                env_idx, 0, step_idx - init_steps, :, :
            ]

        future_steps = int(getattr(self.config, "future_steps", 0))
        if future_steps:
            if log_scenario_data is None:
                raise ValueError("future_steps requires log_scenario_data.")
            if log_scenario_data.frame_drop_mask_all.shape != (
                log_scenario_data.npc_mask_all.shape
            ):
                raise ValueError(
                    "log_scenario_data.frame_drop_mask_all must cover the full log timeline."
                )

            device = scenario_data.agent_positions_all.device
            if env_indices is None or step_indices is None:
                selected_env = torch.arange(positions.shape[0], device=device)
                current_steps = torch.full(
                    (positions.shape[0],),
                    scenario_data.agent_positions_all.shape[2] - 1,
                    dtype=torch.long,
                    device=device,
                )
            else:
                selected_env = torch.as_tensor(env_indices, device=device)
                current_steps = torch.as_tensor(step_indices, device=device)

            future_indices = current_steps[:, None] + torch.arange(
                1, future_steps + 1, device=device
            )
            valid_future = future_indices < log_scenario_data.npc_mask_all.shape[2]
            clamped_future = future_indices.clamp(
                max=log_scenario_data.npc_mask_all.shape[2] - 1
            )
            env_grid = selected_env[:, None].expand_as(clamped_future)

            def gather_future(tensor: torch.Tensor) -> torch.Tensor:
                gathered = tensor[env_grid, :, clamped_future]
                return gathered.permute(0, 2, 1, *range(3, gathered.dim()))

            future_positions = gather_future(log_scenario_data.agent_positions_all)
            future_velocity_all = getattr(log_scenario_data, "agent_velocity_all", None)
            if future_velocity_all is None:
                future_velocity_all = torch.zeros_like(log_scenario_data.agent_positions_all)
            future_velocities = gather_future(future_velocity_all)
            future_orientations = gather_future(
                log_scenario_data.agent_orientation_all
            )
            future_sizes = gather_future(log_scenario_data.agent_size_all)
            future_types = gather_future(log_scenario_data.agent_type_all)
            future_mask = gather_future(log_scenario_data.npc_mask_all).bool()
            future_mask &= valid_future[:, None, :]
            future_agent_allowed = scenario_data.agent_control_manager.log_replay_mask[
                selected_env
            ]
            future_mask &= future_agent_allowed.unsqueeze(-1)
            future_frame_drop = gather_future(
                log_scenario_data.frame_drop_mask_all
            ).bool()
            future_frame_drop &= valid_future[:, None, :]

            positions = torch.cat([positions, future_positions], dim=2)
            velocities = torch.cat([velocities, future_velocities], dim=2)
            orientations = torch.cat([orientations, future_orientations], dim=2)
            sizes = torch.cat([sizes, future_sizes], dim=2)
            types = torch.cat([types, future_types], dim=2)
            agent_mask = torch.cat([agent_mask, future_mask], dim=2)
            steering_state = torch.cat(
                [
                    steering_state,
                    gather_future(log_scenario_data.agent_steering_state_all),
                ],
                dim=2,
            )
            steering_control = torch.cat(
                [
                    steering_control,
                    gather_future(log_scenario_data.agent_steering_control_all),
                ],
                dim=2,
            )
            acceleration_state = torch.cat(
                [
                    acceleration_state,
                    gather_future(log_scenario_data.agent_acceleration_state_all),
                ],
                dim=2,
            )
            acceleration_control = torch.cat(
                [
                    acceleration_control,
                    gather_future(log_scenario_data.agent_acceleration_control_all),
                ],
                dim=2,
            )
            yaw_rates = torch.cat(
                [yaw_rates, gather_future(log_scenario_data.agent_yaw_rate_all)], dim=2
            )
            frame_drop_mask = torch.cat(
                [frame_drop_mask, future_frame_drop], dim=2
            )
            if visible_mask is not None:
                future_visible = future_mask.permute(0, 2, 1).unsqueeze(1).expand(
                    -1, future_mask.shape[1], -1, -1
                )
                visible_mask = torch.cat([visible_mask, future_visible], dim=2)

        # Read goal at the current (or pinned) frame, mirroring how
        # agent_goal_reached_all is read in both branches above. Using -1 for
        # the unpinned case keeps the graph shape constant under torch.compile
        # (no dependency on agent_positions_all.shape[2], which grows every
        # rollout step and would trigger recompiles).
        if env_indices is None or step_indices is None:
            goal_positions = scenario_data.goal_positions[env_idx, :, -1, :]
        else:
            goal_positions = scenario_data.goal_positions[env_idx, :, step_idx, :]
        enabled_randomized_features = (
            scenario_data.randomized_features.get_enabled_features()
        )
        if enabled_randomized_features is not None:
            enabled_randomized_features = enabled_randomized_features[env_idx]
        width_buffer = scenario_data.randomized_features.get("width_buffer")
        sizes[..., 1] += width_buffer[env_idx].unsqueeze(-1)

        not_blind_mask = scenario_data.agent_not_blind_mask[env_idx, :]

        # Lane geometry data
        lanes_points = scenario_data.lanes_points[env_idx]
        lanes_points_mask = scenario_data.lanes_points_mask[env_idx]
        if self.config.enable_traffic_light_features:
            lanes_tl_states = scenario_data.lanes_tl_states
            lanes_tl_masks = scenario_data.lanes_tl_masks
            if lanes_tl_states.numel() > 0 and lanes_tl_masks.numel() > 0:
                if env_indices is None or step_indices is None:
                    tl_state_now = lanes_tl_states[env_idx, :, -1, :]
                    tl_mask_now = lanes_tl_masks[env_idx, :, -1]
                else:
                    tl_state_now = lanes_tl_states[env_idx, :, step_idx, :]
                    tl_mask_now = lanes_tl_masks[env_idx, :, step_idx]
                tl_features = torch.where(
                    tl_mask_now.unsqueeze(-1),
                    tl_state_now,
                    torch.zeros_like(tl_state_now),
                )
                lanes_points = torch.cat(
                    [
                        lanes_points,
                        tl_features.unsqueeze(2).expand(
                            -1, -1, lanes_points.shape[2], -1
                        ),
                    ],
                    dim=-1,
                )
        return (
            positions,
            velocities,
            orientations,
            sizes,
            types,
            agent_mask,
            goal_positions,
            goal_reached,
            steering_state,
            steering_control,
            acceleration_state,
            acceleration_control,
            yaw_rates,
            lanes_points,
            lanes_points_mask,
            visible_mask,
            frame_drop_mask,
            advantage_filtering_mask,
            not_blind_mask,
            enabled_randomized_features,
            occ_points,
        )

    def coordinates_transformation(
        self,
        base_position: torch.Tensor,
        base_orientation: torch.Tensor,
        *tensors: torch.Tensor,
    ) -> list[torch.Tensor]:
        """
        Transform tensors into the ego-centric frame defined by base position/orientation.
        """
        if len(tensors) != 5:
            raise ValueError("coordinates_transformation expects 5 tensors.")

        (
            positions,  # [B, A, history_steps, 2]
            velocities,  # [B, A, history_steps, 2]
            orientations,  # [B, A, history_steps]
            goal_positions,  # [B, A, 2 * num_goals]
            lanes_points,
        ) = tensors
        goal_positions = normalize_goal_positions(goal_positions)
        goal_shape = goal_positions.shape[:-1]
        num_goal_positions = goal_positions.shape[-1] // 2
        goal_points = goal_positions.reshape(*goal_shape, num_goal_positions, 2)

        transformed = [
            translate_and_rotate_2d_points(
                positions,  # [B, A, history_steps, 2]
                base_position.unsqueeze(2),  # [B, A, 1, 2]
                base_orientation.unsqueeze(2),  # [B, A, 1]
            ),
            rotate_2d_points(
                velocities,  # [B, A, history_steps, 2]
                -base_orientation.unsqueeze(2),  # [B, A, 1]
            ),
            wrap_angle(orientations - base_orientation.unsqueeze(2)),
            translate_and_rotate_2d_points(
                goal_points,
                base_position.unsqueeze(-2),
                base_orientation.unsqueeze(-1),
            ).reshape(*goal_shape, num_goal_positions * 2),
        ]

        lanes_points_ego = lanes_points.clone()
        lanes_points_ego[..., 0:2] = translate_and_rotate_2d_points(
            lanes_points[..., 0:2],
            base_position.unsqueeze(1),
            base_orientation.unsqueeze(1),
        )
        transformed.append(lanes_points_ego)

        return transformed

    def transform_scenario_coordinates(
        self,
        scenario_data: ScenarioData,
        env_indices=None,
        step_indices=None,
        category: str = "global",
        log_scenario_data: ScenarioData | None = None,
    ):
        """
        Extracts tensors from scenario data and transforms them into specified coordinate frame.
        The ego agent is assumed to be agent at index 0 for ego-centric transformations.
        """
        # Determine the indices to use for slicing the data.
        # If no indices are provided, default to evaluation mode (last step of all envs).
        (
            positions,  # [B, A, history_steps, 2]
            velocities,  # [B, A, history_steps, 2]
            orientations,  # [B, A, history_steps]
            sizes,  # [B, A, history_steps, 2]
            types,
            agent_mask,  # [B, A, history_steps]
            goal_positions,  # [B, A, 2]
            goal_reached,  # [B, A]
            steering_state,  # [B, A, history_steps]
            steering_control,  # [B, A, history_steps]
            acceleration_state,  # [B, A, history_steps]
            acceleration_control,  # [B, A, history_steps]
            yaw_rates,  # [B, A, history_steps]
            lanes_points,
            lanes_points_mask,
            visible_mask,  # [B, A, history_steps, A]
            frame_drop_mask,  # [B, A, history_steps]
            advantage_filtering_mask,  # [B, A, history_steps]
            not_blind_mask,  # [B, A, history_steps]
            randomized_features,
            occ_points,
        ) = self.slice_history(
            scenario_data,
            env_indices,
            step_indices,
            log_scenario_data=log_scenario_data,
        )

        current_idx = positions.shape[2] - int(
            getattr(self.config, "future_steps", 0)
        ) - 1

        if category == "global":
            velocity_yaws = torch.atan2(
                velocities[..., 1], velocities[..., 0]
            )  # [B, A, history_steps]
            rel_yaws = torch.where(
                velocities.norm(dim=-1) < 0.5, orientations, velocity_yaws
            )  # [B, A, history_steps]
            rel_yaws = orientations
            goal_positions = normalize_goal_positions(goal_positions)
            goal_shape = goal_positions.shape[:-1]
            num_goal_positions = goal_positions.shape[-1] // 2
            goal_points = goal_positions.reshape(*goal_shape, num_goal_positions, 2)
            goal_positions = translate_and_rotate_2d_points(
                goal_points,
                positions[:, :, current_idx, :].unsqueeze(-2),
                rel_yaws[:, :, current_idx].unsqueeze(-1),
            ).reshape(*goal_shape, num_goal_positions * 2)
        elif category == "ego":
            ego_velocity_all = getattr(scenario_data, "agent_velocity_all", None)
            if env_indices is None or step_indices is None:
                base_positions = scenario_data.agent_positions_all[:, 0:1, -1].clone()
                if ego_velocity_all is None:
                    ego_velocities = torch.zeros_like(base_positions)
                else:
                    ego_velocities = ego_velocity_all[:, 0:1, -1].clone()
                ego_orientations = scenario_data.agent_orientation_all[
                    :, 0:1, -1
                ].clone()
            else:
                env_idx = torch.as_tensor(env_indices, device=self.config.device)
                step_idx = torch.as_tensor(step_indices, device=self.config.device)
                base_positions = scenario_data.agent_positions_all[
                    env_idx, 0, step_idx
                ].unsqueeze(1)
                if ego_velocity_all is None:
                    ego_velocities = torch.zeros_like(base_positions)
                else:
                    ego_velocities = ego_velocity_all[
                        env_idx, 0, step_idx
                    ].unsqueeze(1)
                ego_orientations = scenario_data.agent_orientation_all[
                    env_idx, 0, step_idx
                ].unsqueeze(1)
            # Calculate yaw from velocity, which is more stable at higher speeds.
            ego_velocity_yaws = torch.atan2(
                ego_velocities[..., 1], ego_velocities[..., 0]
            )  # [B, 1]
            base_orientations = torch.where(
                ego_velocities.norm(dim=-1) < 0.5, ego_orientations, ego_velocity_yaws
            )  # [B, 1]
            base_orientations = ego_orientations
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
        else:
            raise ValueError(f"Unsupported category: {category}")

        return (
            positions,  # [B, A, history_steps, 2]
            velocities,  # [B, A, history_steps, 2]
            orientations,  # [B, A, history_steps]
            sizes,  # [B, A, history_steps, 2]
            types,
            agent_mask,  # [B, A, history_steps]
            goal_positions,  # [B, A, 2]
            goal_reached,  # [B, A]
            steering_state,  # [B, A, history_steps]
            steering_control,  # [B, A, history_steps]
            acceleration_state,  # [B, A, history_steps]
            acceleration_control,  # [B, A, history_steps]
            yaw_rates,  # [B, A, history_steps]
            lanes_points,
            lanes_points_mask,
            visible_mask,  # [B, A, history_steps, A]
            frame_drop_mask,  # [B, A, history_steps]
            advantage_filtering_mask,  # [B, A, history_steps]
            not_blind_mask,  # [B, A]
            randomized_features,  # [B, A, feature_dim]
            occ_points,
        )
