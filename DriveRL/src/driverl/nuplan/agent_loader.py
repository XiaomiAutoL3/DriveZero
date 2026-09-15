"""DriveRL checkpoint/config loading for nuPlan simulation."""

from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import yaml

from driverl.agents import BaseAgent, LearningAgent
from driverl.agents.config import AgentConfig
from driverl.env.config import EnvConfig
from driverl.env.engine.config import EngineConfig
from driverl.utils.gym_compat import Box
from driverl.utils.misc import RecursiveLoader, apply_extends


@dataclass(frozen=True)
class LoadedDriveRLAgent:
    """Loaded eval agent plus runtime metadata needed by the planner."""

    agent: LearningAgent
    action_keys_tensor: torch.Tensor | None
    env_config: EnvConfig
    agent_config: AgentConfig
    engine_config: EngineConfig
    frame_time_interval: float
    tts_gamma: float
    checkpoint_update: int
    missing_keys: list[str]
    unexpected_keys: list[str]


def load_driverl_agent_for_nuplan(
    *,
    config_path: str,
    checkpoint_path: str,
    device: str,
    strict_checkpoint: bool = True,
    compile_agent: bool = False,
) -> LoadedDriveRLAgent:
    """Build the DriveRL policy exactly enough for closed-loop eval inference."""
    config_data = _load_yaml_config(config_path)
    env_config = EnvConfig.from_dict(config_data["env"], allow_extra=True)
    agent_config = AgentConfig.from_dict(config_data["agent"], allow_extra=True)
    env_config.device = device
    agent_config.device = device
    agent_config.mode = "test"
    agent_config.compile = False
    agent_config.enable_occupancy_grid = env_config.enable_occupancy_grid
    agent_config.num_goal_positions = env_config.num_goal_positions
    engine_config = env_config.engine_config

    if env_config.action_type != "continuous":
        raise NotImplementedError(
            "nuPlan eval loader currently supports continuous DriveRL policies only; "
            f"got action_type={env_config.action_type!r}."
        )

    if env_config.dynamics_model == "nuplan_bicycle_model":
        lateral_low = -float(
            getattr(engine_config, "nuplan_bicycle_max_steering_rate", 0.5)
        )
        lateral_high = float(
            getattr(engine_config, "nuplan_bicycle_max_steering_rate", 0.5)
        )
    else:
        lateral_low = engine_config.min_jerk_lat
        lateral_high = engine_config.max_jerk_lat

    action_space = Box(
        low=np.asarray(
            [engine_config.min_jerk_long, lateral_low],
            dtype=np.float32,
        ),
        high=np.asarray(
            [engine_config.max_jerk_long, lateral_high],
            dtype=np.float32,
        ),
        dtype=np.float32,
    )
    frame_time_interval = _policy_frame_time_interval(env_config)
    agent = BaseAgent.agent_factory(
        agent_name=agent_config.agent_name,
        config=agent_config,
        action_space=action_space,
        batch_size=1,
        action_key_to_values=None,
        frame_time_interval=frame_time_interval,
        no_goal_allowed=not engine_config.done_after_reaching_goal,
        domain_randomization_config=engine_config.domain_randomization,
    )
    if not isinstance(agent, LearningAgent):
        raise TypeError(f"Expected LearningAgent, got {type(agent).__name__}.")

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except (pickle.UnpicklingError, RuntimeError, ValueError) as exc:
        raise ValueError(
            "DriveRL checkpoints must use the public 'driverl_weights_only' "
            "format. Use a bundled artifact under release/checkpoints/."
        ) from exc
    if not isinstance(checkpoint, dict) or checkpoint.get("format") != "driverl_weights_only":
        raise ValueError(
            "Unsupported checkpoint format. Expected a 'driverl_weights_only' "
            "release artifact containing update and model_state_dict."
        )
    state_dict = checkpoint["model_state_dict"]
    if strict_checkpoint:
        agent.load_state_dict(_normalize_state_keys(agent, state_dict), strict=True)
        missing: list[str] = []
        unexpected: list[str] = []
    else:
        missing, unexpected = agent._load_state_dict_flexible(state_dict)

    if compile_agent:
        agent = torch.compile(agent)  # type: ignore[assignment]

    agent.to(device)
    agent.eval()
    return LoadedDriveRLAgent(
        agent=agent,
        action_keys_tensor=None,
        env_config=env_config,
        agent_config=agent_config,
        engine_config=engine_config,
        frame_time_interval=frame_time_interval,
        tts_gamma=float(config_data.get("tts", {}).get("gamma", 0.99)),
        checkpoint_update=int(checkpoint.get("update", 0)),
        missing_keys=missing,
        unexpected_keys=unexpected,
    )


def _policy_frame_time_interval(env_config: EnvConfig) -> float:
    """Return the policy input interval encoded by the release config."""
    sample_rate_hz = env_config.dataloader_config.target_sample_rate_hz
    if sample_rate_hz is None:
        return 0.2
    sample_rate_hz = float(sample_rate_hz)
    if sample_rate_hz <= 0:
        raise ValueError(f"target_sample_rate_hz must be positive, got {sample_rate_hz}")
    return 1.0 / sample_rate_hz


def _load_yaml_config(path: str) -> dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.load(f, Loader=RecursiveLoader)
    data = apply_extends(data)
    return data


def _normalize_state_keys(
    agent: LearningAgent, state_dict: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    """Match compiled/non-compiled key prefixes before strict load."""
    model_keys = agent.state_dict().keys()
    model_is_compiled = any(key.startswith("_orig_mod.") for key in model_keys)
    checkpoint_is_compiled = all(key.startswith("_orig_mod.") for key in state_dict)
    if model_is_compiled and not checkpoint_is_compiled:
        return {"_orig_mod." + key: value for key, value in state_dict.items()}
    if not model_is_compiled and checkpoint_is_compiled:
        return {key[len("_orig_mod.") :]: value for key, value in state_dict.items()}
    return state_dict
