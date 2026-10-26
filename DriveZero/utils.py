"""Training utilities for Hermes distillation."""

from __future__ import annotations

import datetime
import functools
import logging
import math
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import yaml


@functools.lru_cache(maxsize=None)
def _drivor_cubic_spline_weights(
    num_poses: int, additional_steps: int
) -> torch.Tensor:
    """Return DrivoR's not-a-knot spline operator for the long target."""
    source_count = num_poses + additional_steps
    if num_poses < 1 or additional_steps < 1 or source_count < 4:
        raise ValueError(
            "Long trajectory spline requires positive steps and 4 source poses."
        )
    dtype = torch.float64
    system = torch.zeros(source_count, source_count, dtype=dtype)
    rhs = torch.zeros_like(system)
    system[0, :3] = torch.tensor((-1.0, 2.0, -1.0), dtype=dtype)
    system[-1, -3:] = torch.tensor((1.0, -2.0, 1.0), dtype=dtype)
    for index in range(1, source_count - 1):
        system[index, index - 1 : index + 2] = torch.tensor(
            (1.0, 4.0, 1.0), dtype=dtype
        )
        rhs[index, index - 1 : index + 2] = torch.tensor(
            (6.0, -12.0, 6.0), dtype=dtype
        )
    second_derivatives = torch.linalg.solve(system, rhs)

    indices = torch.arange(num_poses, dtype=dtype)
    alpha = 2.0 * additional_steps / (num_poses * (num_poses + 1))
    query = indices + torch.cumsum((indices + 1.0) * alpha, dim=0)
    segments = query.floor().long().clamp(max=source_count - 2)
    right_fraction = query - segments
    left_fraction = 1.0 - right_fraction
    basis = torch.eye(source_count, dtype=dtype)

    weights = []
    for row, segment in enumerate(segments.tolist()):
        left = left_fraction[row]
        right = right_fraction[row]
        weights.append(
            second_derivatives[segment] * left.pow(3) / 6.0
            + second_derivatives[segment + 1] * right.pow(3) / 6.0
            + (basis[segment] - second_derivatives[segment] / 6.0) * left
            + (basis[segment + 1] - second_derivatives[segment + 1] / 6.0)
            * right
        )
    return torch.stack(weights)


def drivor_long_trajectory_target(
    trajectory: torch.Tensor, num_poses: int, additional_steps: int
) -> torch.Tensor:
    expected_steps = num_poses + additional_steps
    if trajectory.ndim != 3 or trajectory.shape[1:] != (expected_steps, 3):
        raise ValueError(
            "Long trajectory source must have shape "
            f"[B,{expected_steps},3], got {tuple(trajectory.shape)}."
        )
    weights = _drivor_cubic_spline_weights(num_poses, additional_steps).to(
        device=trajectory.device, dtype=trajectory.dtype
    )

    yaw = trajectory[..., 2]
    yaw_delta = yaw[:, 1:] - yaw[:, :-1]
    yaw_delta = torch.atan2(torch.sin(yaw_delta), torch.cos(yaw_delta))
    yaw_unwrapped = torch.cat(
        (yaw[:, :1], yaw[:, :1] + torch.cumsum(yaw_delta, dim=1)), dim=1
    )
    spline_input = torch.cat((trajectory[..., :2], yaw_unwrapped[..., None]), dim=-1)
    target = torch.einsum("ts,bsc->btc", weights, spline_input)
    target_yaw = torch.atan2(torch.sin(target[..., 2]), torch.cos(target[..., 2]))
    return torch.cat((target[..., :2], target_yaw[..., None]), dim=-1)


def seed_everything(seed: int | None, torch_deterministic: bool) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    if seed is not None:
        torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = torch_deterministic


def load_plain_yaml(path: str) -> dict[str, Any]:
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a YAML mapping in {path}.")
    return data


def resolve_per_rank_batch_size(global_batch_size: int, ddp_world_size: int) -> int:
    global_batch_size = int(global_batch_size)
    ddp_world_size = int(ddp_world_size)
    if global_batch_size <= 0:
        raise ValueError(f"dataset.batch_size must be positive, got {global_batch_size}.")
    if ddp_world_size <= 0:
        raise ValueError(f"ddp_world_size must be positive, got {ddp_world_size}.")
    if global_batch_size % ddp_world_size != 0:
        raise ValueError(
            "dataset.batch_size is a global batch size and must be divisible by "
            f"WORLD_SIZE: batch_size={global_batch_size}, WORLD_SIZE={ddp_world_size}."
        )
    return global_batch_size // ddp_world_size


def setup_distributed(config_device: str) -> tuple[str, int, int, int, bool]:
    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    if distributed:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
    else:
        rank = 0
        world_size = 1
        local_rank = 0
        device = config_device
        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.set_device(torch.device(device).index or 0)
    return device, rank, world_size, local_rank, distributed


def ph_cast_img_pts_to_cam_ray(pts, intrinsic, distortion) -> np.ndarray:
    import cv2

    pts_norm = cv2.undistortPoints(
        src=pts,
        cameraMatrix=intrinsic,
        distCoeffs=distortion,
    ).reshape(-1, 2)
    return np.hstack([pts_norm, np.ones_like(pts_norm[:, [0]])])


def init_logging(log_path: Path, file_mode: str = "w") -> None:
    logging.root = logging.RootLogger("INFO")
    formatter = logging.Formatter("[%(asctime)s][%(levelname)s] - %(message)s")

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logging.root.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_path, mode=file_mode)
    file_handler.setFormatter(formatter)
    logging.root.addHandler(file_handler)

    logging.Logger.root = logging.root
    logging.Logger.manager.root = logging.root


def prepare_run_dir(config: dict[str, Any], training_cfg: dict[str, Any], fresh: bool = False) -> Path:
    output_dir = Path(training_cfg["output_root"]) / training_cfg["run_name"]
    if fresh and output_dir.exists():
        shutil.rmtree(output_dir)
    checkpoint_latest = output_dir / "checkpoint_latest.pt"
    resuming = not fresh and checkpoint_latest.exists()
    output_dir.mkdir(parents=True, exist_ok=True)
    init_logging(output_dir / "train.log", file_mode="a" if resuming else "w")
    with open(output_dir / "config.yaml", "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)
    return output_dir


def adjust_learning_rate(
    optimizer: torch.optim.Optimizer,
    update: int,
    max_updates: int,
    base_lr: float,
) -> float:
    progress = min(float(update) / float(max(max_updates - 1, 1)), 1.0)
    lr = base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr


def _format_eta(seconds: float) -> str:
    return str(datetime.timedelta(seconds=int(max(seconds, 0.0))))


class ETATracker:
    def __init__(self, window_size: int = 20):
        self.window_size = window_size
        self.last_time = time.time()
        self.iter_times: list[float] = []

    def update(self) -> float:
        now = time.time()
        iter_time = now - self.last_time
        self.last_time = now
        self.iter_times.append(iter_time)
        if len(self.iter_times) > self.window_size:
            self.iter_times.pop(0)
        return iter_time

    def eta(self, current_step: int, total_steps: int) -> str:
        if not self.iter_times:
            return "N/A"
        avg_time = sum(self.iter_times) / len(self.iter_times)
        return _format_eta(avg_time * (total_steps - current_step))


def load_model_checkpoint(agent: torch.nn.Module, checkpoint_path: str, device: str, strict: bool = True) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint["model_state_dict"]

    agent.load_state_dict(state_dict, strict=strict)


def save_student(
    agent: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    output_dir: Path,
    update: int,
    config: dict[str, Any],
) -> None:
    checkpoint = {
        "update": update,
        "model_state_dict": agent.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config,
    }
    for path in (output_dir / f"checkpoint_{update}.pt", output_dir / "checkpoint_latest.pt"):
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        torch.save(checkpoint, tmp_path)
        tmp_path.replace(path)
