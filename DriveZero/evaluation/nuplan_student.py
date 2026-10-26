"""Generate NAVTEST trajectories from a released student checkpoint."""
import argparse
from pathlib import Path
from typing import Mapping

import numpy as np
import torch

from evaluation.nuplan_common import (
    _rotate_into_frame, _wrap_angle, build_runtime_config,
    finalize_predictions, load_dataset_contract,
)
from loaders import NuplanDataloader
from vanilla_net import build_student

STEER_RATIO = 12.6
MIN_STEERING_RATE = -0.5
MAX_STEERING_RATE = 0.5
MIN_ACCELERATION = -6.5
MAX_ACCELERATION = 1.5
MAX_STEERING_ANGLE = float(np.pi / 3.0)


def aggregate_proposal_scores(score_components):
    probabilities = score_components.sigmoid()
    return (
        probabilities[..., 0].log()
        + probabilities[..., 1].log()
        + (
            5.0 * probabilities[..., 2]
            + 5.0 * probabilities[..., 3]
            + 2.0 * probabilities[..., 4]
        ).log()
    )


def build_student_state_inputs(
    batch: Mapping[str, torch.Tensor],
    *,
    interval_length: float,
) -> torch.Tensor:
    history = batch["objects_history_states"].to(dtype=torch.float32)
    previous = history[:, 0, -2]
    current = history[:, 0, -1]
    current_velocity = current[:, 6:8]
    current_yaw = current[:, 3]

    model_velocity = _rotate_into_frame(current_velocity, current_yaw)
    model_yaw = _wrap_angle(current_yaw - current_yaw)
    current_steering = (current[:, 12] / STEER_RATIO).clamp(
        -MAX_STEERING_ANGLE, MAX_STEERING_ANGLE
    )
    previous_steering = (previous[:, 12] / STEER_RATIO).clamp(
        -MAX_STEERING_ANGLE, MAX_STEERING_ANGLE
    )
    steering_rate = ((current_steering - previous_steering) / interval_length).clamp(
        MIN_STEERING_RATE, MAX_STEERING_RATE
    )
    acceleration_xy = current[:, 8:10]
    acceleration_long = (
        acceleration_xy[:, 0] * torch.cos(current_yaw)
        + acceleration_xy[:, 1] * torch.sin(current_yaw)
    ).clamp(MIN_ACCELERATION, MAX_ACCELERATION)
    ego_wheelbase = (current[:, 1] * 3.0 / 4.995).clamp_min(1.5)
    max_yaw_rate = (
        torch.linalg.norm(current_velocity, dim=-1)
        * np.tan(MAX_STEERING_ANGLE)
        / ego_wheelbase
        * 1.05
    )
    current_yaw_rate = current[:, 11].clamp(min=-max_yaw_rate, max=max_yaw_rate)

    kinematics = torch.cat(
        [
            model_velocity,
            model_yaw[:, None],
            current[:, 1:3],
            current_steering[:, None],
            steering_rate[:, None],
            acceleration_long[:, None],
            current_yaw_rate[:, None],
        ],
        dim=-1,
    ).unsqueeze(1)

    return kinematics


def create_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--backbone-weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser


def run_inference(args):
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    config["student"]["model"]["visual_backbone_weights"] = str(args.backbone_weights)
    source_interval = load_dataset_contract(args.dataset_root)
    interval_length = 0.2
    temporal_stride = interval_length / source_interval
    if temporal_stride < 1 or not np.isclose(temporal_stride, round(temporal_stride)):
        raise ValueError("0.2 seconds must be an integer multiple of the dataset interval.")
    runtime_config = build_runtime_config(
        config, dataset_root=args.dataset_root,
        batch_size=args.batch_size, num_workers=args.num_workers,
    )
    runtime_config["dataset"]["temporal_stride"] = int(round(temporal_stride))
    runtime_config["dataset"]["frame_time_interval"] = interval_length
    loader = NuplanDataloader(runtime_config, device)
    student = build_student(config, device)
    student.load_state_dict(checkpoint["model_state_dict"], strict=True)
    student.eval()
    predictions = {}
    for batch in loader:
        kinematics = build_student_state_inputs(batch, interval_length=interval_length)
        with torch.inference_mode():
            output, _ = student(
                batch["camera_images"], batch["camera_geometry"],
                batch["command"].to(device=device, dtype=torch.float32),
                kinematics.to(device),
            )
            scores = aggregate_proposal_scores(output["proposal_score"])
            trajectories = output["trajectories"][
                torch.arange(len(kinematics), device=device), scores.argmax(1),
            ].float()
        for index, metadata in enumerate(batch["metadata"]):
            predictions[str(metadata["anchor_token"]).lower()] = trajectories[index].cpu().numpy().astype(np.float32)
    finalize_predictions(
        args.output, predictions,
        time_horizon=int(config["chunk"]["horizon_k"]) * interval_length,
        interval_length=interval_length,
    )


if __name__ == "__main__":
    run_inference(create_argument_parser().parse_args())
