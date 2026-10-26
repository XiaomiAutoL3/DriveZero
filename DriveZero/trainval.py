"""Distill Hermes trajectories and PDM proposal scores into the camera student."""
import argparse
import logging
from collections import deque
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

from augmentations.nuplan_goal_command import validate_goal_command_augmentation_config
from loaders import NuplanDataloader
from pdm_oracle import PDMOracle
from teachers import get_teacher_class
from utils import (ETATracker, adjust_learning_rate, load_plain_yaml,
                   prepare_run_dir, save_student, seed_everything, setup_distributed)
from vanilla_net import build_student


def _apply_overrides(config: dict[str, Any], overrides: list[str]) -> None:
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Override must be KEY=VALUE, got {override!r}.")
        path, value = override.split("=", 1)
        keys = path.split(".")
        target = config
        for key in keys[:-1]:
            if key not in target or not isinstance(target[key], dict):
                raise ValueError(f"Unknown config override path: {path!r}.")
            target = target[key]
        if not keys[-1] or keys[-1] not in target:
            raise ValueError(f"Unknown config override path: {path!r}.")
        target[keys[-1]] = yaml.safe_load(value)


def _trajectory_wta_loss(proposals, target):
    return torch.linalg.norm(
        proposals - target[:, None], dim=-1, ord=1
    ).mean(-1).amin(1).mean()


def _sample_proposal_indices(
    batch_size: int,
    num_proposals: int,
    sample_count: int,
    device: torch.device,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    if sample_count < 1 or sample_count > num_proposals:
        raise ValueError(
            f"sample_count must be within [1,{num_proposals}], got {sample_count}."
        )
    random_keys = torch.rand(
        batch_size, num_proposals, device=device, generator=generator
    )
    return random_keys.argsort(dim=1)[:, :sample_count]


def _gather_proposals(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    if values.ndim < 2 or indices.ndim != 2 or values.shape[0] != indices.shape[0]:
        raise ValueError("Expected values [B,P,...] and indices [B,N].")
    gather_index = indices.reshape(
        indices.shape + (1,) * (values.ndim - 2)
    ).expand(indices.shape + values.shape[2:])
    return values.gather(1, gather_index)


def _pdm_score_loss(logits, targets, validity=None):
    targets = targets.clone()
    targets[..., 0][targets[..., 0] == 0.5] = 0.0
    targets[..., 5][targets[..., 5] == 0.5] = 0.0
    losses = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    ttc_valid = targets[..., 3] != 2.0
    direction_loss = losses[..., 5].mean()
    if validity is not None:
        if validity.shape != targets.shape:
            raise ValueError(
                f"PDM validity shape {validity.shape} must match targets {targets.shape}."
            )
        direction_valid = validity[..., 5]
        direction_loss = (
            losses[..., 5][direction_valid].sum()
            / direction_valid.sum().clamp(min=1)
        )
    return (
        losses[..., 0].mean()
        + losses[..., 1].mean()
        + losses[..., 2].mean()
        + losses[..., 3][ttc_valid].sum() / ttc_valid.sum().clamp(min=1)
        + losses[..., 4].mean()
        + direction_loss
    )


def _aggregate_pdm_targets(targets: torch.Tensor) -> torch.Tensor:
    return targets[..., 0] * targets[..., 1] * (
        5.0 * targets[..., 2]
        + 5.0 * targets[..., 3]
        + 2.0 * targets[..., 4]
    ) / 12.0


def _aggregate_pdm_logits(logits: torch.Tensor) -> torch.Tensor:
    probabilities = logits.sigmoid()
    return (
        probabilities[..., 0].log()
        + probabilities[..., 1].log()
        + (
            5.0 * probabilities[..., 2]
            + 5.0 * probabilities[..., 3]
            + 2.0 * probabilities[..., 4]
        ).log()
    )


def _pdm_selection_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> dict[str, torch.Tensor]:
    predicted_scores = _aggregate_pdm_logits(logits)
    target_scores = _aggregate_pdm_targets(targets)

    predicted_index = predicted_scores.argmax(dim=1)
    target_top1 = target_scores.argmax(dim=1)
    target_top5 = target_scores.topk(
        k=min(5, target_scores.shape[1]), dim=1
    ).indices

    top1 = (predicted_index == target_top1).float().mean()
    top5 = (target_top5 == predicted_index[:, None]).any(dim=1).float().mean()

    return {"top1": top1, "top5": top5}


def _distill_losses(student_model, teacher, batch, *, cfg, pdm_oracle):
    prepared = teacher.prepare_batch(
        batch,
        horizon_k=int(cfg["chunk"]["horizon_k"]),
        long_trajectory_additional_steps=int(cfg["chunk"]["long_trajectory"]["additional_steps"]),
    )
    student_output, _ = student_model(
        batch["camera_images"], batch["camera_geometry"],
        prepared["student_command"].to(batch["camera_images"].device),
        prepared["kinematics_features"],
    )
    trajectories = student_output["trajectories"].float()
    trajectory_loss = _trajectory_wta_loss(trajectories, prepared["trajectory_target"].float())
    trajectory_long_loss = _trajectory_wta_loss(
        trajectories, prepared["trajectory_long_target"].float()
    )
    policy_loss = float(cfg["training"]["trajectory_loss_weight"]) * (
        trajectory_loss + trajectory_long_loss
    )
    indices = _sample_proposal_indices(
        trajectories.shape[0], trajectories.shape[1],
        int(cfg["planning"]["proposal_score_sample_count"]), trajectories.device,
    )
    score_trajectories = _gather_proposals(trajectories, indices)
    score_logits = _gather_proposals(student_output["proposal_score"], indices).float()
    score_targets, score_validity = pdm_oracle.score(
        score_trajectories, batch["metadata"],
        route_points=batch["intent_route_points"],
        route_masks=batch["intent_route_mask"],
        route_augmented=batch["intent_route_augmented"],
        reference_trajectories=prepared["trajectory_target"],
    )
    score_loss = _pdm_score_loss(score_logits, score_targets.float(), score_validity)
    loss = policy_loss + float(cfg["training"]["score_loss_weight"]) * score_loss
    return loss, {
        "loss_trajectory": trajectory_loss,
        "loss_trajectory_long": trajectory_long_loss,
        "loss_score": score_loss,
        **_pdm_selection_metrics(score_logits, score_targets.float()),
    }


def run(config_path, fresh=False, run_name=None, overrides=None):
    cfg = load_plain_yaml(config_path)
    _apply_overrides(cfg, overrides or [])
    if run_name is not None:
        cfg["training"]["run_name"] = run_name
    validate_goal_command_augmentation_config(cfg)
    training_cfg = cfg["training"]
    device, rank, world_size, local_rank, distributed = setup_distributed(training_cfg["device"])
    output_dir = Path(training_cfg["output_root"]) / training_cfg["run_name"]
    checkpoint_latest = output_dir / "checkpoint_latest.pt"
    resuming = not fresh and checkpoint_latest.exists()
    checkpoint = (
        torch.load(checkpoint_latest, map_location=device, weights_only=False)
        if resuming else None
    )
    update = int(checkpoint["update"]) if resuming else 0
    if resuming:
        previous = checkpoint["config"]
        for key in ("teacher", "student", "chunk", "augmentation", "planning"):
            if previous[key] != cfg[key]:
                raise ValueError(f"Cannot resume with changed {key}; use a new run name.")
    if rank == 0:
        output_dir = prepare_run_dir(cfg, training_cfg, fresh=fresh)
        tb_writer = SummaryWriter(output_dir, purge_step=update + 1 if resuming else None)

    pdm_oracle = PDMOracle(cfg["planning"]["pdm_oracle"])
    seed_everything(training_cfg["seed"] + rank, torch_deterministic=False)
    loader = NuplanDataloader(cfg, device, ddp_world_size=world_size)
    dataset_cfg = cfg["dataset"]
    teacher = get_teacher_class(cfg["teacher"]["name"])(
        loader=loader, device=device,
        checkpoint_path=cfg["teacher"]["checkpoint_path"],
        compile_model=cfg["chunk"]["compile_teacher"],
        **{key: dataset_cfg[key] for key in (
            "init_steps", "num_steps", "goal_sampling_mode", "goal_sampling_gamma",
            "goal_count_probs", "goal_snap_keep_prob", "goal_snap_k", "goal_snap_max_distance",
        )},
    )
    student = build_student(cfg, device)
    student.train()
    base_lr = float(training_cfg["learning_rate"])
    optimizer = torch.optim.AdamW(
        (param for param in student.parameters() if param.requires_grad),
        lr=base_lr, weight_decay=0.01,
    )
    if resuming:
        student.load_state_dict(checkpoint["model_state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        del checkpoint
    student_model = DDP(student, device_ids=[local_rank]) if distributed else student
    max_updates = int(training_cfg["max_updates"])
    metric_keys = ["loss_trajectory", "loss_trajectory_long", "loss_score", "top1", "top5"]
    running = {key: deque(maxlen=100) for key in ["loss", *metric_keys]}
    eta_tracker = ETATracker()
    logging.info("Teacher checkpoint: %s; Hermes commit: %s", teacher.checkpoint_path, teacher.expected_hermes_commit)
    logging.info("Run %s; world size %d; global batch %d", training_cfg["run_name"], world_size, loader.global_batch_size)
    logging.info("Student trainable parameters: %d", sum(p.numel() for p in student.parameters() if p.requires_grad))
    while update < max_updates:
        batch = next(loader)
        lr = adjust_learning_rate(optimizer, update, max_updates, base_lr)
        loss, metrics = _distill_losses(student_model, teacher, batch, cfg=cfg, pdm_oracle=pdm_oracle)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), float(training_cfg["grad_clip_norm"]))
        optimizer.step()
        update += 1
        iter_time = eta_tracker.update()
        for key, value in {"loss": loss, **metrics}.items():
            running[key].append(value.detach().cpu().item())
        if rank == 0 and update % int(training_cfg["log_interval"]) == 0:
            means = {key: sum(values) / len(values) for key, values in running.items()}
            logging.info(
                "Iter [%d/%d]: %s, lr: %.6f, time: %.2f, eta: %s",
                update, max_updates, ", ".join(f"{k}: {v:.4f}" for k, v in means.items()),
                lr, iter_time, eta_tracker.eta(update, max_updates),
            )
            for key, value in means.items():
                tb_writer.add_scalar(f"training/{key}", value, update)
        if rank == 0 and update % int(training_cfg["save_interval"]) == 0:
            save_student(student, optimizer, output_dir, update, cfg)
    if rank == 0:
        save_student(student, optimizer, output_dir, update, cfg)
        tb_writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/nuplan.yaml")
    parser.add_argument("--fresh", action="store_true", help="Replace this run's outputs and start from scratch.")
    parser.add_argument("--run-name")
    parser.add_argument("--override", nargs="+", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()
    run(args.config, fresh=args.fresh, run_name=args.run_name, overrides=args.override)
