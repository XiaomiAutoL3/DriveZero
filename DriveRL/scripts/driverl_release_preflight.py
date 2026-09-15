#!/usr/bin/env python3
"""Validate a DriveRL release artifact before starting nuPlan simulation.

The release manifest and runtime config describe the inference contract. This helper
checks both layers and can optionally import the companion simulation entry
point.  It is deliberately read-only: no files are created or modified.
"""

from __future__ import annotations

import argparse
import importlib
import pickle
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch
import yaml


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--release-id", required=True)
    parser.add_argument("--driverl-root", type=Path, default=None)
    parser.add_argument("--nuplan-root", type=Path, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--check-simulation-import",
        action="store_true",
        help="Import nuplan.planning.script.run_simulation as a runtime check.",
    )
    parser.add_argument(
        "--skip-checkpoint-load",
        action="store_true",
        help="Validate metadata only; do not instantiate/load the policy.",
    )
    return parser


def _require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise SystemExit(f"PREFLIGHT_ERROR missing_{label}={path}")


def _manifest_artifact(
    manifest: dict[str, Any], release_id: str
) -> tuple[str, dict[str, Any]]:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise SystemExit("PREFLIGHT_ERROR manifest has no artifacts list")

    for artifact in artifacts:
        if isinstance(artifact, dict) and artifact.get("id") == release_id:
            return release_id, artifact
    raise SystemExit(f"PREFLIGHT_ERROR unknown_release_id={release_id!r}")


def _check_raw_config(config: dict[str, Any], artifact: dict[str, Any]) -> None:
    env = config.get("env")
    agent = config.get("agent")
    if not isinstance(env, dict) or not isinstance(agent, dict):
        raise SystemExit("PREFLIGHT_ERROR config must contain env and agent mappings")

    expected_pairs = {
        "release_id": (config.get("release_id"), artifact.get("id")),
        "agent_name": (agent.get("agent_name"), artifact.get("agent_name")),
        "history_steps": (agent.get("history_steps"), artifact.get("history_steps")),
        "future_steps": (agent.get("future_steps", 0), artifact.get("future_steps")),
        "dynamics_model": (env.get("dynamics_model"), artifact.get("dynamics_model")),
        "network_name": (agent.get("network_name"), artifact.get("network_name")),
    }
    for field, (actual, expected) in expected_pairs.items():
        if expected is not None and actual != expected:
            raise SystemExit(
                f"PREFLIGHT_ERROR config_{field}={actual!r} manifest_expected={expected!r}"
            )

    expected_rate = artifact.get("target_sample_rate_hz")
    actual_rate = env.get("dataloader_config", {}).get("target_sample_rate_hz")
    if expected_rate is not None and actual_rate is not None:
        if abs(float(actual_rate) - float(expected_rate)) > 1e-6:
            raise SystemExit(
                "PREFLIGHT_ERROR config_target_sample_rate_hz="
                f"{actual_rate!r} manifest_expected={expected_rate!r}"
            )

    expected_vd = artifact.get("value_decomposition")
    if expected_vd is not None:
        actual_vd = bool(agent.get("enable_value_decomposition", False))
        if bool(expected_vd) != actual_vd:
            raise SystemExit(
                "PREFLIGHT_ERROR value_decomposition mismatch "
                f"agent={actual_vd} manifest={bool(expected_vd)}"
            )


def _check_artifact_contract(artifact: dict[str, Any]) -> None:
    """Require the runtime fields needed to identify a release artifact."""
    required = ("network_name",)
    missing = [field for field in required if field not in artifact]
    if missing:
        raise SystemExit(
            "PREFLIGHT_ERROR manifest artifact missing required fields="
            + ",".join(missing)
        )
    network_name = artifact["network_name"]
    if not isinstance(network_name, str) or not network_name.strip():
        raise SystemExit("PREFLIGHT_ERROR manifest network_name must be a non-empty string")


def _check_manifest_paths(
    *,
    driverl_root: Path | None,
    config_path: Path,
    artifact: dict[str, Any],
) -> None:
    """Reject a bundled release run that silently uses another config."""
    if driverl_root is None:
        return

    for label, actual_path, field in (("config", config_path, "config_file"),):
        expected = artifact.get(field)
        if not expected or actual_path is None:
            continue
        expected_path = Path(str(expected))
        if not expected_path.is_absolute():
            expected_path = driverl_root / expected_path
        if expected_path.is_file() and actual_path.resolve() != expected_path.resolve():
            raise SystemExit(
                f"PREFLIGHT_ERROR {label}_path={str(actual_path)!r} "
                f"manifest_expected={str(expected_path)!r}"
            )


def _load_checkpoint(path: Path) -> dict[str, Any]:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except (pickle.UnpicklingError, RuntimeError, ValueError) as exc:
        raise SystemExit(
            "PREFLIGHT_ERROR checkpoint must be a driverl_weights_only "
            "artifact"
        ) from exc
    if not isinstance(checkpoint, dict):
        raise SystemExit("PREFLIGHT_ERROR checkpoint is not a mapping")
    if checkpoint.get("format") != "driverl_weights_only":
        raise SystemExit(
            "PREFLIGHT_ERROR unsupported checkpoint format; expected "
            "driverl_weights_only"
        )
    if "model_state_dict" not in checkpoint:
        raise SystemExit("PREFLIGHT_ERROR checkpoint has no model_state_dict")
    return checkpoint


def _check_simulation_import(nuplan_root: Path | None) -> None:
    if nuplan_root is not None:
        root = str(nuplan_root)
        if root not in sys.path:
            sys.path.insert(0, root)
    try:
        module = importlib.import_module("nuplan.planning.script.run_simulation")
    except Exception as exc:  # preserve the first actionable import exception
        raise SystemExit(
            "PREFLIGHT_ERROR simulation_import="
            f"{type(exc).__name__}: {exc}"
        ) from exc
    print(f"PREFLIGHT simulation_import=ok module={module.__file__}")


def _check_companion_pin(manifest: dict[str, Any], nuplan_root: Path | None) -> None:
    """Verify a source-checkout companion matches the release manifest pin."""
    if nuplan_root is None:
        return
    companion = manifest.get("nuplan_companion")
    expected = companion.get("commit") if isinstance(companion, dict) else None
    if not expected or not (nuplan_root / ".git").exists():
        return
    try:
        result = subprocess.run(
            ["git", "-C", str(nuplan_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise SystemExit(
            "PREFLIGHT_ERROR companion_commit_lookup="
            f"{type(exc).__name__}: {exc}"
        ) from exc
    actual = result.stdout.strip()
    if actual != expected:
        raise SystemExit(
            f"PREFLIGHT_ERROR companion_commit={actual} manifest_expected={expected}"
        )
    print(f"PREFLIGHT companion_commit=ok commit={actual}")


def _check_policy_load(
    config_path: Path,
    checkpoint_path: Path,
    device: str,
    driverl_root: Path | None,
    artifact: dict[str, Any],
) -> None:
    if driverl_root is not None:
        src = driverl_root / "src"
        if src.is_dir() and str(src) not in sys.path:
            sys.path.insert(0, str(src))
    try:
        from driverl.nuplan.agent_loader import load_driverl_agent_for_nuplan
    except Exception as exc:
        raise SystemExit(
            "PREFLIGHT_ERROR driverl_import="
            f"{type(exc).__name__}: {exc}"
        ) from exc
    try:
        loaded = load_driverl_agent_for_nuplan(
            config_path=str(config_path),
            checkpoint_path=str(checkpoint_path),
            device=device,
            strict_checkpoint=True,
        )
    except Exception as exc:
        raise SystemExit(
            "PREFLIGHT_ERROR strict_policy_load="
            f"{type(exc).__name__}: {exc}"
        ) from exc
    expected_agent = artifact.get("agent_name")
    if expected_agent is not None and loaded.agent_config.agent_name != expected_agent:
        raise SystemExit(
            "PREFLIGHT_ERROR agent_name="
            f"{loaded.agent_config.agent_name!r} manifest_expected={expected_agent!r}"
        )
    print(
        "PREFLIGHT policy_load=ok "
        f"agent={type(loaded.agent).__name__} update={loaded.checkpoint_update} "
        f"interval_s={loaded.frame_time_interval:g}"
    )


def main() -> int:
    args = _parser().parse_args()
    _require_file(args.manifest, "manifest")
    _require_file(args.config, "config")
    _require_file(args.checkpoint, "checkpoint")

    try:
        manifest = yaml.safe_load(args.manifest.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SystemExit(
            f"PREFLIGHT_ERROR manifest_parse={type(exc).__name__}: {exc}"
        ) from exc
    if not isinstance(manifest, dict):
        raise SystemExit("PREFLIGHT_ERROR manifest must be a mapping")
    if manifest.get("route_roadblock_correction") is not True:
        raise SystemExit(
            "PREFLIGHT_ERROR manifest route_roadblock_correction must be true"
        )

    release_id, artifact = _manifest_artifact(manifest, args.release_id)
    _check_artifact_contract(artifact)
    expected_checkpoint_file = artifact.get("checkpoint_file")
    expected_checkpoint_name = (
        Path(str(expected_checkpoint_file)).name
        if expected_checkpoint_file is not None
        else None
    )
    if (
        expected_checkpoint_name is not None
        and args.checkpoint.name != expected_checkpoint_name
    ):
        raise SystemExit(
            f"PREFLIGHT_ERROR checkpoint_file={args.checkpoint.name!r} "
            f"manifest_expected={expected_checkpoint_file!r}"
        )

    try:
        config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SystemExit(
            f"PREFLIGHT_ERROR config_parse={type(exc).__name__}: {exc}"
        ) from exc
    if not isinstance(config, dict):
        raise SystemExit("PREFLIGHT_ERROR config must be a mapping")

    _check_manifest_paths(
        driverl_root=args.driverl_root,
        config_path=args.config,
        artifact=artifact,
    )
    _check_raw_config(config, artifact)
    _check_companion_pin(manifest, args.nuplan_root)

    checkpoint = _load_checkpoint(args.checkpoint)
    actual_update = checkpoint.get("update")
    expected_update = artifact.get("checkpoint_update")
    if expected_update is not None and int(actual_update) != int(expected_update):
        raise SystemExit(
            f"PREFLIGHT_ERROR checkpoint_update={actual_update!r} "
            f"manifest_expected={expected_update!r}"
        )
    expected_format = artifact.get("checkpoint_format")
    if expected_format is not None and checkpoint.get("format") != expected_format:
        raise SystemExit(
            f"PREFLIGHT_ERROR checkpoint_format={checkpoint.get('format')!r} "
            f"manifest_expected={expected_format!r}"
        )

    print(
        "PREFLIGHT artifact=ok "
        f"release_id={release_id} checkpoint={args.checkpoint} update={actual_update}"
    )
    if args.check_simulation_import:
        _check_simulation_import(args.nuplan_root)
    if not args.skip_checkpoint_load:
        _check_policy_load(
            args.config,
            args.checkpoint,
            args.device,
            args.driverl_root,
            artifact,
        )
    print("PREFLIGHT_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
