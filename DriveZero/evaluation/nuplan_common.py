import copy
import json
import pickle
from pathlib import Path
from typing import Mapping

import numpy as np
import torch


def _rotate_into_frame(vectors: torch.Tensor, frame_yaw: torch.Tensor) -> torch.Tensor:
    c, s = torch.cos(frame_yaw), torch.sin(frame_yaw)
    return torch.stack((vectors[..., 0] * c + vectors[..., 1] * s,
                        -vectors[..., 0] * s + vectors[..., 1] * c), dim=-1)


def _wrap_angle(values: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(values), torch.cos(values))


def load_dataset_contract(root: Path) -> float:
    manifest = json.loads((root / "manifest.json").read_text())
    return float(manifest["frame_time_interval"])


def build_runtime_config(
    checkpoint_config: Mapping,
    *,
    dataset_root: Path,
    batch_size: int,
    num_workers: int,
) -> dict:
    config = copy.deepcopy(dict(checkpoint_config))
    dataset = config["dataset"]
    dataset["mode"] = "val"
    dataset["dataset_files"] = {"val": [str(dataset_root)]}
    dataset.update(
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False,
        shuffle=False,
        repeat=False,
        samples_per_shard=128,
        temporal_stride=1,
    )
    return config


def finalize_predictions(
    path: Path,
    predictions: Mapping[str, np.ndarray],
    *,
    time_horizon: float,
    interval_length: float,
) -> None:
    payload = {
        "predictions": predictions,
        "time_horizon": float(time_horizon),
        "interval_length": float(interval_length),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    navsim_path = path.with_name(f"{path.stem}_navsim{path.suffix}")
    for target, protocol in ((path, pickle.HIGHEST_PROTOCOL), (navsim_path, 4)):
        with target.open("wb") as file:
            pickle.dump(payload, file, protocol=protocol)
    print(
        f"Wrote {len(predictions)} predictions to {path} "
        f"(horizon={time_horizon:.3f}s, interval={interval_length:.3f}s)",
        flush=True,
    )
