"""Hermes-free packed WebDataset loading shared by Dermes loaders."""

from __future__ import annotations

import io
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import torch
import webdataset as wds
from torch.utils.data import IterableDataset

from preprocess.camera_image import (
    CAMERA_IMAGE_HEIGHT,
    CAMERA_IMAGE_WIDTH,
    decode_camera_image,
)
from utils import (
    ph_cast_img_pts_to_cam_ray,
    resolve_per_rank_batch_size,
)


CAMERA_MODEL_PINHOLE = 0.0
INVALID_GROUP_ID = np.iinfo(np.uint32).max

BASE_PACKED_FIELDS = (
    "lanes_centers_points",
    "lanes_centers_masks",
    "lanes_centers_attributes",
    "other_lanes_points",
    "other_lanes_masks",
    "other_lanes_attributes",
    "route_points",
    "objects_history_states",
    "objects_history_masks",
    "objects_future_states",
    "objects_future_masks",
    "ego_index",
    "occupancy_grid",
    "calib_intrinsic",
    "calib_distortion",
    "calib_sensor2vehicle",
)


@dataclass(frozen=True)
class PackedMetadata:
    frame_gap: int
    frame_time_interval: float
    history_length: int
    future_length: int
    max_agents: int
    max_lanes_centers: int
    max_lanes_other: int
    route_points: int
    occ_grid_xmin: float = -60.0
    occ_grid_xmax: float = 200.0
    occ_grid_ymin: float = -30.0
    occ_grid_ymax: float = 30.0
    occ_grid_resolution: float = 0.2

    @property
    def occ_grid_length(self) -> int:
        return int(round((self.occ_grid_xmax - self.occ_grid_xmin) / self.occ_grid_resolution))

    @property
    def occ_grid_width(self) -> int:
        return int(round((self.occ_grid_ymax - self.occ_grid_ymin) / self.occ_grid_resolution))


def validate_camera_sensor_fqdns(
    sensor_fqdns: Sequence[str],
    valid_sensors: Sequence[str],
) -> tuple[str, ...]:
    sensors = tuple(str(sensor) for sensor in sensor_fqdns)
    if not sensors:
        raise ValueError("camera_sensor_fqdns must contain at least one camera sensor.")
    duplicates = sorted({sensor for sensor in sensors if sensors.count(sensor) > 1})
    if duplicates:
        raise ValueError(f"camera_sensor_fqdns contains duplicate sensors: {duplicates}")
    unknown = [sensor for sensor in sensors if sensor not in valid_sensors]
    if unknown:
        raise ValueError(f"camera_sensor_fqdns contains unknown sensors: {unknown}")
    return sensors


def camera_depth_bins(depth_num: int, depth_start: float, depth_end: float) -> np.ndarray:
    index = np.arange(depth_num, dtype=np.float32)
    bin_size = (float(depth_end) - float(depth_start)) / (depth_num * (depth_num + 1))
    return float(depth_start) + bin_size * index * (index + 1.0)


def camera_rays_from_calib(
    intrinsics: np.ndarray,
    distortions: np.ndarray,
    patch_size: int,
    *,
    image_height: int = CAMERA_IMAGE_HEIGHT,
    image_width: int = CAMERA_IMAGE_WIDTH,
) -> torch.Tensor:
    grid_h = image_height // patch_size
    grid_w = image_width // patch_size
    coords_h = (np.arange(grid_h, dtype=np.float32) + 0.5) * patch_size
    coords_w = (np.arange(grid_w, dtype=np.float32) + 0.5) * patch_size
    coords_2d = np.stack(np.meshgrid(coords_w, coords_h), axis=-1).reshape(-1, 2)

    rays_all = []
    for intrinsic, distortion in zip(intrinsics, distortions):
        if distortion[0] == CAMERA_MODEL_PINHOLE:
            rays = ph_cast_img_pts_to_cam_ray(coords_2d, intrinsic, distortion[1:6])
        else:
            raise ValueError(f"Unsupported distortion model id: {distortion[0]}")
        rays_all.append(rays.reshape(grid_h, grid_w, 3).astype(np.float32))
    return torch.from_numpy(np.stack(rays_all, axis=0))


def metric_points_from_rays(
    rays: torch.Tensor,
    sensor2vehicle: torch.Tensor,
    depth: torch.Tensor,
) -> torch.Tensor:
    leading = rays.shape[:-3]
    grid_h, grid_w = rays.shape[-3:-1]
    rays_flat = rays.reshape(-1, grid_h, grid_w, 3)
    transforms = sensor2vehicle.reshape(-1, 4, 4)
    camera_points = rays_flat.unsqueeze(-2) * depth.view(-1, 1)
    vehicle = torch.einsum(
        "mhwdc,mkc->mhwdk", camera_points, transforms[:, :3, :3]
    )
    vehicle = vehicle + transforms[:, None, None, None, :3, 3]
    return vehicle.reshape(*leading, grid_h, grid_w, depth.shape[0], 3)


def _build_group_polylines(points, valid_indices, valid_groups, unique_groups):
    polylines = {}
    for group_id in unique_groups:
        indices = valid_indices[valid_groups == group_id]
        polylines[int(group_id)] = np.concatenate(
            [points[indices, 0], points[indices[-1], 1][None]], axis=0
        )
    return polylines


def _build_successor_map(next_groups, unique_groups):
    if next_groups is None:
        return {}
    valid_groups = set(unique_groups.tolist())
    successors = {}
    for group_id in unique_groups:
        group_id = int(group_id)
        if group_id >= next_groups.shape[0]:
            continue
        candidates = set(
            int(value)
            for value in np.unique(next_groups[group_id])
            if value != INVALID_GROUP_ID
        )
        candidates = sorted(candidates & valid_groups - {group_id})
        if candidates:
            successors[group_id] = candidates
    return successors


def _chain_polylines(group_polylines, successor_map):
    has_predecessor = {value for values in successor_map.values() for value in values}
    roots = sorted(set(group_polylines) - has_predecessor)
    result = []
    visited = set()

    def walk(group_id, prefix):
        if group_id in visited or group_id not in group_polylines:
            if prefix:
                result.append(np.concatenate(prefix, axis=0))
            return
        visited.add(group_id)
        polyline = group_polylines[group_id]
        if prefix and np.linalg.norm(polyline[0] - prefix[-1][-1]) < 1.0:
            polyline = polyline[1:]
        parts = prefix + [polyline]
        successors = successor_map.get(group_id, [])
        if successors:
            for successor in successors:
                walk(successor, list(parts))
        else:
            result.append(np.concatenate(parts, axis=0))

    for root in roots:
        walk(root, [])
    for group_id in sorted(group_polylines):
        if group_id not in visited:
            result.append(group_polylines[group_id])
    return result


def stitch_lane_segments(
    points, masks, groups, next_groups, *, max_lines=8, max_points=128
):
    lines = np.zeros((max_lines, max_points, 2), dtype=np.float32)
    line_masks = np.zeros(max_lines, dtype=np.bool_)
    valid = masks & (groups != INVALID_GROUP_ID)
    if not valid.any():
        return lines, line_masks
    valid_indices = np.nonzero(valid)[0]
    valid_groups = groups[valid_indices]
    _, first = np.unique(valid_groups, return_index=True)
    unique_groups = valid_groups[np.sort(first)]
    group_polylines = _build_group_polylines(
        points, valid_indices, valid_groups, unique_groups
    )
    successor_map = _build_successor_map(next_groups, unique_groups)
    polylines = (
        _chain_polylines(group_polylines, successor_map)
        if successor_map
        else [group_polylines[int(group_id)] for group_id in unique_groups]
    )
    selected = np.argsort([float(np.abs(polyline[:, 1]).min()) for polyline in polylines])[
        :max_lines
    ]
    for output_index, polyline_index in enumerate(selected):
        polyline = polylines[polyline_index]
        if len(polyline) >= max_points:
            indices = np.linspace(0, len(polyline) - 1, max_points, dtype=int)
            lines[output_index] = polyline[indices]
        else:
            lines[output_index, : len(polyline)] = polyline
            lines[output_index, len(polyline) :] = polyline[-1]
        line_masks[output_index] = True
    return lines, line_masks


class PackedDataset(IterableDataset):
    camera_sensors: tuple[str, ...] = ()
    required_fields: tuple[str, ...] = BASE_PACKED_FIELDS

    def __init__(self, config: dict[str, Any], *, world_size: int = 1):
        dataset = config["dataset"]
        self.mode = str(dataset["mode"])
        self.world_size = max(int(world_size), 1)
        self.epoch = 0
        self.shuffle = bool(dataset["shuffle"])
        self.repeat = bool(dataset.get("repeat", True))
        self.samples_per_shard = int(dataset["samples_per_shard"])
        self.temporal_stride = int(dataset.get("temporal_stride", 1))
        self.shards = self._resolve_shards(dataset["dataset_files"][self.mode])
        if not self.shards:
            raise ValueError("No packed dataset shards were resolved.")
        if self.world_size > len(self.shards):
            raise RuntimeError(
                f"Number of shards ({len(self.shards)}) is smaller than WORLD_SIZE "
                f"({self.world_size})."
            )
        self.samples_per_epoch = math.ceil(
            len(self.shards) * self.samples_per_shard / self.world_size
        )
        self.metadata = PackedMetadata(
            frame_gap=int(dataset.get("frame_gap", 2)),
            frame_time_interval=float(dataset.get("frame_time_interval", 0.2)),
            history_length=int(dataset["history_length"]),
            future_length=int(dataset["future_length"]),
            max_agents=int(dataset["max_agents"]),
            max_lanes_centers=int(dataset["max_lanes_centers"]),
            max_lanes_other=int(dataset["max_lanes_other"]),
            route_points=int(dataset["route_points"]),
        )
        self.camera_sensor_fqdns = validate_camera_sensor_fqdns(
            dataset["camera_sensor_fqdns"], self.camera_sensors
        )
        self.camera_sensor_indices = tuple(
            self.camera_sensors.index(sensor) for sensor in self.camera_sensor_fqdns
        )
        self.camera_patch_size = int(dataset["camera_patch_size"])

    def _resolve_shards(self, paths: Sequence[str]) -> list[str]:
        shards = []
        for raw_path in paths:
            path = Path(raw_path)
            if not path.exists():
                raise FileNotFoundError(f"Dataset path not found: {path}")
            candidates = sorted(path.glob("*.tar")) if path.is_dir() else [path]
            for candidate in candidates:
                shards.append(str(candidate))
        return shards

    def __len__(self) -> int:
        return self.samples_per_epoch

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        seed = self.epoch
        dataset = wds.WebDataset(
            self.shards,
            resampled=False,
            shardshuffle=max(1, len(self.shards)) if self.shuffle else False,
            detshuffle=self.shuffle,
            seed=seed,
            handler=wds.handlers.warn_and_continue,
            nodesplitter=wds.split_by_node,
            workersplitter=wds.split_by_worker,
            empty_check=False,
        )
        if self.shuffle:
            dataset = dataset.shuffle(1024, initial=512)
        if self.repeat:
            dataset = dataset.repeat()
        for sample in dataset:
            npz_key = next((key for key in sample if key.endswith(".npz")), None)
            if npz_key is None:
                continue
            yield self._decode_sample(sample, npz_key)

    def _decode_sample(self, sample: dict[str, Any], npz_key: str) -> dict[str, Any]:
        with np.load(io.BytesIO(sample[npz_key]), allow_pickle=False) as npz_file:
            arrays = {name: npz_file[name] for name in npz_file.files}
        missing = [field for field in self.required_fields if field not in arrays]
        if missing:
            raise KeyError(f"Packed sample {sample.get('__key__')} is missing fields: {missing}")
        arrays = self._downsample_temporal_arrays(arrays)

        metadata = {}
        if self.mode != "train":
            metadata_bytes = sample.get("meta.json")
            metadata = json.loads(metadata_bytes) if metadata_bytes else {}
        
        metadata["sample_key"] = sample.get("__key__", "unknown")
        metadata["shard_path"] = sample.get("__url__", "unknown")
        result = self._build_tensor_result(arrays)
        result["metadata"] = metadata
        self._attach_camera_data(result, arrays)
        return result

    def _downsample_temporal_arrays(
        self, arrays: dict[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        if self.temporal_stride == 1:
            return arrays

        stride = self.temporal_stride
        result = arrays.copy()

        for field in (
            "objects_history_states",
            "objects_history_masks",
            "lanes_centers_tl_states",
            "lanes_centers_tl_masks",
            "other_lanes_tl_states",
            "other_lanes_tl_masks",
        ):
            if field in arrays:
                result[field] = arrays[field][:, ::stride]

        for field in ("objects_future_states", "objects_future_masks"):
            result[field] = arrays[field][:, stride - 1 :: stride]

        return result

    def _build_tensor_result(self, arrays: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
        raise NotImplementedError

    def _attach_camera_data(self, result, arrays) -> None:
        indices = list(self.camera_sensor_indices)
        camera_images = np.stack(
            [
                decode_camera_image(
                    arrays[f"cam_jpeg_{index}"].tobytes(),
                )
                for index in self.camera_sensor_indices
            ],
            axis=0,
        )
        result["camera_images"] = torch.from_numpy(camera_images)
        result["camera_rays"] = camera_rays_from_calib(
            np.asarray(arrays["calib_intrinsic"])[indices],
            np.asarray(arrays["calib_distortion"])[indices],
            self.camera_patch_size,
            image_height=camera_images.shape[1],
            image_width=camera_images.shape[2],
        )
        result["camera_sensor2vehicle"] = torch.from_numpy(
            np.asarray(arrays["calib_sensor2vehicle"], dtype=np.float32)[indices]
        )


def base_tensor_result(arrays: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
    float_fields = (
        "lanes_centers_points",
        "lanes_centers_attributes",
        "other_lanes_points",
        "other_lanes_attributes",
        "route_points",
        "objects_history_states",
        "objects_future_states",
    )
    bool_fields = (
        "lanes_centers_masks",
        "other_lanes_masks",
        "objects_history_masks",
        "objects_future_masks",
    )
    result = {
        field: torch.from_numpy(np.asarray(arrays[field], dtype=np.float32))
        for field in float_fields
    }
    result.update(
        {
            field: torch.from_numpy(np.asarray(arrays[field], dtype=np.bool_))
            for field in bool_fields
        }
    )
    result["ego_index"] = torch.tensor(int(arrays["ego_index"][0]), dtype=torch.int64)
    result["occupancy_grid"] = torch.from_numpy(
        np.asarray(arrays["occupancy_grid"], dtype=np.uint8)
    )
    groups = np.asarray(
        arrays.get(
            "lanes_centers_groups",
            np.full(arrays["lanes_centers_points"].shape[0], INVALID_GROUP_ID, np.uint32),
        )
    )
    lines, masks = stitch_lane_segments(
        arrays["lanes_centers_points"],
        arrays["lanes_centers_masks"],
        groups,
        arrays.get("lanes_centers_next_groups"),
    )
    result["lanes_centers_lines"] = torch.from_numpy(lines)
    result["lanes_centers_lines_mask"] = torch.from_numpy(masks)
    return result


class BaseDataLoader:
    dataset_class: type[PackedDataset]

    def __init__(self, config: dict[str, Any], device, ddp_world_size: int = 1):
        self.device = torch.device(device)
        self.mode = str(config["dataset"]["mode"])
        self.global_batch_size = int(config["dataset"]["batch_size"])
        self.batch_size = resolve_per_rank_batch_size(
            self.global_batch_size, ddp_world_size
        )
        self.epoch = 0
        self.torch_dataset = self.dataset_class(config, world_size=ddp_world_size)
        self.metadata = self.torch_dataset.metadata
        self.max_agents = self.metadata.max_agents
        self.frame_time_interval = self.metadata.frame_time_interval
        self.history_length = self.metadata.history_length
        self.future_length = self.metadata.future_length
        workers = int(config["dataset"]["num_workers"])
        self.torch_loader = torch.utils.data.DataLoader(
            self.torch_dataset,
            batch_size=self.batch_size,
            num_workers=workers,
            collate_fn=self._collate,
            shuffle=False,
            drop_last=bool(config["dataset"]["drop_last"]),
            pin_memory=True,
            persistent_workers=workers > 0,
            in_order=False,
        )
        self.loader_iterator = None
        dataset = config["dataset"]
        self.camera_depth = torch.from_numpy(
            camera_depth_bins(
                int(dataset["camera_depth_num"]),
                float(dataset["camera_depth_start"]),
                float(dataset["camera_depth_end"]),
            )
        ).to(self.device)

    @staticmethod
    def _collate(samples):
        return {
            key: (
                [value if value is not None else {} for value in values]
                if key == "metadata"
                else torch.utils.data.default_collate(values)
            )
            for key in samples[0]
            for values in [[sample[key] for sample in samples]]
        }

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.torch_loader)

    def __next__(self):
        while True:
            try:
                if self.loader_iterator is None:
                    self.loader_iterator = iter(self.torch_loader)
                return self.to_device(next(self.loader_iterator))
            except StopIteration:
                if not self.torch_dataset.repeat:
                    raise
                self.epoch += 1
                self.torch_dataset.set_epoch(self.epoch)
                self.loader_iterator = iter(self.torch_loader)

    def to_device(self, batch):
        batch["camera_images"] = batch["camera_images"].to(
            self.device, non_blocking=True
        )
        rays = batch.pop("camera_rays").to(self.device, non_blocking=True)
        sensor2vehicle = batch.pop("camera_sensor2vehicle").to(
            self.device, non_blocking=True
        )
        batch["camera_geometry"] = metric_points_from_rays(
            rays, sensor2vehicle, self.camera_depth
        )
        return batch
