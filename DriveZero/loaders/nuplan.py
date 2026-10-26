"""Packed nuPlan-data loader."""

import json

import numpy as np
import torch

from augmentations.nuplan_goal_command import augment_goal_command_sample

from .common import (
    BASE_PACKED_FIELDS,
    BaseDataLoader,
    PackedDataset,
    base_tensor_result,
)


NUPLAN_CAMERA_SENSORS = (
    "CAM_F0",
    "CAM_L0",
    "CAM_L1",
    "CAM_L2",
    "CAM_R0",
    "CAM_R1",
    "CAM_R2",
    "CAM_B0",
)
TRAFFIC_LIGHT_FIELDS = (
    "lanes_centers_tl_states",
    "lanes_centers_tl_masks",
    "other_lanes_tl_states",
    "other_lanes_tl_masks",
)


class NuplanDataset(PackedDataset):
    camera_sensors = NUPLAN_CAMERA_SENSORS
    required_fields = BASE_PACKED_FIELDS + TRAFFIC_LIGHT_FIELDS + ("command",)

    def __init__(self, config, *, world_size=1):
        super().__init__(config, world_size=world_size)
        augmentation = config.get("augmentation", {}).get("goal_command", {})
        self.goal_command_augmentation = augmentation
        self.apply_goal_command_augmentation = self.mode == "train"
        self._augmentation_rng = None

    def __iter__(self):
        if self.apply_goal_command_augmentation:
            self._augmentation_rng = np.random.default_rng(torch.initial_seed())
        yield from super().__iter__()

    def _decode_sample(self, sample, npz_key):
        result = super()._decode_sample(sample, npz_key)
        if self.mode == "train":
            result["metadata"] = {
                "anchor_token": json.loads(sample["meta.json"])["anchor_token"]
            }
        if self.apply_goal_command_augmentation:
            if self._augmentation_rng is None:
                raise RuntimeError("Goal-command augmentation RNG is not initialized.")
            augmented = augment_goal_command_sample(
                result,
                self.goal_command_augmentation,
                self._augmentation_rng,
            )
            result["intent_route_points"] = augmented.route_points
            result["intent_route_mask"] = augmented.route_mask
            result["intent_command"] = augmented.student_command
            use_augmented_route = augmented.accepted_augmented and (
                int(augmented.student_command.argmax()) != int(result["command"].argmax())
            )
            result["intent_route_augmented"] = torch.tensor(use_augmented_route, dtype=torch.bool)
        return result

    def _build_tensor_result(self, arrays):
        result = base_tensor_result(arrays)
        for field in ("lanes_centers_tl_states", "other_lanes_tl_states"):
            result[field] = torch.from_numpy(np.asarray(arrays[field], dtype=np.float32))
        for field in ("lanes_centers_tl_masks", "other_lanes_tl_masks"):
            result[field] = torch.from_numpy(np.asarray(arrays[field], dtype=np.bool_))
        result["command"] = torch.from_numpy(np.asarray(arrays["command"], dtype=np.float32))
        return result


class NuplanDataloader(BaseDataLoader):
    dataset_class = NuplanDataset
