"""Train-only off-road route and command augmentation for the nuPlan Teacher."""
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

EPS = 1e-6


@dataclass
class GoalCommandSampleResult:
    route_points: torch.Tensor
    route_mask: torch.Tensor
    student_command: torch.Tensor
    accepted_augmented: bool


def validate_goal_command_augmentation_config(config):
    augmentation = config["augmentation"]["goal_command"]
    probability = float(augmentation["original_prob"])
    if not 0 <= probability <= 1:
        raise ValueError("original_prob must lie in [0,1].")
    _positive(augmentation, "goal_command", ("command_distance_m", "command_lateral_threshold_m"))
    _validate_off_road_config(augmentation["off_road_goal"])


def _positive(config: dict[str, Any], mode: str, names: tuple[str, ...]) -> None:
    for name in names:
        value = float(config.get(name, 0.0))
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{mode}.{name} must be positive.")


def _range(config: dict[str, Any], mode: str, name: str) -> tuple[float, float]:
    value = config.get(name)
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{mode}.{name} must contain [min,max].")
    lower, upper = (float(item) for item in value)
    if not (math.isfinite(lower) and lower > 0.0 and upper > lower):
        raise ValueError(f"{mode}.{name} must satisfy 0 < min < max.")
    return lower, upper


def _validate_route_geometry_config(config: dict[str, Any], mode: str) -> None:
    _positive(
        config,
        mode,
        (
            "route_resample_spacing_m",
            "max_route_segment_m",
            "max_route_turn_deg",
        ),
    )
    if not 0.0 < float(config["max_route_turn_deg"]) < 90.0:
        raise ValueError(f"{mode}.max_route_turn_deg must be within (0,90).")


def _validate_off_road_config(config: dict[str, Any]) -> None:
    target_min, _ = _range(config, "off_road_goal", "target_progress_range_m")
    _range(config, "off_road_goal", "lateral_offset_range_m")
    transition_start = float(config.get("transition_start_m", -1.0))
    transition_end_min, _ = _range(config, "off_road_goal", "transition_end_range_m")
    if not 0.0 <= transition_start < transition_end_min:
        raise ValueError(
            "off_road_goal.transition_start_m must be before the transition end."
        )
    if transition_end_min > target_min:
        raise ValueError(
            "off_road_goal.transition_end_range_m must not start after the target "
            "progress range."
        )
    _positive(
        config,
        "off_road_goal",
        ("tangent_smoothing_m", "max_route_ego_distance_m"),
    )
    _validate_route_geometry_config(config, "off_road_goal")


def _legacy_route_mask(route: np.ndarray) -> np.ndarray:
    return np.any(np.abs(route) > 0.0, axis=-1)


def _route_polyline(route: np.ndarray) -> np.ndarray | None:
    mask = _legacy_route_mask(route)
    valid_indices = np.flatnonzero(mask)
    if valid_indices.size < 2:
        return None
    # Packed routes are left-aligned. Include embedded (0, 0) points while
    # excluding only trailing zero padding.
    points = np.asarray(route[: valid_indices[-1] + 1], dtype=np.float64)
    return _deduplicate(points)


def _deduplicate(points: np.ndarray) -> np.ndarray:
    if len(points) < 2:
        return points
    keep = np.concatenate(
        (np.ones(1, dtype=np.bool_), np.linalg.norm(np.diff(points, axis=0), axis=1) > EPS)
    )
    return points[keep]


def _polyline_geometry(points: np.ndarray):
    vectors = np.diff(points, axis=0)
    lengths = np.linalg.norm(vectors, axis=1)
    valid = lengths > EPS
    if not valid.any():
        return None
    starts = points[:-1][valid]
    vectors = vectors[valid]
    lengths = lengths[valid]
    progress = np.concatenate(([0.0], np.cumsum(lengths)))
    return starts, vectors, lengths, progress


def _project_point(points: np.ndarray, point: np.ndarray):
    geometry = _polyline_geometry(points)
    if geometry is None:
        return None
    starts, vectors, lengths, progress = geometry
    relative = point[None] - starts
    fractions = np.clip(
        np.sum(relative * vectors, axis=1) / np.maximum(lengths * lengths, EPS),
        0.0,
        1.0,
    )
    projected = starts + fractions[:, None] * vectors
    distances = np.linalg.norm(projected - point[None], axis=1)
    index = int(np.argmin(distances))
    tangent = vectors[index] / lengths[index]
    return (
        float(progress[index] + fractions[index] * lengths[index]),
        projected[index],
        tangent,
        float(distances[index]),
    )


def _sample_polyline(points: np.ndarray, query: np.ndarray):
    geometry = _polyline_geometry(points)
    if geometry is None:
        raise ValueError("Cannot sample a degenerate polyline.")
    starts, vectors, lengths, progress = geometry
    total = float(progress[-1])
    query = np.clip(np.asarray(query, dtype=np.float64), 0.0, total)
    indices = np.searchsorted(progress[1:], query, side="left")
    indices = np.minimum(indices, len(lengths) - 1)
    fractions = (query - progress[indices]) / lengths[indices]
    sampled = starts[indices] + fractions[:, None] * vectors[indices]
    tangents = vectors[indices] / lengths[indices, None]
    return sampled, tangents


def _total_length(points: np.ndarray) -> float:
    geometry = _polyline_geometry(points)
    return 0.0 if geometry is None else float(geometry[-1][-1])


def _resample_augmented_route(
    points: np.ndarray,
    *,
    max_points: int,
    spacing_m: float,
    mandatory_progress: np.ndarray | None = None,
):
    points = _deduplicate(points)
    total = _total_length(points)
    if len(points) < 2 or total <= EPS:
        return None
    mandatory = np.asarray(
        [] if mandatory_progress is None else mandatory_progress, dtype=np.float64
    )
    mandatory = np.unique(np.clip(mandatory, 0.0, total))
    uniform_count = max(
        2,
        min(
            max_points - len(mandatory),
            int(math.ceil(total / spacing_m)) + 1,
        ),
    )
    progress = np.unique(
        np.concatenate((np.linspace(0.0, total, uniform_count), mandatory))
    )
    if len(progress) > max_points:
        raise ValueError("Mandatory route samples exceed the fixed route capacity.")
    count = len(progress)
    sampled, _ = _sample_polyline(points, progress)
    route = np.zeros((max_points, 2), dtype=np.float32)
    mask = np.zeros(max_points, dtype=np.bool_)
    route[:count] = sampled.astype(np.float32)
    mask[:count] = True
    return route, mask


def _proper_segments_intersect(a, b, c, d) -> bool:
    def cross(origin, first, second):
        lhs = first - origin
        rhs = second - origin
        return float(lhs[0] * rhs[1] - lhs[1] * rhs[0])

    ab_c = cross(a, b, c)
    ab_d = cross(a, b, d)
    cd_a = cross(c, d, a)
    cd_b = cross(c, d, b)
    return ab_c * ab_d < -EPS and cd_a * cd_b < -EPS


def _valid_constructed_route(route: np.ndarray, mask: np.ndarray, config) -> bool:
    points = np.asarray(route[mask], dtype=np.float64)
    if len(points) < 2 or not np.isfinite(points).all():
        return False
    vectors = np.diff(points, axis=0)
    lengths = np.linalg.norm(vectors, axis=1)
    if np.any(lengths <= EPS) or np.any(lengths > float(config["max_route_segment_m"])):
        return False
    if len(vectors) >= 2:
        unit = vectors / lengths[:, None]
        cosine = np.clip(np.sum(unit[:-1] * unit[1:], axis=1), -1.0, 1.0)
        turns = np.degrees(np.arccos(cosine))
        if np.any(turns > float(config["max_route_turn_deg"])):
            return False
    for first in range(len(points) - 1):
        for second in range(first + 2, len(points) - 1):
            if _proper_segments_intersect(
                points[first],
                points[first + 1],
                points[second],
                points[second + 1],
            ):
                return False
    return True


def _smoothed_tangents(
    points: np.ndarray,
    progress: np.ndarray,
    fallback: np.ndarray,
    smoothing_m: float,
) -> np.ndarray:
    tangents = np.empty_like(points)
    half_window = smoothing_m / 2.0
    for index, value in enumerate(progress):
        left = int(np.searchsorted(progress, value - half_window, side="left"))
        right = int(np.searchsorted(progress, value + half_window, side="right")) - 1
        left = max(0, min(left, len(points) - 1))
        right = max(0, min(right, len(points) - 1))
        delta = points[right] - points[left]
        norm = float(np.linalg.norm(delta))
        tangents[index] = fallback[index] if norm <= EPS else delta / norm
    return tangents


def _build_normal_offset_route(
    route: np.ndarray,
    *,
    lateral_offset_m: float,
    transition_start_m: float,
    transition_end_m: float,
    forward_length_m: float,
    tangent_smoothing_m: float,
    route_resample_spacing_m: float,
    max_points: int,
):
    projection = _project_point(route, np.zeros(2, dtype=np.float64))
    if projection is None:
        return None
    route_progress, route_projected, _, _ = projection
    available = _total_length(route) - route_progress
    forward_length = min(float(forward_length_m), available)
    if forward_length < transition_end_m - EPS:
        return None

    geometry = _polyline_geometry(route)
    if geometry is None:
        return None
    vertex_progress = geometry[-1]
    prefix = route[vertex_progress < route_progress - EPS]
    prefix = np.concatenate((prefix, route_projected[None]), axis=0)

    construction_spacing = min(float(route_resample_spacing_m), 1.0)
    count = max(2, int(math.ceil(forward_length / construction_spacing)) + 1)
    relative_progress = np.linspace(0.0, forward_length, count)
    original_points, raw_tangents = _sample_polyline(
        route, route_progress + relative_progress
    )
    tangents = _smoothed_tangents(
        original_points,
        relative_progress,
        raw_tangents,
        float(tangent_smoothing_m),
    )
    normals = np.stack((-tangents[:, 1], tangents[:, 0]), axis=1)
    blend = np.clip(
        (relative_progress - transition_start_m)
        / (transition_end_m - transition_start_m),
        0.0,
        1.0,
    )
    blend = blend * blend * (3.0 - 2.0 * blend)
    future = original_points + lateral_offset_m * blend[:, None] * normals
    augmented = np.concatenate((prefix, future[1:]), axis=0)
    augmented_projection = _project_point(augmented, np.zeros(2, dtype=np.float64))
    if augmented_projection is None:
        return None
    ego_progress = float(augmented_projection[0])
    resampled = _resample_augmented_route(
        augmented,
        max_points=max_points,
        spacing_m=float(route_resample_spacing_m),
        mandatory_progress=np.asarray(
            [
                0.0,
                ego_progress,
                ego_progress + transition_start_m,
                ego_progress + transition_end_m,
                _total_length(augmented),
            ]
        ),
    )
    if resampled is None:
        return None
    return resampled


def _command_from_route(
    route: np.ndarray,
    mask: np.ndarray,
    *,
    distance_m: float,
    lateral_threshold_m: float,
) -> np.ndarray | None:
    points = _deduplicate(np.asarray(route[mask], dtype=np.float64))
    projection = _project_point(points, np.zeros(2, dtype=np.float64))
    if projection is None:
        return None
    progress, projected, tangent, _ = projection
    target_progress = min(progress + distance_m, _total_length(points))
    target, _ = _sample_polyline(points, np.asarray([target_progress]))
    normal = np.asarray([-tangent[1], tangent[0]])
    lateral = float(np.dot(target[0] - projected, normal))
    command = np.zeros(4, dtype=np.float32)
    if lateral >= lateral_threshold_m:
        command[0] = 1.0
    elif lateral <= -lateral_threshold_m:
        command[2] = 1.0
    else:
        command[1] = 1.0
    return command


def _augment_goal_command_arrays(route_values, command_values, config, rng):
    original = (route_values.copy(), _legacy_route_mask(route_values), command_values.copy(), False)
    # The two nonzero probabilities preserve the original four-mode RNG draw.
    if rng.choice(2, p=[config["original_prob"], 1.0 - config["original_prob"]]) == 0:
        return original
    route = _route_polyline(route_values)
    if route is None:
        return original
    mode_config = config["off_road_goal"]
    projection = _project_point(route, np.zeros(2, dtype=np.float64))
    if projection is None or projection[3] > float(mode_config["max_route_ego_distance_m"]):
        return original
    available = _total_length(route) - float(projection[0])
    target_lower, target_upper = (float(v) for v in mode_config["target_progress_range_m"])
    if available < target_lower - EPS:
        return original
    target_progress = float(rng.uniform(target_lower, min(target_upper, available)))
    transition_lower, transition_upper = (float(v) for v in mode_config["transition_end_range_m"])
    transition_end = float(rng.uniform(transition_lower, min(transition_upper, target_progress)))
    offset_lower, offset_upper = (float(v) for v in mode_config["lateral_offset_range_m"])
    lateral_offset = (-1.0 if rng.random() < 0.5 else 1.0) * float(rng.uniform(offset_lower, offset_upper))
    augmented = _build_normal_offset_route(
        route,
        lateral_offset_m=lateral_offset,
        transition_start_m=float(mode_config["transition_start_m"]),
        transition_end_m=transition_end,
        forward_length_m=target_progress,
        tangent_smoothing_m=float(mode_config["tangent_smoothing_m"]),
        route_resample_spacing_m=float(mode_config["route_resample_spacing_m"]),
        max_points=route_values.shape[0],
    )
    if augmented is None:
        return original
    route, mask = augmented
    if not _valid_constructed_route(route, mask, mode_config):
        return original
    command = _command_from_route(
        route, mask,
        distance_m=float(config["command_distance_m"]),
        lateral_threshold_m=float(config["command_lateral_threshold_m"]),
    )
    if command is None:
        return original
    return route, mask, command, True


def augment_goal_command_sample(sample, config, rng):
    route, mask, command, accepted = _augment_goal_command_arrays(
        sample["route_points"].numpy(), sample["command"].numpy(), config, rng,
    )
    return GoalCommandSampleResult(
        torch.from_numpy(route).to(dtype=sample["route_points"].dtype),
        torch.from_numpy(mask),
        torch.from_numpy(command).to(dtype=sample["command"].dtype),
        accepted,
    )
