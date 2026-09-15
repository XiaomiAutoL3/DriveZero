"""Map feature helpers shared by DriveRL inference and data export."""

from __future__ import annotations

import hashlib
import heapq
import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np

try:
    from nuplan.common.actor_state.state_representation import Point2D
    from nuplan.common.maps.abstract_map_objects import LaneGraphEdgeMapObject
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer, StopLineType
except ImportError:
    Point2D = None
    LaneGraphEdgeMapObject = object
    SemanticMapLayer = None
    StopLineType = None

try:
    from shapely import make_valid
    from shapely.geometry import Point
    from shapely.ops import unary_union
except ImportError:
    make_valid = None
    Point = None
    unary_union = None

ROUTE_POINTS = 100


MAX_MAP_POINTS_PER_OBJECT = 20


MAX_MAP_POLYGON_POINTS_PER_OBJECT = 256


DEFAULT_MAP_SIMPLIFICATION_AREA_THRESHOLD_M2 = 0.12


INVALID_LANE_ID = np.int64(-1)


INVALID_GROUP_ID = np.iinfo(np.uint32).max


DEFAULT_MAP_RADIUS_M = 10.0


DEFAULT_MAP_QUERY_POINTS = 15


LANE_TYPE_CURB = 0.0


LANE_TYPE_LANE_CENTER_LINE = 1.0


LANE_TYPE_ROAD_BOUNDARY = 5.0


LANE_TYPE_STOP_LINE = 6.0


LANE_TYPE_CROSSWALK = 11.0


TL_STATUS_TO_INDEX = {
    "unknown": 0,
    "red": 1,
    "yellow": 2,
    "green": 3,
}


@dataclass(frozen=True)
class FrameRow:
    token: bytes
    token_hex: str
    timestamp_us: int
    scene_token: bytes | None
    x: float
    y: float
    yaw: float
    vx: float
    vy: float
    ax: float
    ay: float
    yaw_rate: float


@dataclass(frozen=True)
class MapSegmentCandidate:
    p0: tuple[float, float]
    p1: tuple[float, float]
    group_id: int
    raw_id: object
    lane_type: float
    speed_mps: float
    distance_m: float
    order: int
    connector_key: int | None = None


def _wrap_angle(angle: np.ndarray | float) -> np.ndarray | float:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def _rotate_to_anchor(dx: float, dy: float, anchor_yaw: float) -> tuple[float, float]:
    c = math.cos(anchor_yaw)
    s = math.sin(anchor_yaw)
    return c * dx + s * dy, -s * dx + c * dy


def _global_to_anchor_xy(x: float, y: float, anchor: FrameRow) -> tuple[float, float]:
    return _rotate_to_anchor(x - anchor.x, y - anchor.y, anchor.yaw)


def _stable_int64_id(raw_id: object) -> np.int64:
    text = str(raw_id)
    try:
        return np.int64(int(text))
    except ValueError:
        digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
        return np.int64(
            int.from_bytes(digest, byteorder="little", signed=False) & ((1 << 63) - 1)
        )


def _dedupe_route_points(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if len(points) == 0:
        return points
    points = points[np.isfinite(points).all(axis=1)]
    nonzero = np.flatnonzero(np.any(np.abs(points) > 0.0, axis=1))
    if len(nonzero) == 0:
        return points[:0]
    points = points[: int(nonzero[-1]) + 1]
    if len(points) < 2:
        return points
    deltas = np.linalg.norm(np.diff(points, axis=0), axis=1)
    return points[np.concatenate(([True], deltas > 1e-3))]


def _resample_polyline(points: np.ndarray, num_points: int) -> np.ndarray:
    route = np.zeros((num_points, 2), dtype=np.float32)
    points = _dedupe_route_points(points)
    if len(points) == 0:
        return route
    if len(points) == 1:
        route[0] = points[0]
        return route

    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    progress = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    if progress[-1] <= 1e-3:
        route[: len(points)] = points
        return route

    target = np.linspace(0.0, float(progress[-1]), num_points)
    route[:] = np.stack(
        [
            np.interp(target, progress, points[:, 0]),
            np.interp(target, progress, points[:, 1]),
        ],
        axis=1,
    ).astype(np.float32)
    return route


def _smooth_route_points(route_points: np.ndarray, window: int = 5) -> np.ndarray:
    route = np.zeros_like(route_points, dtype=np.float32)
    points = _dedupe_route_points(route_points)
    if len(points) == 0:
        return route
    if len(points) < 3:
        route[: len(points)] = points
        return route

    fitted = _resample_polyline(points, len(points))

    window = min(int(window), len(fitted))
    if window >= 3:
        if window % 2 == 0:
            window -= 1
        kernel = np.ones(window, dtype=np.float32) / float(window)
        pad = window // 2
        padded = np.pad(fitted, ((pad, pad), (0, 0)), mode="edge")
        fitted = np.stack(
            [
                np.convolve(padded[:, 0], kernel, mode="valid"),
                np.convolve(padded[:, 1], kernel, mode="valid"),
            ],
            axis=1,
        ).astype(np.float32)
        fitted[0] = points[0]
        fitted[-1] = points[-1]

    route[: len(fitted)] = fitted
    return route


def _fit_route_polyline(
    points: np.ndarray, num_points: int = ROUTE_POINTS
) -> np.ndarray:
    points = _dedupe_route_points(points)
    if len(points) == 0:
        return np.zeros((num_points, 2), dtype=np.float32)
    # Smooth at map resolution first, then resample to a dense fixed-size route.
    smoothed = _smooth_route_points(points, window=5)
    return _resample_polyline(smoothed, num_points)


def _empty_map_arrays(
    total_frames: int,
    *,
    max_lanes_centers: int,
    max_lanes_other: int,
    include_traffic_lights: bool = True,
) -> dict[str, np.ndarray]:
    arrays = {
        "lanes_centers_points": np.zeros((max_lanes_centers, 2, 2), dtype=np.float32),
        "lanes_centers_masks": np.zeros((max_lanes_centers,), dtype=np.bool_),
        "lanes_centers_attributes": np.zeros((max_lanes_centers, 3), dtype=np.float32),
        "lanes_centers_groups": np.full(
            (max_lanes_centers,), INVALID_GROUP_ID, dtype=np.uint32
        ),
        "lanes_centers_next_groups": np.full(
            (max_lanes_centers, 4), INVALID_GROUP_ID, dtype=np.uint32
        ),
        "lanes_centers_left_groups": np.full(
            (max_lanes_centers, 2), INVALID_GROUP_ID, dtype=np.uint32
        ),
        "lanes_centers_right_groups": np.full(
            (max_lanes_centers, 2), INVALID_GROUP_ID, dtype=np.uint32
        ),
        "lanes_centers_ids": np.full(
            (max_lanes_centers,), INVALID_LANE_ID, dtype=np.int64
        ),
        "other_lanes_points": np.zeros((max_lanes_other, 2, 2), dtype=np.float32),
        "other_lanes_masks": np.zeros((max_lanes_other,), dtype=np.bool_),
        "other_lanes_attributes": np.zeros((max_lanes_other, 3), dtype=np.float32),
        "other_lanes_groups": np.full(
            (max_lanes_other,), INVALID_GROUP_ID, dtype=np.uint32
        ),
        "other_lanes_next_groups": np.full(
            (max_lanes_other, 4), INVALID_GROUP_ID, dtype=np.uint32
        ),
        "other_lanes_ids": np.full((max_lanes_other,), INVALID_LANE_ID, dtype=np.int64),
    }
    if include_traffic_lights:
        lanes_centers_tl_states = np.zeros(
            (max_lanes_centers, total_frames, 4), dtype=np.float32
        )
        lanes_centers_tl_states[..., TL_STATUS_TO_INDEX["unknown"]] = 1.0
        other_lanes_tl_states = np.zeros(
            (max_lanes_other, total_frames, 4), dtype=np.float32
        )
        other_lanes_tl_states[..., TL_STATUS_TO_INDEX["unknown"]] = 1.0
        arrays.update(
            {
                "lanes_centers_tl_states": lanes_centers_tl_states,
                "lanes_centers_tl_masks": np.zeros(
                    (max_lanes_centers, total_frames), dtype=np.bool_
                ),
                "other_lanes_tl_states": other_lanes_tl_states,
                "other_lanes_tl_masks": np.zeros(
                    (max_lanes_other, total_frames), dtype=np.bool_
                ),
            }
        )
    return arrays


def _line_coords(line_obj: object) -> list[tuple[float, float]]:
    linestring = getattr(line_obj, "linestring", None)
    if linestring is not None:
        return [(float(x), float(y)) for x, y, *_ in linestring.coords]
    discrete_path = getattr(line_obj, "discrete_path", None)
    if discrete_path is not None:
        return [(float(state.x), float(state.y)) for state in discrete_path]
    return []


def _polygon_exterior_coords(poly_obj: object) -> list[tuple[float, float]]:
    polygon = getattr(poly_obj, "polygon", poly_obj)
    exterior = getattr(polygon, "exterior", None)
    if exterior is None:
        return []
    return [(float(x), float(y)) for x, y, *_ in exterior.coords]


def _geometry_line_coords(geometry: object) -> list[list[tuple[float, float]]]:
    if geometry is None or getattr(geometry, "is_empty", False):
        return []
    geom_type = getattr(geometry, "geom_type", "")
    if geom_type == "LineString":
        return [[(float(x), float(y)) for x, y, *_ in geometry.coords]]
    if geom_type == "LinearRing":
        return [[(float(x), float(y)) for x, y, *_ in geometry.coords]]
    if geom_type == "MultiLineString":
        return [
            [(float(x), float(y)) for x, y, *_ in line.coords]
            for line in geometry.geoms
            if not getattr(line, "is_empty", False)
        ]
    if geom_type == "Polygon":
        coords = [[(float(x), float(y)) for x, y, *_ in geometry.exterior.coords]]
        coords.extend(
            [(float(x), float(y)) for x, y, *_ in interior.coords]
            for interior in geometry.interiors
        )
        return coords
    if geom_type in {"MultiPolygon", "GeometryCollection"}:
        output: list[list[tuple[float, float]]] = []
        for part in geometry.geoms:
            output.extend(_geometry_line_coords(part))
        return output
    return []


def _remove_immediate_backtracks(
    coords: list[tuple[float, float]],
    tolerance_m: float = 1e-6,
) -> list[tuple[float, float]]:
    """Remove zero-width A-B-A spikes while preserving the surrounding line or ring."""

    def points_match(first: tuple[float, float], second: tuple[float, float]) -> bool:
        return math.hypot(first[0] - second[0], first[1] - second[1]) <= tolerance_m

    cleaned = list(coords)
    while (
        len(cleaned) >= 4
        and points_match(cleaned[0], cleaned[-1])
        and points_match(cleaned[1], cleaned[-2])
    ):
        cleaned = cleaned[1:-1]

    output: list[tuple[float, float]] = []
    for point in cleaned:
        if len(output) >= 2 and points_match(output[-2], point):
            output.pop()
            continue
        if not output or not points_match(output[-1], point):
            output.append(point)
    return output


def _downsample_coords(
    coords: list[tuple[float, float]],
    max_points: int = MAX_MAP_POINTS_PER_OBJECT,
    area_threshold: float = DEFAULT_MAP_SIMPLIFICATION_AREA_THRESHOLD_M2,
) -> list[tuple[float, float]]:
    if max_points <= 0:
        return []
    if area_threshold <= 0.0:
        return coords[:max_points]

    is_closed = (
        len(coords) >= 4
        and math.hypot(coords[0][0] - coords[-1][0], coords[0][1] - coords[-1][1])
        < 1e-6
    )
    if is_closed:
        unique_coords = coords[:-1]
        sampled = _visvalingam_whyatt_downsample(
            unique_coords,
            closed=True,
            area_threshold=area_threshold,
        )
        target_unique_points = max(3, max_points - 1)
        if len(sampled) > target_unique_points:
            sampled = _visvalingam_whyatt_downsample(
                sampled,
                closed=True,
                target_points=target_unique_points,
            )
        if not sampled:
            return []
        return sampled + [sampled[0]]

    sampled = _visvalingam_whyatt_downsample(
        coords,
        closed=False,
        area_threshold=area_threshold,
    )
    target_points = max(2, max_points)
    if len(sampled) > target_points:
        sampled = _visvalingam_whyatt_downsample(
            sampled,
            closed=False,
            target_points=target_points,
        )
    return sampled


def _triangle_area(
    p0: tuple[float, float],
    p1: tuple[float, float],
    p2: tuple[float, float],
) -> float:
    return (
        abs((p1[0] - p0[0]) * (p2[1] - p0[1]) - (p2[0] - p0[0]) * (p1[1] - p0[1])) * 0.5
    )


def _visvalingam_whyatt_downsample(
    coords: list[tuple[float, float]],
    *,
    closed: bool,
    target_points: int | None = None,
    area_threshold: float | None = None,
) -> list[tuple[float, float]]:
    if target_points is None and area_threshold is None:
        return coords

    min_points = 3 if closed else 2
    if target_points is not None:
        target_points = max(min_points, min(target_points, len(coords)))
        if len(coords) <= target_points:
            return coords
    else:
        target_points = min_points
    prev_idx = list(range(-1, len(coords) - 1))
    next_idx = list(range(1, len(coords) + 1))
    if closed:
        prev_idx[0] = len(coords) - 1
        next_idx[-1] = 0
    else:
        prev_idx[0] = 0
        next_idx[-1] = len(coords) - 1

    removed = [False] * len(coords)
    versions = [0] * len(coords)
    heap: list[tuple[float, int, int]] = []

    def push_effective_area(idx: int) -> None:
        if removed[idx] or (not closed and (idx == 0 or idx == len(coords) - 1)):
            return
        versions[idx] += 1
        area = _triangle_area(coords[idx], coords[prev_idx[idx]], coords[next_idx[idx]])
        heapq.heappush(heap, (area, idx, versions[idx]))

    if closed:
        for idx in range(len(coords)):
            push_effective_area(idx)
    else:
        for idx in range(1, len(coords) - 1):
            push_effective_area(idx)

    retained = len(coords)
    while heap and retained > target_points:
        area, idx, version = heapq.heappop(heap)
        if removed[idx] or version != versions[idx]:
            continue
        if area_threshold is not None and area > area_threshold:
            break

        removed[idx] = True
        retained -= 1
        left = prev_idx[idx]
        right = next_idx[idx]
        next_idx[left] = right
        prev_idx[right] = left
        push_effective_area(left)
        push_effective_area(right)

    return [coord for idx, coord in enumerate(coords) if not removed[idx]]


def _point_to_segment_distance_sq(
    point: tuple[float, float],
    p0: tuple[float, float],
    p1: tuple[float, float],
) -> float:
    dx = p1[0] - p0[0]
    dy = p1[1] - p0[1]
    length_sq = dx * dx + dy * dy
    if length_sq <= 1e-12:
        px = point[0] - p0[0]
        py = point[1] - p0[1]
        return px * px + py * py
    t = ((point[0] - p0[0]) * dx + (point[1] - p0[1]) * dy) / length_sq
    t = min(1.0, max(0.0, t))
    closest_x = p0[0] + t * dx
    closest_y = p0[1] + t * dy
    px = point[0] - closest_x
    py = point[1] - closest_y
    return px * px + py * py


def _segment_distance_to_frames(
    p0: tuple[float, float],
    p1: tuple[float, float],
    frames: list[FrameRow],
) -> float:
    if not frames:
        return 0.0
    min_distance_sq = min(
        _point_to_segment_distance_sq((frame.x, frame.y), p0, p1) for frame in frames
    )
    return math.sqrt(min_distance_sq)


def _make_segment_candidates(
    *,
    coords: list[tuple[float, float]],
    distance_frames: list[FrameRow],
    group_id: int,
    raw_id: object,
    lane_type: float,
    speed_mps: float,
    order_start: int,
    connector_key: int | None = None,
) -> tuple[list[MapSegmentCandidate], int]:
    candidates: list[MapSegmentCandidate] = []
    order = order_start
    if len(coords) < 2:
        return candidates, order
    for p0, p1 in zip(coords[:-1], coords[1:]):
        if math.hypot(p1[0] - p0[0], p1[1] - p0[1]) < 1e-3:
            continue
        candidates.append(
            MapSegmentCandidate(
                p0=p0,
                p1=p1,
                group_id=group_id,
                raw_id=raw_id,
                lane_type=lane_type,
                speed_mps=speed_mps,
                distance_m=_segment_distance_to_frames(p0, p1, distance_frames),
                order=order,
                connector_key=connector_key,
            )
        )
        order += 1
    return candidates, order


def _write_segment_candidates(
    *,
    candidates: list[MapSegmentCandidate],
    anchor: FrameRow,
    points: np.ndarray,
    masks: np.ndarray,
    attributes: np.ndarray,
    groups: np.ndarray,
    ids: np.ndarray,
    max_segments: int,
) -> list[MapSegmentCandidate]:
    selected = sorted(candidates, key=lambda item: (item.distance_m, item.order))[
        :max_segments
    ]
    selected.sort(key=lambda item: item.order)
    for slot, candidate in enumerate(selected):
        x0, y0 = _global_to_anchor_xy(candidate.p0[0], candidate.p0[1], anchor)
        x1, y1 = _global_to_anchor_xy(candidate.p1[0], candidate.p1[1], anchor)
        points[slot, 0] = (x0, y0)
        points[slot, 1] = (x1, y1)
        masks[slot] = True
        attributes[slot] = (candidate.lane_type, candidate.speed_mps, 0.0)
        groups[slot] = (
            np.uint32(candidate.group_id)
            if candidate.group_id >= 0
            else INVALID_GROUP_ID
        )
        ids[slot] = _stable_int64_id(candidate.raw_id)
    return selected


def _edge_centerline_coords(edge: object) -> list[tuple[float, float]]:
    baseline_path = getattr(edge, "baseline_path", None)
    if baseline_path is None:
        return []
    return _line_coords(baseline_path)


def _query_frames_along_path(
    sampled_frames: list[FrameRow], num_points: int = 5
) -> list[FrameRow]:
    if not sampled_frames:
        return []
    if len(sampled_frames) <= num_points:
        return sampled_frames
    indices = (
        np.linspace(0, len(sampled_frames) - 1, num_points).round().astype(np.int64)
    )
    output: list[FrameRow] = []
    seen_indices: set[int] = set()
    for raw_idx in indices:
        idx = int(raw_idx)
        if idx in seen_indices:
            continue
        seen_indices.add(idx)
        output.append(sampled_frames[idx])
    return output


def _edge_distance_to_frames(edge: object, frames: list[FrameRow]) -> float:
    if Point is None:
        return 0.0
    baseline_path = getattr(edge, "baseline_path", None)
    linestring = getattr(baseline_path, "linestring", None)
    if linestring is None or not frames:
        return 0.0
    return min(float(linestring.distance(Point(frame.x, frame.y))) for frame in frames)


def _polygon_distance_to_frames(poly_obj: object, frames: list[FrameRow]) -> float:
    if Point is None:
        return 0.0
    polygon = getattr(poly_obj, "polygon", None)
    if polygon is None or not frames:
        return 0.0
    return min(float(polygon.distance(Point(frame.x, frame.y))) for frame in frames)


def _unique_map_objects(objects: Iterable[object]) -> list[object]:
    unique: dict[str, object] = {}
    for idx, obj in enumerate(objects):
        obj_id = str(getattr(obj, "id", idx))
        unique.setdefault(obj_id, obj)
    return list(unique.values())


def _merge_proximal_map_objects(
    *,
    map_api: object,
    query_frames: list[FrameRow],
    radius_m: float,
    layer_names: list[object],
) -> dict[object, list[object]]:
    merged: dict[object, list[object]] = {layer_name: [] for layer_name in layer_names}
    for frame in query_frames:
        proximal = map_api.get_proximal_map_objects(
            Point2D(frame.x, frame.y),
            radius_m,
            layer_names,
        )
        for layer_name in layer_names:
            merged[layer_name].extend(proximal.get(layer_name, []))
    return {
        layer_name: _unique_map_objects(objects)
        for layer_name, objects in merged.items()
    }


def _polygonal_parts(geometry: object) -> list[object]:
    if geometry is None or getattr(geometry, "is_empty", False):
        return []
    if not getattr(geometry, "is_valid", True):
        if make_valid is None:
            return []
        geometry = make_valid(geometry)

    geom_type = getattr(geometry, "geom_type", "")
    if geom_type == "Polygon":
        return [geometry]
    if geom_type in {"MultiPolygon", "GeometryCollection"}:
        polygons: list[object] = []
        for part in geometry.geoms:
            polygons.extend(_polygonal_parts(part))
        return polygons
    return []


def _merged_drivable_area_boundaries(
    geometries: Iterable[object],
) -> list[list[tuple[float, float]]]:
    polygons: list[object] = []
    for geometry in geometries:
        polygons.extend(_polygonal_parts(geometry))
    if not polygons or unary_union is None:
        return []

    merged_area = unary_union(polygons)
    boundaries = [
        _remove_immediate_backtracks(coords)
        for coords in _geometry_line_coords(merged_area.boundary)
    ]
    return [coords for coords in boundaries if len(coords) >= 2]


def _drivable_area_boundaries(
    *,
    map_api: object,
    query_frames: list[FrameRow],
    radius_m: float,
) -> list[list[tuple[float, float]]]:
    if (
        Point is None
        or SemanticMapLayer is None
        or unary_union is None
        or not query_frames
    ):
        return []
    get_vector_map_layer = getattr(map_api, "_get_vector_map_layer", None)
    if get_vector_map_layer is None:
        return []

    query_region = unary_union(
        [Point(frame.x, frame.y).buffer(radius_m) for frame in query_frames]
    )
    geometries: list[object] = []
    for layer in (
        SemanticMapLayer.DRIVABLE_AREA,
        SemanticMapLayer.CARPARK_AREA,
    ):
        try:
            layer_df = get_vector_map_layer(layer)
        except (KeyError, ValueError):
            continue
        nearby = layer_df[layer_df["geometry"].intersects(query_region)]
        geometries.extend(nearby["geometry"].tolist())
    return _merged_drivable_area_boundaries(geometries)


def _fill_lane_topology(
    *,
    edges: list[object],
    group_by_edge_id: dict[str, int],
    next_groups: np.ndarray,
    left_groups: np.ndarray,
    right_groups: np.ndarray,
) -> None:
    for edge in edges:
        gid = group_by_edge_id.get(str(getattr(edge, "id", "")))
        if gid is None or gid >= next_groups.shape[0]:
            continue
        successors = []
        for outgoing in getattr(edge, "outgoing_edges", []) or []:
            out_gid = group_by_edge_id.get(str(getattr(outgoing, "id", "")))
            if out_gid is not None and out_gid != gid:
                successors.append(out_gid)
        for idx, out_gid in enumerate(sorted(set(successors))[: next_groups.shape[1]]):
            next_groups[gid, idx] = np.uint32(out_gid)

        adjacent_edges = getattr(edge, "adjacent_edges", None)
        if adjacent_edges is None:
            continue
        left_edge, right_edge = adjacent_edges
        left_gid = (
            group_by_edge_id.get(str(getattr(left_edge, "id", "")))
            if left_edge is not None
            else None
        )
        right_gid = (
            group_by_edge_id.get(str(getattr(right_edge, "id", "")))
            if right_edge is not None
            else None
        )
        if left_gid is not None:
            left_groups[gid, 0] = np.uint32(left_gid)
        if right_gid is not None:
            right_groups[gid, 0] = np.uint32(right_gid)


def _fill_traffic_light_tensors(
    *,
    map_arrays: dict[str, np.ndarray],
    center_slots_by_id: dict[int, list[int]],
    boundary_slots_by_id: dict[int, list[int]],
    sampled_frames: list[FrameRow],
    tl_by_token: dict[bytes, list[dict[str, object]]],
) -> int:
    filled = 0
    for frame_idx, frame in enumerate(sampled_frames):
        for tl in tl_by_token.get(frame.token, []):
            connector_id = int(tl["lane_connector_id"])
            status = str(tl["status"]).lower()
            status_idx = TL_STATUS_TO_INDEX.get(status, TL_STATUS_TO_INDEX["unknown"])
            for prefix, slots_by_id in (
                ("lanes_centers", center_slots_by_id),
                ("other_lanes", boundary_slots_by_id),
            ):
                slots = slots_by_id.get(connector_id)
                if not slots:
                    continue
                tl_states = map_arrays[f"{prefix}_tl_states"]
                tl_masks = map_arrays[f"{prefix}_tl_masks"]
                for slot in slots:
                    tl_states[slot, frame_idx, :] = 0.0
                    tl_states[slot, frame_idx, status_idx] = 1.0
                    tl_masks[slot, frame_idx] = True
                    filled += 1
    return filled


def _route_roadblock_for_id(map_api: object, roadblock_id: str) -> object | None:
    roadblock = map_api.get_map_object(roadblock_id, SemanticMapLayer.ROADBLOCK)
    if roadblock is None:
        roadblock = map_api.get_map_object(
            roadblock_id, SemanticMapLayer.ROADBLOCK_CONNECTOR
        )
    return roadblock


def _route_edge_heading(edge: object) -> float:
    coords = _edge_centerline_coords(edge)
    if len(coords) < 2:
        return 0.0
    return math.atan2(coords[-1][1] - coords[0][1], coords[-1][0] - coords[0][0])


def _route_edge_anchor_score(edge: object, anchor: FrameRow) -> float:
    coords = _edge_centerline_coords(edge)
    if not coords:
        return float("inf")
    distance = min(math.hypot(x - anchor.x, y - anchor.y) for x, y in coords)
    heading_error = abs(float(_wrap_angle(_route_edge_heading(edge) - anchor.yaw)))
    return distance + 5.0 * heading_error


def _route_edge_continuity_score(edge: object, prev_edge: object) -> float:
    coords = _edge_centerline_coords(edge)
    prev_coords = _edge_centerline_coords(prev_edge)
    if not coords or not prev_coords:
        return float("inf")
    gap = math.hypot(
        coords[0][0] - prev_coords[-1][0], coords[0][1] - prev_coords[-1][1]
    )
    heading_error = abs(
        float(_wrap_angle(_route_edge_heading(edge) - _route_edge_heading(prev_edge)))
    )
    return gap + 5.0 * heading_error


def _edge_ids(edges: Iterable[object]) -> set[str]:
    return {str(getattr(edge, "id", "")) for edge in edges}


def _choose_route_edge(
    edges: list[object], anchor: FrameRow, prev_edge: object | None
) -> object | None:
    candidates = [edge for edge in edges if _edge_centerline_coords(edge)]
    if not candidates:
        return None
    if prev_edge is None:
        return min(candidates, key=lambda edge: _route_edge_anchor_score(edge, anchor))

    outgoing_ids = _edge_ids(getattr(prev_edge, "outgoing_edges", []) or [])
    linked = [
        edge for edge in candidates if str(getattr(edge, "id", "")) in outgoing_ids
    ]
    if not linked:
        prev_id = str(getattr(prev_edge, "id", ""))
        linked = [
            edge
            for edge in candidates
            if prev_id in _edge_ids(getattr(edge, "incoming_edges", []) or [])
        ]
    if linked:
        candidates = linked
    return min(
        candidates, key=lambda edge: _route_edge_continuity_score(edge, prev_edge)
    )


def _stitch_route_edge_coords(edges: list[object]) -> list[tuple[float, float]]:
    coords: list[tuple[float, float]] = []
    for edge in edges:
        edge_coords = _edge_centerline_coords(edge)
        if not edge_coords:
            continue
        if (
            coords
            and math.hypot(
                coords[-1][0] - edge_coords[0][0], coords[-1][1] - edge_coords[0][1]
            )
            < 1.0
        ):
            coords.extend(edge_coords[1:])
        else:
            coords.extend(edge_coords)
    return coords


def _route_start_index(roadblocks: list[object], anchor: FrameRow) -> int:
    scores = [
        min(
            (
                _route_edge_anchor_score(edge, anchor)
                for edge in getattr(roadblock, "interior_edges", []) or []
            ),
            default=float("inf"),
        )
        for roadblock in roadblocks
    ]
    if not scores:
        return 0
    return int(np.argmin(scores))


def _sample_route_points_from_map(
    *,
    map_api: object | None,
    route_roadblock_ids: list[str],
    anchor: FrameRow,
) -> np.ndarray | None:
    if map_api is None or SemanticMapLayer is None or not route_roadblock_ids:
        return None

    roadblocks = [
        roadblock
        for roadblock_id in route_roadblock_ids
        if (roadblock := _route_roadblock_for_id(map_api, roadblock_id)) is not None
    ]
    if not roadblocks:
        return None

    route_edges: list[object] = []
    prev_edge: object | None = None
    for roadblock in roadblocks[_route_start_index(roadblocks, anchor) :]:
        edge = _choose_route_edge(
            list(getattr(roadblock, "interior_edges", []) or []), anchor, prev_edge
        )
        if edge is None:
            continue
        route_edges.append(edge)
        prev_edge = edge

    coords = _stitch_route_edge_coords(route_edges)
    if not coords:
        return None
    local = np.asarray(
        [_global_to_anchor_xy(x, y, anchor) for x, y in coords], dtype=np.float32
    )
    return _fit_route_polyline(local, ROUTE_POINTS)


def _build_map_arrays(
    *,
    map_api: object | None,
    anchor: FrameRow,
    total_frames: int,
    sampled_frames: list[FrameRow],
    map_query_frames: list[FrameRow],
    tl_by_token: dict[bytes, list[dict[str, object]]],
    route_roadblock_ids: list[str],
    radius_m: float,
    max_center_segments: int,
    max_boundary_segments: int,
    include_traffic_lights: bool = True,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    arrays = _empty_map_arrays(
        total_frames,
        max_lanes_centers=max_center_segments,
        max_lanes_other=max_boundary_segments,
        include_traffic_lights=include_traffic_lights,
    )
    stats: dict[str, object] = {
        "enabled": map_api is not None,
        "radius_m": radius_m,
        "max_center_segments": max_center_segments,
        "max_boundary_segments": max_boundary_segments,
        "center_segments": 0,
        "other_segments": 0,
        "traffic_light_assignments": 0,
    }
    if map_api is None or Point2D is None or SemanticMapLayer is None:
        return arrays, stats

    query_frames = map_query_frames
    if not query_frames:
        query_frames = _query_frames_along_path(
            sampled_frames, num_points=DEFAULT_MAP_QUERY_POINTS
        )
    distance_frames = query_frames if query_frames else [anchor]
    layer_names = [
        SemanticMapLayer.LANE,
        SemanticMapLayer.LANE_CONNECTOR,
        SemanticMapLayer.STOP_LINE,
        SemanticMapLayer.CROSSWALK,
    ]
    proximal = _merge_proximal_map_objects(
        map_api=map_api,
        query_frames=query_frames,
        radius_m=radius_m,
        layer_names=layer_names,
    )
    lanes = sorted(
        _unique_map_objects(proximal.get(SemanticMapLayer.LANE, [])),
        key=lambda edge: _edge_distance_to_frames(edge, query_frames),
    )
    connectors = sorted(
        _unique_map_objects(proximal.get(SemanticMapLayer.LANE_CONNECTOR, [])),
        key=lambda edge: _edge_distance_to_frames(edge, query_frames),
    )
    center_edges = sorted(
        list(lanes) + list(connectors),
        key=lambda edge: _edge_distance_to_frames(edge, query_frames),
    )
    center_segments_before_truncation = 0
    center_candidates: list[MapSegmentCandidate] = []
    center_candidate_order = 0
    group_by_edge_id: dict[str, int] = {}
    for group_id, edge in enumerate(center_edges):
        lane_group_id = group_id
        edge_id = str(getattr(edge, "id", group_id))
        group_by_edge_id[edge_id] = lane_group_id
        speed_mps = getattr(edge, "speed_limit_mps", None)
        if speed_mps is None:
            speed_mps = 0.0
        coords = _downsample_coords(_edge_centerline_coords(edge))
        connector_key = int(_stable_int64_id(edge_id)) if edge in connectors else None
        candidates, center_candidate_order = _make_segment_candidates(
            coords=coords,
            distance_frames=distance_frames,
            group_id=lane_group_id,
            raw_id=edge_id,
            lane_type=LANE_TYPE_LANE_CENTER_LINE,
            speed_mps=float(speed_mps),
            order_start=center_candidate_order,
            connector_key=connector_key,
        )
        center_segments_before_truncation += len(candidates)
        center_candidates.extend(candidates)

    selected_center_candidates = _write_segment_candidates(
        candidates=center_candidates,
        anchor=anchor,
        points=arrays["lanes_centers_points"],
        masks=arrays["lanes_centers_masks"],
        attributes=arrays["lanes_centers_attributes"],
        groups=arrays["lanes_centers_groups"],
        ids=arrays["lanes_centers_ids"],
        max_segments=max_center_segments,
    )
    connector_slots_by_id: dict[int, list[int]] = {}
    for slot, candidate in enumerate(selected_center_candidates):
        if candidate.connector_key is not None:
            connector_slots_by_id.setdefault(candidate.connector_key, []).append(slot)

    _fill_lane_topology(
        edges=center_edges,
        group_by_edge_id=group_by_edge_id,
        next_groups=arrays["lanes_centers_next_groups"],
        left_groups=arrays["lanes_centers_left_groups"],
        right_groups=arrays["lanes_centers_right_groups"],
    )

    other_segments_before_truncation = 0
    other_candidates: list[MapSegmentCandidate] = []
    other_candidate_order = 0
    other_group = 0
    emitted_boundary_ids: set[tuple[str, str]] = set()

    def append_other_polyline(
        *,
        coords: list[tuple[float, float]],
        raw_id: object,
        lane_type: float,
        speed_mps: float = -1.0,
        max_points: int = MAX_MAP_POINTS_PER_OBJECT,
        max_slot: int | None = None,
        connector_key: int | None = None,
    ) -> None:
        nonlocal other_group, other_segments_before_truncation, other_candidate_order
        sampled_coords = _downsample_coords(coords, max_points=max_points)
        candidates, other_candidate_order = _make_segment_candidates(
            coords=sampled_coords,
            distance_frames=distance_frames,
            group_id=other_group,
            raw_id=raw_id,
            lane_type=lane_type,
            speed_mps=speed_mps,
            order_start=other_candidate_order,
            connector_key=connector_key,
        )
        other_segments_before_truncation += len(candidates)
        if candidates:
            other_candidates.extend(candidates)
            other_group += 1

    curb_lines = _drivable_area_boundaries(
        map_api=map_api,
        query_frames=query_frames,
        radius_m=radius_m,
    )
    for curb_idx, curb_coords in enumerate(curb_lines):
        append_other_polyline(
            coords=curb_coords,
            raw_id=f"curb:{curb_idx}",
            lane_type=LANE_TYPE_CURB,
            max_points=MAX_MAP_POLYGON_POINTS_PER_OBJECT,
        )

    stop_lines = sorted(
        [
            stop_line
            for stop_line in _unique_map_objects(
                proximal.get(SemanticMapLayer.STOP_LINE, [])
            )
            if StopLineType is None
            or getattr(stop_line, "stop_line_type", None) != StopLineType.TURN_STOP
        ],
        key=lambda poly: _polygon_distance_to_frames(poly, query_frames),
    )
    for stop_line in stop_lines:
        append_other_polyline(
            coords=_polygon_exterior_coords(stop_line),
            raw_id=f"stop:{getattr(stop_line, 'id', other_group)}",
            lane_type=LANE_TYPE_STOP_LINE,
            max_points=MAX_MAP_POLYGON_POINTS_PER_OBJECT,
        )

    crosswalk_objects = sorted(
        _unique_map_objects(proximal.get(SemanticMapLayer.CROSSWALK, [])),
        key=lambda poly: _polygon_distance_to_frames(poly, query_frames),
    )
    for crosswalk in crosswalk_objects:
        append_other_polyline(
            coords=_polygon_exterior_coords(crosswalk),
            raw_id=f"crosswalk:{getattr(crosswalk, 'id', other_group)}",
            lane_type=LANE_TYPE_CROSSWALK,
            max_points=MAX_MAP_POLYGON_POINTS_PER_OBJECT,
        )

    for edge in center_edges:
        edge_id = str(getattr(edge, "id", ""))
        connector_key = int(_stable_int64_id(edge_id)) if edge in connectors else None
        for side in ("left_boundary", "right_boundary"):
            try:
                boundary = getattr(edge, side)
            except Exception:
                continue
            boundary_id = str(
                getattr(boundary, "id", f"{getattr(edge, 'id', other_group)}:{side}")
            )
            boundary_key = ("road_boundary", boundary_id)
            if boundary_key in emitted_boundary_ids:
                continue
            emitted_boundary_ids.add(boundary_key)
            append_other_polyline(
                coords=_line_coords(boundary),
                raw_id=boundary_id,
                lane_type=LANE_TYPE_ROAD_BOUNDARY,
                connector_key=connector_key,
            )

    selected_other_candidates = _write_segment_candidates(
        candidates=other_candidates,
        anchor=anchor,
        points=arrays["other_lanes_points"],
        masks=arrays["other_lanes_masks"],
        attributes=arrays["other_lanes_attributes"],
        groups=arrays["other_lanes_groups"],
        ids=arrays["other_lanes_ids"],
        max_segments=max_boundary_segments,
    )
    boundary_slots_by_id: dict[int, list[int]] = {}
    for slot, candidate in enumerate(selected_other_candidates):
        if candidate.connector_key is not None:
            boundary_slots_by_id.setdefault(candidate.connector_key, []).append(slot)
    boundary_type_counts: dict[int, int] = {}
    for candidate in selected_other_candidates:
        lane_type_key = int(candidate.lane_type)
        boundary_type_counts[lane_type_key] = (
            boundary_type_counts.get(lane_type_key, 0) + 1
        )

    stats["center_segments"] = int(arrays["lanes_centers_masks"].sum())
    stats["other_segments"] = int(arrays["other_lanes_masks"].sum())
    stats["center_segments_before_truncation"] = center_segments_before_truncation
    stats["other_segments_before_truncation"] = other_segments_before_truncation
    stats["center_segments_truncated"] = max(
        0, center_segments_before_truncation - int(stats["center_segments"])
    )
    stats["other_segments_truncated"] = max(
        0, other_segments_before_truncation - int(stats["other_segments"])
    )
    stats["query_points"] = [(frame.x, frame.y) for frame in query_frames]
    stats["query_point_count"] = len(query_frames)
    stats["query_timestamps_us"] = [frame.timestamp_us for frame in query_frames]
    stats["query_relative_times_s"] = [
        (frame.timestamp_us - anchor.timestamp_us) / 1e6 for frame in query_frames
    ]
    stats["lane_objects"] = len(lanes)
    stats["lane_connector_objects"] = len(connectors)
    stats["curb_polylines"] = len(curb_lines)
    stats["curb_segments"] = boundary_type_counts.get(int(LANE_TYPE_CURB), 0)
    stats["road_boundary_segments"] = boundary_type_counts.get(
        int(LANE_TYPE_ROAD_BOUNDARY), 0
    )
    stats["stop_line_objects"] = len(stop_lines)
    stats["crosswalk_objects"] = len(crosswalk_objects)
    stats["boundary_type_counts"] = boundary_type_counts
    stats["polygon_note"] = (
        "nuPlan VectorSetMap-style features queried around uniformly sampled future ego-path points: "
        "lane, drivable-area polygon curbs, road boundaries, stop lines, crosswalks, and route lanes."
    )
    if include_traffic_lights:
        stats["traffic_light_assignments"] = _fill_traffic_light_tensors(
            map_arrays=arrays,
            center_slots_by_id=connector_slots_by_id,
            boundary_slots_by_id=boundary_slots_by_id,
            sampled_frames=sampled_frames,
            tl_by_token=tl_by_token,
        )
    return arrays, stats
