#!/usr/bin/env python3
"""Export nuPlan DB samples to a DriveRL WebDataset shard.

This converter extracts ego, tracked objects, traffic-light status, route/goal
metadata, nearby map vectors, and writes DriveRL compatible ``.npz`` samples
inside tar shards.

Example:
    python3 nuplan/planning/script/export_driverl_webdataset.py \
        --input /path/to/nuplan-v1.1/splits/trainval \
        --output output/standard_train_base \
        --sample-rate-hz 5 \
        --history-frames 21 \
        --future-frames 100 \
        --max-samples 1
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import sqlite3
import sys
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

try:
    from nuplan.common.actor_state.state_representation import Point2D
    from nuplan.common.maps.abstract_map_objects import LaneGraphEdgeMapObject
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer, StopLineType
    from nuplan.common.maps.nuplan_map.map_factory import get_maps_api
    from nuplan.database.nuplan_db.nuplan_scenario_queries import (
        get_ego_state_for_lidarpc_token_from_db,
        get_images_from_lidar_tokens,
    )
except ImportError:  # pragma: no cover - keeps --help usable outside nuPlan envs.
    Point2D = None  # type: ignore[assignment]
    LaneGraphEdgeMapObject = object  # type: ignore[assignment,misc]
    SemanticMapLayer = None  # type: ignore[assignment]
    StopLineType = None  # type: ignore[assignment]
    get_maps_api = None  # type: ignore[assignment]
    get_ego_state_for_lidarpc_token_from_db = None  # type: ignore[assignment]
    get_images_from_lidar_tokens = None  # type: ignore[assignment]

try:
    from shapely import make_valid
    from shapely.geometry import Point
    from shapely.ops import unary_union
except ImportError:  # pragma: no cover - keeps --help usable without map dependencies.
    make_valid = None  # type: ignore[assignment]
    Point = None  # type: ignore[assignment]
    unary_union = None  # type: ignore[assignment]

from nuplan.planning.script.driverl_runtime_map_features import (
    DEFAULT_MAP_QUERY_POINTS,
    DEFAULT_MAP_RADIUS_M,
    FrameRow,
    ROUTE_POINTS,
    _build_map_arrays,
    _dedupe_route_points,
    _fit_route_polyline,
    _rotate_to_anchor,
    _sample_route_points_from_map,
    _wrap_angle,
)

NUM_OBJECT_FEATURES = 13

TYPE_FEATURE_INDEX = 0
LENGTH_FEATURE_INDEX = 1
WIDTH_FEATURE_INDEX = 2
HEADING_FEATURE_INDEX = 3
X_FEATURE_INDEX = 4
Y_FEATURE_INDEX = 5
VELOCITY_X_FEATURE_INDEX = 6
VELOCITY_Y_FEATURE_INDEX = 7
ACCELERATION_X_FEATURE_INDEX = 8
ACCELERATION_Y_FEATURE_INDEX = 9
TIMESTAMP_FEATURE_INDEX = 10
YAW_RATE_FEATURE_INDEX = 11
STEERING_ANGLE_FEATURE_INDEX = 12
STEER_RATIO = 12.6

MAX_AGENTS = 128
DEFAULT_MAX_LANES_CENTERS = 1024
DEFAULT_MAX_LANES_OTHER = 1024
OCC_GRID_SHAPE = (1300, 300)
DEFAULT_MAX_CENTER_SEGMENTS = DEFAULT_MAX_LANES_CENTERS
DEFAULT_MAX_BOUNDARY_SEGMENTS = DEFAULT_MAX_LANES_OTHER
DEFAULT_MAP_QUERY_DURATION_S = 32.0
ROUTE_FORWARD_MAP_QUERY_POINTS = 5
ROUTE_FORWARD_MAP_QUERY_DISTANCE_M = 100.0
# Keep defaults repository-local and portable.  Callers can override the map
# root with ``--map-root`` or ``NUPLAN_MAPS_ROOT``; the exporter will also
# discover a sibling ``maps`` directory next to the input dataset.
DEFAULT_MAP_ROOT = Path("data/nuplan/maps")
DEFAULT_MAP_VERSION = "nuplan-maps-v1.0"
DEFAULT_OUTPUT_DIR = Path("output/standard_train_base")
CAMERA_CHANNELS = (
    "CAM_F0",
    "CAM_R0",
    "CAM_R1",
    "CAM_R2",
    "CAM_B0",
    "CAM_L0",
    "CAM_L1",
    "CAM_L2",
)



NUPLAN_SIMULATION_TEST_SCENARIO_TYPES = (
    "accelerating_at_crosswalk",
    "accelerating_at_stop_sign",
    "accelerating_at_stop_sign_no_crosswalk",
    "accelerating_at_traffic_light",
    "accelerating_at_traffic_light_with_lead",
    "accelerating_at_traffic_light_without_lead",
    "behind_bike",
    "behind_long_vehicle",
    "behind_pedestrian_on_driveable",
    "behind_pedestrian_on_pickup_dropoff",
    "changing_lane",
    "changing_lane_to_left",
    "changing_lane_to_right",
    "changing_lane_with_lead",
    "changing_lane_with_trail",
    "crossed_by_bike",
    "crossed_by_vehicle",
    "following_lane_with_lead",
    "following_lane_with_slow_lead",
    "following_lane_without_lead",
    "high_lateral_acceleration",
    "high_magnitude_jerk",
    "high_magnitude_speed",
    "low_magnitude_speed",
    "medium_magnitude_speed",
    "near_barrier_on_driveable",
    "near_construction_zone_sign",
    "near_high_speed_vehicle",
    "near_long_vehicle",
    "near_multiple_bikes",
    "near_multiple_pedestrians",
    "near_multiple_vehicles",
    "near_pedestrian_at_pickup_dropoff",
    "near_pedestrian_on_crosswalk",
    "near_pedestrian_on_crosswalk_with_ego",
    "near_trafficcone_on_driveable",
    "on_all_way_stop_intersection",
    "on_carpark",
    "on_intersection",
    "on_pickup_dropoff",
    "on_stopline_crosswalk",
    "on_stopline_stop_sign",
    "on_stopline_traffic_light",
    "on_traffic_light_intersection",
    "starting_high_speed_turn",
    "starting_left_turn",
    "starting_low_speed_turn",
    "starting_protected_cross_turn",
    "starting_protected_noncross_turn",
    "starting_right_turn",
    "starting_straight_stop_sign_intersection_traversal",
    "starting_straight_traffic_light_intersection_traversal",
    "starting_u_turn",
    "starting_unprotected_cross_turn",
    "starting_unprotected_noncross_turn",
    "stationary",
    "stationary_at_crosswalk",
    "stationary_at_traffic_light_with_lead",
    "stationary_at_traffic_light_without_lead",
    "stationary_in_traffic",
    "stopping_at_crosswalk",
    "stopping_at_stop_sign_no_crosswalk",
    "stopping_at_stop_sign_with_lead",
    "stopping_at_stop_sign_without_lead",
    "stopping_at_traffic_light_with_lead",
    "stopping_at_traffic_light_without_lead",
    "stopping_with_lead",
    "traversing_crosswalk",
    "traversing_intersection",
    "traversing_narrow_lane",
    "traversing_pickup_dropoff",
    "traversing_traffic_light_intersection",
    "waiting_for_pedestrian_to_cross",
)


@dataclass(frozen=True)
class BoxRow:
    track_token_hex: str
    category: str
    x: float
    y: float
    yaw: float
    vx: float
    vy: float
    length: float
    width: float


@dataclass(frozen=True)
class AnchorRecord:
    db_path: Path
    token_hex: str
    scenario_type: str
    timestamp_us: int
    exact: bool = False


@dataclass(frozen=True)
class MapContext:
    map_root: Path | None
    map_version: str | None
    radius_m: float
    query_duration_s: float
    query_points: int
    max_center_segments: int
    max_boundary_segments: int


def _yaw_from_quaternion(qw: float, qx: float, qy: float, qz: float) -> float:
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)




def _rotate_local_vector_to_anchor(
    vx: float,
    vy: float,
    frame_yaw: float,
    anchor_yaw: float,
) -> tuple[float, float]:
    """Rotate a vector expressed in a frame-local coordinate into the anchor frame."""
    delta_yaw = frame_yaw - anchor_yaw
    c = math.cos(delta_yaw)
    s = math.sin(delta_yaw)
    return c * vx - s * vy, s * vx + c * vy




def _anchor_to_global_xy(x: float, y: float, anchor: FrameRow) -> tuple[float, float]:
    c = math.cos(anchor.yaw)
    s = math.sin(anchor.yaw)
    return anchor.x + c * x - s * y, anchor.y + s * x + c * y




def _object_type_id(category_name: str) -> float:
    name = category_name.lower()
    if "pedestrian" in name:
        return 4.0
    if "bicycle" in name or "motorcycle" in name:
        return 5.0
    if _is_vehicle_category(name):
        return 6.0
    return 1.0


def _is_vehicle_category(category_name: str) -> bool:
    name = category_name.lower()
    return any(key in name for key in ("vehicle", "car", "bus", "truck", "trailer"))


def _select_agent_tracks(
    candidates: dict[str, tuple[int, float, str]], min_vehicle_agents: int
) -> list[str]:
    ranked = sorted(candidates.items(), key=lambda item: item[1][:2])
    selected = [
        track for track, (_, _, category) in ranked if _is_vehicle_category(category)
    ][:min_vehicle_agents]
    selected_set = set(selected)
    selected.extend(
        track
        for track, _ in ranked
        if track not in selected_set
    )
    return selected[: MAX_AGENTS - 1]


def _iter_db_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    return sorted(input_path.glob("*.db"))


def _read_anchor_tokens(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    tokens: set[str] = set()
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        # Accept either "token" or "log_name,token".
        tokens.add(stripped.split(",")[-1].strip())
    return tokens


def _anchors_from_token_file(path: Path | None, db_files: list[Path]) -> list[AnchorRecord] | None:
    if path is None:
        return None

    db_by_stem = {db_path.stem: db_path for db_path in db_files}
    anchors: list[AnchorRecord] = []
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = [part.strip() for part in stripped.split(",", 1)]
        if len(parts) != 2:
            return None
        db_path = db_by_stem.get(parts[0])
        if db_path is None:
            raise FileNotFoundError(f"Token file references DB not found in input: {parts[0]}")
        anchors.append(
            AnchorRecord(
                db_path=db_path,
                token_hex=parts[1],
                scenario_type="",
                timestamp_us=0,
                exact=True,
            )
        )
    return anchors


def _load_frames(conn: sqlite3.Connection) -> list[FrameRow]:
    query = """
        SELECT
            lp.token AS token,
            lp.timestamp AS timestamp,
            lp.scene_token AS scene_token,
            ep.x AS x,
            ep.y AS y,
            ep.qw AS qw,
            ep.qx AS qx,
            ep.qy AS qy,
            ep.qz AS qz,
            ep.vx AS vx,
            ep.vy AS vy,
            ep.acceleration_x AS ax,
            ep.acceleration_y AS ay,
            ep.angular_rate_z AS yaw_rate
        FROM lidar_pc AS lp
        INNER JOIN ego_pose AS ep
            ON ep.token = lp.ego_pose_token
        ORDER BY lp.timestamp ASC
    """
    rows = []
    for row in conn.execute(query):
        token = bytes(row["token"])
        rows.append(
            FrameRow(
                token=token,
                token_hex=token.hex(),
                timestamp_us=int(row["timestamp"]),
                scene_token=bytes(row["scene_token"]) if row["scene_token"] is not None else None,
                x=float(row["x"]),
                y=float(row["y"]),
                yaw=_yaw_from_quaternion(
                    float(row["qw"]),
                    float(row["qx"]),
                    float(row["qy"]),
                    float(row["qz"]),
                ),
                vx=float(row["vx"] or 0.0),
                vy=float(row["vy"] or 0.0),
                ax=float(row["ax"] or 0.0),
                ay=float(row["ay"] or 0.0),
                yaw_rate=float(row["yaw_rate"] or 0.0),
            )
        )
    return rows


def _camera_aligned_start_idx(db_path: Path, conn: sqlite3.Connection) -> int:
    if get_images_from_lidar_tokens is None:
        raise RuntimeError("nuPlan camera queries are required for camera-aligned frame selection")

    cursor = iter(conn.execute("SELECT token, timestamp FROM lidar_pc ORDER BY timestamp ASC"))
    for start_idx, row in enumerate(cursor):
        token_hex = bytes(row["token"]).hex()
        images = list(get_images_from_lidar_tokens(str(db_path), [token_hex], list(CAMERA_CHANNELS)))
        if not images:
            continue

        next_row = next(cursor, None)
        if next_row is None:
            return start_idx

        def camera_gap_us(candidate: sqlite3.Row, candidate_images: Iterable[object]) -> float:
            candidate_timestamp = int(candidate["timestamp"])
            return min(
                (
                    abs(int(image.timestamp) - candidate_timestamp)
                    for image in candidate_images
                    if image.timestamp is not None
                ),
                default=math.inf,
            )

        next_images = get_images_from_lidar_tokens(
            str(db_path),
            [bytes(next_row["token"]).hex()],
            list(CAMERA_CHANNELS),
        )
        return start_idx if camera_gap_us(row, images) < camera_gap_us(next_row, next_images) else start_idx + 1

    raise RuntimeError(f"No camera data found for LiDAR frames in {db_path}")


def _valid_scene_tokens(conn: sqlite3.Connection) -> set[bytes]:
    query = """
        WITH ordered_scenes AS
        (
            SELECT token, ROW_NUMBER() OVER (ORDER BY name ASC) AS row_num
            FROM scene
        ),
        num_scenes AS
        (
            SELECT COUNT(*) AS cnt
            FROM scene
        )
        SELECT o.token
        FROM ordered_scenes AS o
        CROSS JOIN num_scenes AS n
        WHERE o.row_num >= 3 AND o.row_num < n.cnt - 1
    """
    return {bytes(row["token"]) for row in conn.execute(query)}


def _scene_metadata(conn: sqlite3.Connection, scene_token: bytes | None) -> dict[str, object]:
    if scene_token is None:
        return {}
    row = conn.execute(
        """
        SELECT
            s.name AS scene_name,
            s.roadblock_ids AS roadblock_ids,
            gep.x AS goal_x,
            gep.y AS goal_y,
            gep.qw AS goal_qw,
            gep.qx AS goal_qx,
            gep.qy AS goal_qy,
            gep.qz AS goal_qz
        FROM scene AS s
        LEFT JOIN ego_pose AS gep
            ON gep.token = s.goal_ego_pose_token
        WHERE s.token = ?
        """,
        (scene_token,),
    ).fetchone()
    if row is None:
        return {}
    goal = None
    if row["goal_x"] is not None:
        goal = {
            "x": float(row["goal_x"]),
            "y": float(row["goal_y"]),
            "yaw": _yaw_from_quaternion(
                float(row["goal_qw"]),
                float(row["goal_qx"]),
                float(row["goal_qy"]),
                float(row["goal_qz"]),
            ),
        }
    return {
        "scene_name": row["scene_name"],
        "route_roadblock_ids": _parse_roadblock_ids(row["roadblock_ids"]),
        "mission_goal": goal,
    }


def _scenario_tags(conn: sqlite3.Connection, token: bytes) -> list[str]:
    return [
        str(row["type"])
        for row in conn.execute(
            "SELECT type FROM scenario_tag WHERE lidar_pc_token = ? ORDER BY type ASC",
            (token,),
        )
        if row["type"] is not None
    ]


def _log_metadata(conn: sqlite3.Connection) -> dict[str, object]:
    row = conn.execute("SELECT logfile, map_version FROM log LIMIT 1").fetchone()
    if row is None:
        return {}
    return {"log_name": row["logfile"], "map_name": row["map_version"]}


def _parse_roadblock_ids(raw: str | None) -> list[str]:
    if not raw:
        return []
    text = raw.strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
    except json.JSONDecodeError:
        pass
    return [part for part in text.replace(",", " ").split() if part]


def _infer_map_root(input_path: Path) -> Path | None:
    env_root = os.environ.get("NUPLAN_MAPS_ROOT")
    if env_root:
        return Path(env_root)
    candidates = [
        input_path.parent / "maps",
        input_path.parent.parent / "maps",
        input_path.parent.parent.parent / "maps",
        DEFAULT_MAP_ROOT,
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return None


def _infer_map_version(map_root: Path | None) -> str | None:
    env_version = os.environ.get("NUPLAN_MAP_VERSION")
    if env_version:
        return env_version
    if map_root is None:
        return None
    json_files = sorted(map_root.glob("nuplan-maps-v*.json"))
    if json_files:
        return json_files[-1].stem
    if (map_root / f"{DEFAULT_MAP_VERSION}.json").is_file():
        return DEFAULT_MAP_VERSION
    return None


def _load_boxes_for_tokens(conn: sqlite3.Connection, tokens: list[bytes]) -> dict[bytes, list[BoxRow]]:
    if not tokens:
        return {}
    placeholders = ",".join("?" for _ in tokens)
    query = f"""
        SELECT
            lb.lidar_pc_token AS lidar_pc_token,
            lb.track_token AS track_token,
            lb.x AS x,
            lb.y AS y,
            lb.yaw AS yaw,
            lb.vx AS vx,
            lb.vy AS vy,
            lb.length AS length,
            lb.width AS width,
            c.name AS category
        FROM lidar_box AS lb
        INNER JOIN track AS tr
            ON tr.token = lb.track_token
        INNER JOIN category AS c
            ON c.token = tr.category_token
        WHERE lb.lidar_pc_token IN ({placeholders})
    """
    output: dict[bytes, list[BoxRow]] = {token: [] for token in tokens}
    for row in conn.execute(query, tokens):
        lidar_token = bytes(row["lidar_pc_token"])
        output.setdefault(lidar_token, []).append(
            BoxRow(
                track_token_hex=bytes(row["track_token"]).hex(),
                category=str(row["category"] or ""),
                x=float(row["x"]),
                y=float(row["y"]),
                yaw=float(row["yaw"]),
                vx=float(row["vx"] or 0.0),
                vy=float(row["vy"] or 0.0),
                length=float(row["length"] or 0.0),
                width=float(row["width"] or 0.0),
            )
        )
    return output


def _load_traffic_lights(conn: sqlite3.Connection, tokens: list[bytes]) -> dict[bytes, list[dict[str, object]]]:
    if not tokens:
        return {}
    placeholders = ",".join("?" for _ in tokens)
    query = f"""
        SELECT lidar_pc_token, lane_connector_id, status
        FROM traffic_light_status
        WHERE lidar_pc_token IN ({placeholders})
        ORDER BY lane_connector_id ASC
    """
    output: dict[bytes, list[dict[str, object]]] = {token: [] for token in tokens}
    for row in conn.execute(query, tokens):
        token = bytes(row["lidar_pc_token"])
        output.setdefault(token, []).append(
            {
                "lane_connector_id": int(row["lane_connector_id"]),
                "status": str(row["status"]).lower(),
            }
        )
    return output


def _sample_route_points(ego_states: np.ndarray) -> np.ndarray:
    if ego_states.size == 0:
        return np.zeros((ROUTE_POINTS, 2), dtype=np.float32)
    xy = ego_states[:, [X_FEATURE_INDEX, Y_FEATURE_INDEX]]
    return _fit_route_polyline(xy, ROUTE_POINTS)










def _route_forward_query_frames(
    *,
    anchor: FrameRow,
    route_points: np.ndarray,
    num_points: int = ROUTE_FORWARD_MAP_QUERY_POINTS,
) -> list[FrameRow]:
    points = _dedupe_route_points(route_points)
    if num_points <= 0 or len(points) < 2:
        return []

    deltas = np.diff(points, axis=0)
    lengths = np.linalg.norm(deltas, axis=1)
    valid = lengths > 1e-3
    if not np.any(valid):
        return []

    progress = np.concatenate(([0.0], np.cumsum(lengths)))
    best_distance_sq = float("inf")
    anchor_progress = 0.0
    for idx, valid_segment in enumerate(valid):
        if not valid_segment:
            continue
        p0 = points[idx]
        delta = deltas[idx]
        t = float(np.clip(-float(np.dot(p0, delta)) / float(lengths[idx] ** 2), 0.0, 1.0))
        closest = p0 + t * delta
        distance_sq = float(np.dot(closest, closest))
        if distance_sq < best_distance_sq:
            best_distance_sq = distance_sq
            anchor_progress = float(progress[idx] + t * lengths[idx])

    targets = anchor_progress + np.linspace(
        ROUTE_FORWARD_MAP_QUERY_DISTANCE_M / num_points,
        ROUTE_FORWARD_MAP_QUERY_DISTANCE_M,
        num_points,
        dtype=np.float32,
    )
    targets = np.clip(targets, 0.0, float(progress[-1]))
    local = np.stack(
        [
            np.interp(targets, progress, points[:, 0]),
            np.interp(targets, progress, points[:, 1]),
        ],
        axis=1,
    )

    frames: list[FrameRow] = []
    for idx, (local_x, local_y) in enumerate(local):
        x, y = _anchor_to_global_xy(float(local_x), float(local_y), anchor)
        frames.append(
            FrameRow(
                token=b"",
                token_hex=f"route_forward_query_{idx}",
                timestamp_us=anchor.timestamp_us,
                scene_token=anchor.scene_token,
                x=x,
                y=y,
                yaw=anchor.yaw,
                vx=0.0,
                vy=0.0,
                ax=0.0,
                ay=0.0,
                yaw_rate=0.0,
            )
        )
    return frames


















def _valid_segment_count(coords: list[tuple[float, float]]) -> int:
    if len(coords) < 2:
        return 0
    count = 0
    for p0, p1 in zip(coords[:-1], coords[1:]):
        if math.hypot(p1[0] - p0[0], p1[1] - p0[1]) >= 1e-3:
            count += 1
    return count












def _edge_distance_to_anchor(edge: object, anchor: FrameRow) -> float:
    if Point is None:
        return 0.0
    baseline_path = getattr(edge, "baseline_path", None)
    linestring = getattr(baseline_path, "linestring", None)
    if linestring is None:
        return 0.0
    return float(linestring.distance(Point(anchor.x, anchor.y)))




def _query_frames_by_time(
    frames: list[FrameRow],
    start_pos: int,
    duration_s: float,
    num_points: int,
) -> list[FrameRow]:
    if not frames or start_pos < 0 or start_pos >= len(frames) or num_points <= 0:
        return []
    if num_points == 1 or duration_s <= 0.0:
        return [frames[start_pos]]

    target_offsets_us = np.linspace(0.0, duration_s * 1e6, num_points)
    output: list[FrameRow] = []
    seen_positions: set[int] = set()
    search_pos = start_pos
    for offset_us in target_offsets_us:
        target_timestamp = frames[start_pos].timestamp_us + int(round(float(offset_us)))
        while search_pos + 1 < len(frames) and frames[search_pos + 1].timestamp_us <= target_timestamp:
            search_pos += 1

        nearest_pos = search_pos
        if search_pos + 1 < len(frames):
            prev_delta = abs(frames[search_pos].timestamp_us - target_timestamp)
            next_delta = abs(frames[search_pos + 1].timestamp_us - target_timestamp)
            if next_delta < prev_delta:
                nearest_pos = search_pos + 1

        if nearest_pos in seen_positions:
            continue
        seen_positions.add(nearest_pos)
        output.append(frames[nearest_pos])
    return output


def _unique_frames_by_token(frames: Iterable[FrameRow]) -> list[FrameRow]:
    output: list[FrameRow] = []
    seen_tokens: set[str] = set()
    for frame in frames:
        if frame.token_hex in seen_tokens:
            continue
        seen_tokens.add(frame.token_hex)
        output.append(frame)
    return output




def _polygon_distance_to_anchor(poly_obj: object, anchor: FrameRow) -> float:
    if Point is None:
        return 0.0
    polygon = getattr(poly_obj, "polygon", None)
    if polygon is None:
        return 0.0
    return float(polygon.distance(Point(anchor.x, anchor.y)))






































def _get_map_api(
    *,
    map_name: str | None,
    map_context: MapContext | None,
    map_cache: dict[str, object],
) -> object | None:
    if not map_name or map_context is None or map_context.map_root is None or map_context.map_version is None:
        return None
    if get_maps_api is None:
        return None
    key = f"{map_context.map_root}:{map_context.map_version}:{map_name}"
    if key not in map_cache:
        map_cache[key] = get_maps_api(str(map_context.map_root), map_context.map_version, map_name)
    return map_cache[key]


def _ego_tire_steering_by_token(db_path: Path, sampled_frames: list[FrameRow]) -> dict[str, float]:
    if get_ego_state_for_lidarpc_token_from_db is None:
        return {}
    output: dict[str, float] = {}
    for frame in sampled_frames:
        ego_state = get_ego_state_for_lidarpc_token_from_db(str(db_path), frame.token_hex)
        if ego_state is None:
            continue
        output[frame.token_hex] = float(ego_state.tire_steering_angle)
    return output


def _build_sample(
    conn: sqlite3.Connection,
    db_path: Path,
    frames: list[FrameRow],
    sampled_indices: list[int],
    anchor_pos: int,
    dt: float,
    num_goal_positions: int,
    map_context: MapContext | None,
    map_cache: dict[str, object],
    min_vehicle_agents: int = 64,
    include_traffic_lights: bool = True,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    history_frames = anchor_pos + 1
    future_frames = len(sampled_indices) - history_frames
    anchor_global_pos = sampled_indices[anchor_pos]
    anchor = frames[anchor_global_pos]
    sampled_frames = [frames[idx] for idx in sampled_indices]
    sampled_tokens = [frame.token for frame in sampled_frames]
    boxes_by_token = _load_boxes_for_tokens(conn, sampled_tokens)
    ego_tire_steering_by_token = _ego_tire_steering_by_token(db_path, sampled_frames)
    tl_by_token = _load_traffic_lights(conn, sampled_tokens) if include_traffic_lights else {}
    log_meta = _log_metadata(conn)
    scene_meta = _scene_metadata(conn, anchor.scene_token)
    route_roadblock_ids = [str(item) for item in scene_meta.get("route_roadblock_ids", [])]
    map_api = _get_map_api(
        map_name=str(log_meta.get("map_name")) if log_meta.get("map_name") is not None else None,
        map_context=map_context,
        map_cache=map_cache,
    )

    total_frames = len(sampled_indices)
    all_states = np.zeros((MAX_AGENTS, total_frames, NUM_OBJECT_FEATURES), dtype=np.float32)
    all_masks = np.zeros((MAX_AGENTS, total_frames), dtype=np.bool_)

    for t, frame in enumerate(sampled_frames):
        x, y = _rotate_to_anchor(frame.x - anchor.x, frame.y - anchor.y, anchor.yaw)
        vx, vy = _rotate_local_vector_to_anchor(frame.vx, frame.vy, frame.yaw, anchor.yaw)
        ax, ay = _rotate_local_vector_to_anchor(frame.ax, frame.ay, frame.yaw, anchor.yaw)
        all_states[0, t] = np.array(
            [
                6.0,
                5.176,
                2.297,
                _wrap_angle(frame.yaw - anchor.yaw),
                x,
                y,
                vx,
                vy,
                ax,
                ay,
                (frame.timestamp_us - anchor.timestamp_us) / 1e6,
                frame.yaw_rate,
                ego_tire_steering_by_token.get(frame.token_hex, 0.0) * STEER_RATIO,
            ],
            dtype=np.float32,
        )
        all_masks[0, t] = True

    track_candidates = {
        box.track_token_hex: (
            0,
            (box.x - anchor.x) ** 2 + (box.y - anchor.y) ** 2,
            box.category,
        )
        for box in boxes_by_token.get(anchor.token, [])
    }
    for t in range(history_frames, total_frames):
        for box in boxes_by_token.get(sampled_frames[t].token, []):
            if box.track_token_hex in track_candidates:
                continue
            track_candidates[box.track_token_hex] = (
                t,
                (box.x - anchor.x) ** 2 + (box.y - anchor.y) ** 2,
                box.category,
            )
    selected_tracks = _select_agent_tracks(track_candidates, min_vehicle_agents)
    track_to_slot = {track: idx + 1 for idx, track in enumerate(selected_tracks)}
    track_category = {
        track: category for track, (_, _, category) in track_candidates.items()
    }

    for t, frame in enumerate(sampled_frames):
        for box in boxes_by_token.get(frame.token, []):
            slot = track_to_slot.get(box.track_token_hex)
            if slot is None:
                continue
            x, y = _rotate_to_anchor(box.x - anchor.x, box.y - anchor.y, anchor.yaw)
            vx, vy = _rotate_to_anchor(box.vx, box.vy, anchor.yaw)
            all_states[slot, t] = np.array(
                [
                    _object_type_id(track_category.get(box.track_token_hex, box.category)),
                    box.length,
                    box.width,
                    _wrap_angle(box.yaw - anchor.yaw),
                    x,
                    y,
                    vx,
                    vy,
                    0.0,
                    0.0,
                    (frame.timestamp_us - anchor.timestamp_us) / 1e6,
                    0.0,
                    0.0,
                ],
                dtype=np.float32,
            )
            all_masks[slot, t] = True

    goal_points = np.zeros((total_frames, num_goal_positions * 2), dtype=np.float32)
    ego_xy = all_states[0][:, [X_FEATURE_INDEX, Y_FEATURE_INDEX]]
    for goal_idx in range(num_goal_positions):
        goal_points[:, goal_idx * 2 : goal_idx * 2 + 2] = ego_xy

    route_points = _sample_route_points_from_map(
        map_api=map_api,
        route_roadblock_ids=route_roadblock_ids,
        anchor=anchor,
    )
    route_forward_query_frames = (
        _route_forward_query_frames(
            anchor=anchor,
            route_points=route_points,
        )
        if route_points is not None
        else []
    )
    future_map_query_frames = _query_frames_by_time(
        frames=frames,
        start_pos=anchor_global_pos,
        duration_s=map_context.query_duration_s if map_context is not None else DEFAULT_MAP_QUERY_DURATION_S,
        num_points=map_context.query_points if map_context is not None else DEFAULT_MAP_QUERY_POINTS,
    )
    map_query_frames = _unique_frames_by_token(
        sampled_frames[:history_frames] + future_map_query_frames + route_forward_query_frames
    )
    map_arrays, map_stats = _build_map_arrays(
        map_api=map_api,
        anchor=anchor,
        total_frames=total_frames,
        sampled_frames=sampled_frames,
        map_query_frames=map_query_frames,
        tl_by_token=tl_by_token,
        route_roadblock_ids=route_roadblock_ids,
        radius_m=map_context.radius_m if map_context is not None else DEFAULT_MAP_RADIUS_M,
        max_center_segments=map_context.max_center_segments if map_context is not None else DEFAULT_MAX_CENTER_SEGMENTS,
        max_boundary_segments=map_context.max_boundary_segments if map_context is not None else DEFAULT_MAX_BOUNDARY_SEGMENTS,
        include_traffic_lights=include_traffic_lights,
    )
    if route_points is None:
        route_points = _sample_route_points(all_states[0])
        map_stats["route_source"] = "ego_trajectory_fallback"
    else:
        map_stats["route_source"] = "map_roadblock_ids"
    map_stats["route_forward_query_points"] = len(route_forward_query_frames)

    history_slice = slice(0, history_frames)
    future_slice = slice(history_frames, total_frames)
    arrays: dict[str, np.ndarray] = {
        **map_arrays,
        "route_points": route_points,
        "goal_points": goal_points,
        "objects_history_states": all_states[:, history_slice],
        "objects_history_masks": all_masks[:, history_slice],
        "objects_future_states": all_states[:, future_slice],
        "objects_future_masks": all_masks[:, future_slice],
        "ego_index": np.array([0], dtype=np.uint32),
        "occupancy_grid": np.zeros(OCC_GRID_SHAPE, dtype=np.uint8),
    }

    meta = {
        "adapter": "nuplan_db_minimal",
        "source_db": str(db_path),
        "scenario_id": f"{db_path.stem}_{anchor.token_hex}",
        "anchor_token": anchor.token_hex,
        "anchor_timestamp_us": anchor.timestamp_us,
        "sample_rate_hz": 1.0 / dt,
        "frame_time_interval": dt,
        "history_frames_stored": history_frames,
        "future_frames": future_frames,
        "model_history_frames_expected": max(history_frames - 1, 0),
        "sampled_lidar_tokens": [frame.token_hex for frame in sampled_frames],
        "sampled_timestamps_us": [frame.timestamp_us for frame in sampled_frames],
        "scenario_tags": _scenario_tags(conn, anchor.token),
        "map_features": map_stats,
        **log_meta,
        **scene_meta,
    }
    if include_traffic_lights:
        traffic_light_status = []
        for frame_idx, frame in enumerate(sampled_frames):
            relative_idx = frame_idx - anchor_pos
            traffic_light_status.append(
                {
                    "frame_index": relative_idx,
                    "timestamp_us": frame.timestamp_us,
                    "states": tl_by_token.get(frame.token, []),
                }
            )
        meta["traffic_light_status"] = traffic_light_status
    return arrays, meta


def _write_tar_member(tar: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name=name)
    info.size = len(data)
    tar.addfile(info, io.BytesIO(data))


def _serialize_sample(arrays: dict[str, np.ndarray], meta: dict[str, object]) -> tuple[bytes, bytes]:
    npz_buf = io.BytesIO()
    np.savez_compressed(npz_buf, **arrays)
    meta_bytes = json.dumps(meta, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8")
    return npz_buf.getvalue(), meta_bytes


class ShardWriter:
    def __init__(self, output_dir: Path, samples_per_shard: int) -> None:
        self.output_dir = output_dir
        self.samples_per_shard = samples_per_shard
        self.shard_idx = 0
        self.sample_idx = 0
        self.tar: tarfile.TarFile | None = None

    def __enter__(self) -> "ShardWriter":
        self.output_dir.mkdir(parents=True, exist_ok=True)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        if self.tar is not None:
            self.tar.close()
            self.tar = None

    def _ensure_open(self) -> None:
        if self.tar is not None and self.sample_idx % self.samples_per_shard != 0:
            return
        self.close()
        shard_path = self.output_dir / f"shard_{self.shard_idx:05d}.tar"
        self.tar = tarfile.open(shard_path, "w")
        self.shard_idx += 1

    def write(self, key: str, arrays: dict[str, np.ndarray], meta: dict[str, object]) -> None:
        data_bytes, meta_bytes = _serialize_sample(arrays, meta)
        self.write_serialized(key, data_bytes, meta_bytes)

    def write_serialized(self, key: str, data_bytes: bytes, meta_bytes: bytes) -> None:
        self._ensure_open()
        assert self.tar is not None
        _write_tar_member(self.tar, f"{key}.data.npz", data_bytes)
        _write_tar_member(self.tar, f"{key}.meta.json", meta_bytes)
        self.sample_idx += 1


def _candidate_anchor_positions(
    frames: list[FrameRow],
    start_idx: int,
    stride: int,
    history_frames: int,
    future_frames: int,
    valid_scenes: set[bytes],
    anchor_tokens: set[str] | None,
    camera_alignment: bool = True,
) -> Iterator[int]:
    min_pos = (history_frames - 1) * stride
    max_pos = len(frames) - 1 - future_frames * stride
    yielded: set[int] = set()
    for pos in range(min_pos, max_pos + 1):
        frame = frames[pos]
        if anchor_tokens is not None and frame.token_hex not in anchor_tokens:
            continue
        if frame.scene_token is None or frame.scene_token not in valid_scenes:
            continue
        aligned_pos = (
            _nearest_aligned_position(pos, start_idx, stride, min_pos, max_pos)
            if camera_alignment
            else pos
        )
        if aligned_pos is not None and aligned_pos not in yielded:
            yielded.add(aligned_pos)
            yield aligned_pos


def _nearest_aligned_position(
    pos: int,
    start_idx: int,
    stride: int,
    min_pos: int,
    max_pos: int,
) -> int | None:
    lower = start_idx + ((pos - start_idx) // stride) * stride
    candidates = [candidate for candidate in (lower, lower + stride) if min_pos <= candidate <= max_pos]
    return min(candidates, key=lambda candidate: (abs(candidate - pos), candidate)) if candidates else None


def _standard_scenario_types_by_token(
    conn: sqlite3.Connection,
    *,
    remove_invalid_goals: bool = True,
) -> dict[str, str]:
    """Return devkit-like tagged scenario anchors keyed by lidar_pc token hex."""
    invalid_goal_join = (
        """
        INNER JOIN scene AS invalid_goal_scene
            ON invalid_goal_scene.token = lp.scene_token
        INNER JOIN ego_pose AS invalid_goal_ego_pose
            ON invalid_goal_scene.goal_ego_pose_token = invalid_goal_ego_pose.token
        """
        if remove_invalid_goals
        else ""
    )
    query = f"""
        SELECT
            lp.token AS token,
            MAX(st.type) AS scenario_type
        FROM lidar_pc AS lp
        INNER JOIN scenario_tag AS st
            ON lp.token = st.lidar_pc_token
        {invalid_goal_join}
        GROUP BY lp.token, lp.timestamp
        ORDER BY lp.timestamp ASC
    """
    return {
        bytes(row["token"]).hex(): str(row["scenario_type"])
        for row in conn.execute(query)
        if row["scenario_type"] is not None
    }


def _collect_standard_177k_anchors(
    db_files: list[Path],
    *,
    stride: int,
    history_frames: int,
    future_frames: int,
    num_scenarios_per_type: int,
    max_dbs: int | None,
    anchor_tokens: set[str] | None,
    stop_after: int | None,
    min_fast_types: int,
    remove_invalid_goals: bool,
) -> list[AnchorRecord]:
    """Collect and equisample anchors using the PDM/tuPlan 177k convention."""
    grouped: dict[str, list[AnchorRecord]] = {}
    min_row_idx = (history_frames - 1) * stride
    max_future_offset = future_frames * stride
    invalid_goal_join = (
        """
        INNER JOIN scene AS invalid_goal_scene
            ON invalid_goal_scene.token = olp.scene_token
        INNER JOIN ego_pose AS invalid_goal_ego_pose
            ON invalid_goal_scene.goal_ego_pose_token = invalid_goal_ego_pose.token
        """
        if remove_invalid_goals
        else ""
    )
    query = f"""
        WITH ordered_lidar_pc AS
        (
            SELECT
                lp.token AS token,
                lp.timestamp AS timestamp,
                lp.scene_token AS scene_token,
                ROW_NUMBER() OVER (ORDER BY lp.timestamp ASC) - 1 AS row_idx,
                COUNT(*) OVER () AS total_rows
            FROM lidar_pc AS lp
        ),
        ordered_scenes AS
        (
            SELECT token, ROW_NUMBER() OVER (ORDER BY name ASC) AS row_num
            FROM scene
        ),
        num_scenes AS
        (
            SELECT COUNT(*) AS cnt
            FROM scene
        ),
        valid_scenes AS
        (
            SELECT o.token
            FROM ordered_scenes AS o
            CROSS JOIN num_scenes AS n
            WHERE o.row_num >= 3 AND o.row_num < n.cnt - 1
        )
        SELECT
            olp.token AS token,
            olp.timestamp AS timestamp,
            MAX(st.type) AS scenario_type
        FROM ordered_lidar_pc AS olp
        INNER JOIN scenario_tag AS st
            ON olp.token = st.lidar_pc_token
        INNER JOIN valid_scenes AS vs
            ON olp.scene_token = vs.token
        {invalid_goal_join}
        WHERE olp.row_idx >= ?
          AND olp.row_idx <= olp.total_rows - 1 - ?
        GROUP BY olp.token, olp.timestamp
        ORDER BY olp.timestamp ASC
    """
    for db_idx, db_path in enumerate(db_files):
        if max_dbs is not None and db_idx >= max_dbs:
            break
        if db_idx > 0 and db_idx % 1000 == 0:
            print(f"Collected standard anchors from {db_idx} DBs...", file=sys.stderr, flush=True)
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        try:
            for row in conn.execute(query, (min_row_idx, max_future_offset)):
                token_hex = bytes(row["token"]).hex()
                if anchor_tokens is not None and token_hex not in anchor_tokens:
                    continue
                scenario_type = str(row["scenario_type"])
                grouped.setdefault(scenario_type, []).append(
                    AnchorRecord(
                        db_path=db_path,
                        token_hex=token_hex,
                        scenario_type=scenario_type,
                        timestamp_us=int(row["timestamp"]),
                    )
                )
                if (
                    stop_after is not None
                    and sum(len(items) for items in grouped.values()) >= stop_after
                    and len(grouped) >= min_fast_types
                ):
                    break
        finally:
            conn.close()
        if (
            stop_after is not None
            and sum(len(items) for items in grouped.values()) >= stop_after
            and len(grouped) >= min_fast_types
        ):
            break

    grouped_selected: dict[str, list[AnchorRecord]] = {}
    for scenario_type in sorted(grouped):
        anchors = grouped[scenario_type]
        if len(anchors) > num_scenarios_per_type:
            step = max(len(anchors) // num_scenarios_per_type, 1)
            anchors = anchors[::step][:num_scenarios_per_type]
        grouped_selected[scenario_type] = sorted(
            anchors,
            key=lambda anchor: (str(anchor.db_path), anchor.timestamp_us, anchor.token_hex),
        )

    selected: list[AnchorRecord] = []
    if stop_after is not None:
        while len(selected) < stop_after:
            added = False
            for scenario_type in sorted(grouped_selected):
                anchors = grouped_selected[scenario_type]
                if not anchors:
                    continue
                selected.append(anchors.pop(0))
                added = True
                if len(selected) >= stop_after:
                    break
            if not added:
                break
        return selected

    for anchors in grouped_selected.values():
        selected.extend(anchors)
    selected.sort(key=lambda anchor: (str(anchor.db_path), anchor.timestamp_us, anchor.token_hex))
    return selected


def _collect_nuplan_simulation_test_anchors(
    db_files: list[Path],
    *,
    stride: int,
    history_frames: int,
    future_frames: int,
    num_scenarios_per_type: int,
    max_dbs: int | None,
    anchor_tokens: set[str] | None,
    remove_invalid_goals: bool,
) -> list[AnchorRecord]:
    """Collect anchors using nuPlan's official simulation_test_split scenario filter."""
    grouped: dict[str, list[AnchorRecord]] = {scenario_type: [] for scenario_type in NUPLAN_SIMULATION_TEST_SCENARIO_TYPES}
    min_row_idx = (history_frames - 1) * stride
    max_future_offset = future_frames * stride
    placeholders = ",".join("?" for _ in NUPLAN_SIMULATION_TEST_SCENARIO_TYPES)
    invalid_goal_join = (
        """
        INNER JOIN scene AS invalid_goal_scene
            ON invalid_goal_scene.token = olp.scene_token
        INNER JOIN ego_pose AS invalid_goal_ego_pose
            ON invalid_goal_scene.goal_ego_pose_token = invalid_goal_ego_pose.token
        """
        if remove_invalid_goals
        else ""
    )
    query = f"""
        WITH ordered_lidar_pc AS
        (
            SELECT
                lp.token AS token,
                lp.timestamp AS timestamp,
                lp.scene_token AS scene_token,
                ROW_NUMBER() OVER (ORDER BY lp.timestamp ASC) - 1 AS row_idx,
                COUNT(*) OVER () AS total_rows
            FROM lidar_pc AS lp
        ),
        ordered_scenes AS
        (
            SELECT token, ROW_NUMBER() OVER (ORDER BY name ASC) AS row_num
            FROM scene
        ),
        num_scenes AS
        (
            SELECT COUNT(*) AS cnt
            FROM scene
        ),
        valid_scenes AS
        (
            SELECT o.token
            FROM ordered_scenes AS o
            CROSS JOIN num_scenes AS n
            WHERE o.row_num >= 3 AND o.row_num < n.cnt - 1
        )
        SELECT
            olp.token AS token,
            olp.timestamp AS timestamp,
            MAX(st.type) AS scenario_type
        FROM ordered_lidar_pc AS olp
        INNER JOIN scenario_tag AS st
            ON olp.token = st.lidar_pc_token
        INNER JOIN valid_scenes AS vs
            ON olp.scene_token = vs.token
        {invalid_goal_join}
        WHERE olp.row_idx >= ?
          AND olp.row_idx <= olp.total_rows - 1 - ?
          AND st.type IN ({placeholders})
        GROUP BY olp.token, olp.timestamp
        ORDER BY olp.timestamp ASC
    """
    query_args = (min_row_idx, max_future_offset, *NUPLAN_SIMULATION_TEST_SCENARIO_TYPES)
    for db_idx, db_path in enumerate(db_files):
        if max_dbs is not None and db_idx >= max_dbs:
            break
        if db_idx > 0 and db_idx % 1000 == 0:
            print(f"Collected nuPlan simulation-test anchors from {db_idx} DBs...", file=sys.stderr, flush=True)
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        try:
            for row in conn.execute(query, query_args):
                token_hex = bytes(row["token"]).hex()
                if anchor_tokens is not None and token_hex not in anchor_tokens:
                    continue
                scenario_type = str(row["scenario_type"])
                grouped.setdefault(scenario_type, []).append(
                    AnchorRecord(
                        db_path=db_path,
                        token_hex=token_hex,
                        scenario_type=scenario_type,
                        timestamp_us=int(row["timestamp"]),
                    )
                )
        finally:
            conn.close()

    selected: list[AnchorRecord] = []
    per_type_counts: dict[str, int] = {}
    for scenario_type in NUPLAN_SIMULATION_TEST_SCENARIO_TYPES:
        anchors = grouped.get(scenario_type, [])
        if len(anchors) > num_scenarios_per_type:
            step = max(len(anchors) // num_scenarios_per_type, 1)
            anchors = anchors[::step][:num_scenarios_per_type]
        per_type_counts[scenario_type] = len(anchors)
        selected.extend(anchors)

    selected.sort(key=lambda anchor: (str(anchor.db_path), anchor.timestamp_us, anchor.token_hex))
    print(
        "[export] nuPlan simulation-test per-type counts: "
        + json.dumps(per_type_counts, sort_keys=True),
        flush=True,
    )
    return selected


def _convert_anchor_records(
    anchors: list[AnchorRecord],
    *,
    stride: int,
    args: argparse.Namespace,
    target_dt: float,
    writer: ShardWriter,
    manifest: dict[str, object],
    map_context: MapContext | None,
    map_cache: dict[str, object],
) -> int:
    written = 0
    camera_alignment = bool(getattr(args, "camera_alignment", False))
    by_db: dict[Path, list[AnchorRecord]] = {}
    for anchor in anchors:
        by_db.setdefault(anchor.db_path, []).append(anchor)

    for db_path in sorted(by_db):
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        try:
            frames = _load_frames(conn)
            start_idx = (
                None
                if not camera_alignment or all(anchor.exact for anchor in by_db[db_path])
                else _camera_aligned_start_idx(db_path, conn)
            )
            token_to_pos = {frame.token_hex: idx for idx, frame in enumerate(frames)}
            converted_positions: set[int] = set()
            for anchor in by_db[db_path]:
                original_pos = token_to_pos.get(anchor.token_hex)
                if original_pos is None:
                    continue
                anchor_pos = (
                    original_pos
                    if not camera_alignment or anchor.exact
                    else _nearest_aligned_position(
                        original_pos,
                        int(start_idx),
                        stride,
                        (args.history_frames - 1) * stride,
                        len(frames) - 1 - args.future_frames * stride,
                    )
                )
                if anchor_pos is None or anchor_pos in converted_positions:
                    continue
                converted_positions.add(anchor_pos)
                sampled_indices = [
                    anchor_pos + (rel * stride)
                    for rel in range(-(args.history_frames - 1), args.future_frames + 1)
                ]
                arrays, meta = _build_sample(
                    conn,
                    db_path,
                    frames,
                    sampled_indices,
                    args.history_frames - 1,
                    target_dt,
                    args.num_goal_positions,
                    map_context,
                    map_cache,
                    min_vehicle_agents=args.min_vehicle_agents,
                    include_traffic_lights=not args.no_traffic_lights,
                )
                meta["standard_selection"] = {
                    "name": args.selection,
                    "scenario_type": anchor.scenario_type,
                    "num_scenarios_per_type": args.num_scenarios_per_type,
                }
                meta["source_anchor_token"] = anchor.token_hex
                key = f"{db_path.stem}_{frames[anchor_pos].token_hex}"
                writer.write(key, arrays, meta)
                manifest["samples"].append(
                    {
                        "key": key,
                        "source_db": str(db_path),
                        "anchor_token": frames[anchor_pos].token_hex,
                        "anchor_timestamp_us": frames[anchor_pos].timestamp_us,
                        "source_anchor_token": anchor.token_hex,
                        "scenario_type": anchor.scenario_type,
                    }
                )
                written += 1
                if args.max_samples is not None and written >= args.max_samples:
                    return written
        finally:
            conn.close()
    return written


def _convert_anchor_batch_serialized(
    anchors: list[AnchorRecord],
    *,
    stride: int,
    args_dict: dict[str, object],
    target_dt: float,
    map_context: MapContext | None,
) -> list[dict[str, object]]:
    args = argparse.Namespace(**args_dict)
    camera_alignment = bool(getattr(args, "camera_alignment", False))
    map_cache: dict[str, object] = {}
    results: list[dict[str, object]] = []
    by_db: dict[Path, list[AnchorRecord]] = {}
    for anchor in anchors:
        by_db.setdefault(anchor.db_path, []).append(anchor)

    for db_path in sorted(by_db):
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        try:
            frames = _load_frames(conn)
            start_idx = (
                None
                if not camera_alignment or all(anchor.exact for anchor in by_db[db_path])
                else _camera_aligned_start_idx(db_path, conn)
            )
            token_to_pos = {frame.token_hex: idx for idx, frame in enumerate(frames)}
            converted_positions: set[int] = set()
            for anchor in by_db[db_path]:
                original_pos = token_to_pos.get(anchor.token_hex)
                if original_pos is None:
                    continue
                anchor_pos = (
                    original_pos
                    if not camera_alignment or anchor.exact
                    else _nearest_aligned_position(
                        original_pos,
                        int(start_idx),
                        stride,
                        (args.history_frames - 1) * stride,
                        len(frames) - 1 - args.future_frames * stride,
                    )
                )
                if anchor_pos is None or anchor_pos in converted_positions:
                    continue
                converted_positions.add(anchor_pos)
                sampled_indices = [
                    anchor_pos + (rel * stride)
                    for rel in range(-(args.history_frames - 1), args.future_frames + 1)
                ]
                arrays, meta = _build_sample(
                    conn,
                    db_path,
                    frames,
                    sampled_indices,
                    args.history_frames - 1,
                    target_dt,
                    args.num_goal_positions,
                    map_context,
                    map_cache,
                    min_vehicle_agents=args.min_vehicle_agents,
                    include_traffic_lights=not args.no_traffic_lights,
                )
                if args.selection in {"standard_177k", "nuplan_simulation_test"}:
                    meta["standard_selection"] = {
                        "name": args.selection,
                        "scenario_type": anchor.scenario_type,
                        "num_scenarios_per_type": args.num_scenarios_per_type,
                    }
                meta["source_anchor_token"] = anchor.token_hex
                key = f"{db_path.stem}_{frames[anchor_pos].token_hex}"
                data_bytes, meta_bytes = _serialize_sample(arrays, meta)
                manifest_entry = {
                    "key": key,
                    "source_db": str(db_path),
                    "anchor_token": frames[anchor_pos].token_hex,
                    "anchor_timestamp_us": frames[anchor_pos].timestamp_us,
                    "source_anchor_token": anchor.token_hex,
                }
                if anchor.scenario_type:
                    manifest_entry["scenario_type"] = anchor.scenario_type
                results.append(
                    {
                        "key": key,
                        "data_bytes": data_bytes,
                        "meta_bytes": meta_bytes,
                        "manifest_entry": manifest_entry,
                    }
                )
        finally:
            conn.close()
    return results


def _write_serialized_results(
    *,
    results: list[dict[str, object]],
    writer: ShardWriter,
    manifest: dict[str, object],
    written_keys: set[str],
) -> int:
    written = 0
    for result in results:
        key = str(result["key"])
        if key in written_keys:
            continue
        writer.write_serialized(
            key,
            result["data_bytes"],  # type: ignore[arg-type]
            result["meta_bytes"],  # type: ignore[arg-type]
        )
        written_keys.add(key)
        manifest["samples"].append(result["manifest_entry"])
        written += 1
    return written


def _batched(items: list[AnchorRecord], batch_size: int) -> Iterator[list[AnchorRecord]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def _convert_anchor_records_ray(
    anchors: list[AnchorRecord],
    *,
    stride: int,
    args: argparse.Namespace,
    target_dt: float,
    writer: ShardWriter,
    manifest: dict[str, object],
    map_context: MapContext | None,
) -> int:
    if not anchors:
        return 0
    try:
        import ray  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError("Ray is required when --num-workers > 1. Install ray or set --num-workers 1.") from exc

    if not ray.is_initialized():
        ray.init(num_cpus=args.num_workers, ignore_reinit_error=True, include_dashboard=False)

    args_dict = vars(args).copy()
    remote_convert = ray.remote(num_cpus=1)(_convert_anchor_batch_serialized)
    batch_size = max(1, int(args.ray_batch_size))
    max_in_flight = max(1, int(args.num_workers) * 2)
    batches = iter(_batched(anchors, batch_size))
    pending: list[object] = []
    written = 0
    written_keys: set[str] = set()

    def submit_next() -> bool:
        try:
            batch = next(batches)
        except StopIteration:
            return False
        pending.append(
            remote_convert.remote(
                batch,
                stride=stride,
                args_dict=args_dict,
                target_dt=target_dt,
                map_context=map_context,
            )
        )
        return True

    for _ in range(max_in_flight):
        if not submit_next():
            break

    while pending:
        ready, pending = ray.wait(pending, num_returns=1)
        batch_results = ray.get(ready[0])
        written += _write_serialized_results(
            results=batch_results,
            writer=writer,
            manifest=manifest,
            written_keys=written_keys,
        )
        if written % args.samples_per_shard == 0:
            print(f"[export] wrote {written}/{len(anchors)} samples", flush=True)
        while len(pending) < max_in_flight and submit_next():
            pass
    print(f"[export] wrote {written}/{len(anchors)} samples", flush=True)
    return written


def _collect_all_valid_anchors(
    db_files: list[Path],
    *,
    stride: int,
    history_frames: int,
    future_frames: int,
    max_dbs: int | None,
    anchor_tokens: set[str] | None,
    max_samples: int | None,
    camera_alignment: bool = True,
) -> list[AnchorRecord]:
    anchors: list[AnchorRecord] = []
    for db_idx, db_path in enumerate(db_files):
        if max_dbs is not None and db_idx >= max_dbs:
            break
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        try:
            frames = _load_frames(conn)
            start_idx = 0
            if camera_alignment:
                try:
                    start_idx = _camera_aligned_start_idx(db_path, conn)
                except RuntimeError as exc:
                    print(f"[export] skipping {db_path.name}: {exc}", file=sys.stderr, flush=True)
                    continue
            valid_scenes = _valid_scene_tokens(conn)
            for anchor_pos in _candidate_anchor_positions(
                frames,
                start_idx,
                stride,
                history_frames,
                future_frames,
                valid_scenes,
                anchor_tokens,
                camera_alignment,
            ):
                anchors.append(
                    AnchorRecord(
                        db_path=db_path,
                        token_hex=frames[anchor_pos].token_hex,
                        scenario_type="",
                        timestamp_us=frames[anchor_pos].timestamp_us,
                        exact=not camera_alignment,
                    )
                )
                if max_samples is not None and len(anchors) >= max_samples:
                    return anchors
        finally:
            conn.close()
    return anchors


def export(args: argparse.Namespace) -> int:
    input_path = Path(args.input)
    output_dir = Path(args.output)
    db_files = _iter_db_files(input_path)
    if not db_files:
        raise FileNotFoundError(f"No .db files found under {input_path}")

    source_dt = 0.05
    target_dt = 1.0 / args.sample_rate_hz
    stride_float = target_dt / source_dt
    stride = int(round(stride_float))
    if stride <= 0 or abs(stride - stride_float) > 1e-3:
        raise ValueError(
            f"sample-rate-hz={args.sample_rate_hz} is not an integer stride from 20Hz nuPlan data"
        )
    camera_alignment = bool(getattr(args, "camera_alignment", False))
    if camera_alignment and stride % 2 != 0:
        raise ValueError(
            f"sample-rate-hz={args.sample_rate_hz} uses stride {stride}; camera-aligned sampling requires an even stride"
        )

    anchor_tokens = _read_anchor_tokens(Path(args.anchor_token_file) if args.anchor_token_file else None)
    map_root = Path(args.map_root) if args.map_root else _infer_map_root(input_path)
    map_version = args.map_version or _infer_map_version(map_root)
    if args.no_map:
        map_context = None
    else:
        map_context = MapContext(
            map_root=map_root,
            map_version=map_version,
            radius_m=args.map_radius_m,
            query_duration_s=args.map_query_duration_s,
            query_points=args.map_query_points,
            max_center_segments=args.max_center_segments,
            max_boundary_segments=args.max_boundary_segments,
        )
    map_cache: dict[str, object] = {}
    manifest = {
        "format": "driverl_webdataset_npz",
        "source": str(input_path),
        "sample_rate_hz": args.sample_rate_hz,
        "stride_from_20hz": stride,
        "camera_aligned_frame_selection": camera_alignment,
        "anchor_alignment": "nearest_start_idx_plus_k_stride" if camera_alignment else "exact_token",
        "camera_alignment_channels": list(CAMERA_CHANNELS) if camera_alignment else [],
        "history_frames": args.history_frames,
        "future_frames": args.future_frames,
        "stored_total_frames": args.history_frames + args.future_frames,
        "model_history_frames_expected": max(args.history_frames - 1, 0),
        "selection": args.selection,
        "num_scenarios_per_type": args.num_scenarios_per_type,
        "remove_invalid_goals": not args.keep_invalid_goals,
        "full_standard_scan": args.full_standard_scan,
        "max_agents": MAX_AGENTS,
        "min_vehicle_agents": args.min_vehicle_agents,
        "max_lanes_centers": map_context.max_center_segments if map_context else DEFAULT_MAX_LANES_CENTERS,
        "max_lanes_other": map_context.max_boundary_segments if map_context else DEFAULT_MAX_LANES_OTHER,
        "map_root": str(map_context.map_root) if map_context and map_context.map_root else None,
        "map_version": map_context.map_version if map_context else None,
        "map_radius_m": map_context.radius_m if map_context else None,
        "map_query_duration_s": map_context.query_duration_s if map_context else None,
        "map_query_points": map_context.query_points if map_context else None,
        "max_center_segments": map_context.max_center_segments if map_context else None,
        "max_boundary_segments": map_context.max_boundary_segments if map_context else None,
        "include_traffic_lights": not args.no_traffic_lights,
        "num_workers": args.num_workers,
        "ray_batch_size": args.ray_batch_size,
        "samples": [],
    }

    written = 0
    with ShardWriter(output_dir, args.samples_per_shard) as writer:
        if args.selection in {"standard_177k", "nuplan_simulation_test"}:
            standard_stop_after = None if args.full_standard_scan else args.max_samples
            if args.selection == "standard_177k":
                anchors = _collect_standard_177k_anchors(
                    db_files,
                    stride=stride,
                    history_frames=args.history_frames,
                    future_frames=args.future_frames,
                    num_scenarios_per_type=args.num_scenarios_per_type,
                    max_dbs=args.max_dbs,
                    anchor_tokens=anchor_tokens,
                    stop_after=standard_stop_after,
                    min_fast_types=args.fast_standard_min_types,
                    remove_invalid_goals=not args.keep_invalid_goals,
                )
                manifest["standard_177k_candidate_count"] = len(anchors)
                manifest["standard_177k_fast_smoke"] = standard_stop_after is not None
                print(f"[export] collected {len(anchors)} standard_177k anchors", flush=True)
            else:
                anchors = _collect_nuplan_simulation_test_anchors(
                    db_files,
                    stride=stride,
                    history_frames=args.history_frames,
                    future_frames=args.future_frames,
                    num_scenarios_per_type=args.num_scenarios_per_type,
                    max_dbs=args.max_dbs,
                    anchor_tokens=anchor_tokens,
                    remove_invalid_goals=not args.keep_invalid_goals,
                )
                manifest["nuplan_simulation_test_candidate_count"] = len(anchors)
                manifest["nuplan_simulation_test_scenario_types"] = list(NUPLAN_SIMULATION_TEST_SCENARIO_TYPES)
                manifest["nuplan_simulation_test_filter"] = {
                    "source_config": "nuplan/planning/script/config/common/scenario_filter/simulation_test_split.yaml",
                    "remove_invalid_goals": not args.keep_invalid_goals,
                    "shuffle": False,
                    "expand_scenarios": False,
                    "num_scenarios_per_type": args.num_scenarios_per_type,
                }
                print(f"[export] collected {len(anchors)} nuplan_simulation_test anchors", flush=True)
            if args.num_workers > 1:
                written = _convert_anchor_records_ray(
                    anchors,
                    stride=stride,
                    args=args,
                    target_dt=target_dt,
                    writer=writer,
                    manifest=manifest,
                    map_context=map_context,
                )
            else:
                written = _convert_anchor_records(
                    anchors,
                    stride=stride,
                    args=args,
                    target_dt=target_dt,
                    writer=writer,
                    manifest=manifest,
                    map_context=map_context,
                    map_cache=map_cache,
                )
        else:
            if args.num_workers > 1:
                anchors = _anchors_from_token_file(
                    Path(args.anchor_token_file) if args.anchor_token_file else None,
                    db_files,
                )
                if anchors is None:
                    anchors = _collect_all_valid_anchors(
                        db_files,
                        stride=stride,
                        history_frames=args.history_frames,
                        future_frames=args.future_frames,
                        max_dbs=args.max_dbs,
                        anchor_tokens=anchor_tokens,
                        max_samples=args.max_samples,
                        camera_alignment=camera_alignment,
                    )
                elif args.max_samples is not None:
                    anchors = anchors[: args.max_samples]
                manifest["all_valid_candidate_count"] = len(anchors)
                written = _convert_anchor_records_ray(
                    anchors,
                    stride=stride,
                    args=args,
                    target_dt=target_dt,
                    writer=writer,
                    manifest=manifest,
                    map_context=map_context,
                )
            else:
                for db_idx, db_path in enumerate(db_files):
                    if args.max_dbs is not None and db_idx >= args.max_dbs:
                        break
                    conn = sqlite3.connect(str(db_path))
                    conn.row_factory = sqlite3.Row
                    try:
                        frames = _load_frames(conn)
                        start_idx = (
                            _camera_aligned_start_idx(db_path, conn)
                            if camera_alignment
                            else 0
                        )
                        valid_scenes = _valid_scene_tokens(conn)
                        for anchor_pos in _candidate_anchor_positions(
                            frames,
                            start_idx,
                            stride,
                            args.history_frames,
                            args.future_frames,
                            valid_scenes,
                            anchor_tokens,
                            camera_alignment,
                        ):
                            sampled_indices = [
                                anchor_pos + (rel * stride)
                                for rel in range(-(args.history_frames - 1), args.future_frames + 1)
                            ]
                            arrays, meta = _build_sample(
                                conn,
                                db_path,
                                frames,
                                sampled_indices,
                                args.history_frames - 1,
                                target_dt,
                                args.num_goal_positions,
                                map_context,
                                map_cache,
                                min_vehicle_agents=args.min_vehicle_agents,
                                include_traffic_lights=not args.no_traffic_lights,
                            )
                            key = f"{db_path.stem}_{frames[anchor_pos].token_hex}"
                            writer.write(key, arrays, meta)
                            manifest["samples"].append(
                                {
                                    "key": key,
                                    "source_db": str(db_path),
                                    "anchor_token": frames[anchor_pos].token_hex,
                                    "anchor_timestamp_us": frames[anchor_pos].timestamp_us,
                                }
                            )
                            written += 1
                            if args.max_samples is not None and written >= args.max_samples:
                                break
                    finally:
                        conn.close()
                    if args.max_samples is not None and written >= args.max_samples:
                        break

    manifest["num_samples"] = written
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    if written == 0:
        raise RuntimeError("No samples were written. Check input DBs and requested history/future window.")
    return written


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="nuPlan .db file or directory containing .db files.")
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Output directory for WebDataset tar shards. Defaults to {DEFAULT_OUTPUT_DIR}.",
    )
    parser.add_argument("--sample-rate-hz", type=float, default=5.0, help="Target sampling rate. 5Hz uses stride 4.")
    parser.add_argument("--history-frames", type=int, default=21, help="Stored history frames including current frame.")
    parser.add_argument("--future-frames", type=int, default=100, help="Stored future frames after current frame.")
    parser.add_argument("--num-goal-positions", type=int, default=1, help="Flattened goal slots in goal_points.")
    parser.add_argument(
        "--min-vehicle-agents",
        type=int,
        default=64,
        help="Minimum vehicle tracks reserved within the 127 non-ego agent slots.",
    )
    parser.add_argument("--samples-per-shard", type=int, default=4096, help="Number of samples per output tar shard.")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of Ray workers for sample conversion. Values greater than 1 enable Ray.",
    )
    parser.add_argument(
        "--ray-batch-size",
        type=int,
        default=8,
        help="Number of samples processed by each Ray task.",
    )
    parser.add_argument(
        "--selection",
        choices=("all_valid", "standard_177k", "nuplan_simulation_test"),
        default="all_valid",
        help=(
            "all_valid scans every valid lidar_pc anchor. standard_177k uses tagged "
            "nuPlan scenarios with devkit-style equisampling capped per scenario type. "
            "nuplan_simulation_test mirrors nuPlan's official simulation_test_split filter."
        ),
    )
    parser.add_argument(
        "--num-scenarios-per-type",
        type=int,
        default=4000,
        help="Per-type cap used by standard_177k and nuplan_simulation_test selections.",
    )
    parser.add_argument(
        "--keep-invalid-goals",
        action="store_true",
        help="Do not drop tagged scenarios whose scene goal_ego_pose_token is missing from ego_pose.",
    )
    parser.add_argument(
        "--full-standard-scan",
        action="store_true",
        help=(
            "When using --selection standard_177k with --max-samples, scan all DBs before equisampling. "
            "By default smoke runs stop after enough tagged anchors are found."
        ),
    )
    parser.add_argument(
        "--fast-standard-min-types",
        type=int,
        default=24,
        help="Minimum scenario types to collect before stopping a fast --selection standard_177k smoke scan.",
    )
    parser.add_argument("--max-samples", type=int, default=None, help="Optional global sample limit for smoke tests.")
    parser.add_argument("--max-dbs", type=int, default=None, help="Optional DB file limit for smoke tests.")
    parser.add_argument(
        "--anchor-token-file",
        default=None,
        help="Optional file with lidar_pc anchor tokens, one per line or log_name,token.",
    )
    parser.add_argument(
        "--camera-alignment",
        action="store_true",
        help="Align requested lidar_pc tokens to the nearest camera-grid frame.",
    )
    parser.add_argument(
        "--map-root",
        default=None,
        help=(
            "nuPlan maps root. Defaults to NUPLAN_MAPS_ROOT, a sibling dataset/maps directory, "
            "or the repository-local data/nuplan/maps directory."
        ),
    )
    parser.add_argument(
        "--map-version",
        default=None,
        help=f"nuPlan map package version. Defaults to NUPLAN_MAP_VERSION or {DEFAULT_MAP_VERSION}.",
    )
    parser.add_argument(
        "--map-radius-m",
        type=float,
        default=DEFAULT_MAP_RADIUS_M,
        help="Radius around each map query ego pose for nearby map vector queries.",
    )
    parser.add_argument(
        "--map-query-duration-s",
        type=float,
        default=DEFAULT_MAP_QUERY_DURATION_S,
        help="Future time span from the current ego pose used only for map queries.",
    )
    parser.add_argument(
        "--map-query-points",
        type=int,
        default=DEFAULT_MAP_QUERY_POINTS,
        help="Number of uniformly spaced ego poses used only for map queries.",
    )
    parser.add_argument(
        "--max-center-segments",
        type=int,
        default=DEFAULT_MAX_CENTER_SEGMENTS,
        help=(
            "Lane-center tensor capacity per sample. Features beyond this capacity are truncated."
        ),
    )
    parser.add_argument(
        "--max-boundary-segments",
        type=int,
        default=DEFAULT_MAX_BOUNDARY_SEGMENTS,
        help=(
            "Other-lane tensor capacity per sample. Features beyond this capacity are truncated."
        ),
    )
    parser.add_argument("--no-map", action="store_true", help="Disable map vector queries and export zero map tensors.")
    parser.add_argument(
        "--no-traffic-lights",
        action="store_true",
        help=(
            "Skip traffic_light_status DB reads and omit traffic-light tensors/metadata "
            "from exported samples."
        ),
    )
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    if args.history_frames <= 0:
        raise ValueError("--history-frames must be positive")
    if args.future_frames < 0:
        raise ValueError("--future-frames must be non-negative")
    if args.num_goal_positions <= 0:
        raise ValueError("--num-goal-positions must be positive")
    if not 0 <= args.min_vehicle_agents < MAX_AGENTS:
        raise ValueError(f"--min-vehicle-agents must be in [0, {MAX_AGENTS - 1}]")
    if args.samples_per_shard <= 0:
        raise ValueError("--samples-per-shard must be positive")
    if args.num_workers <= 0:
        raise ValueError("--num-workers must be positive")
    if args.ray_batch_size <= 0:
        raise ValueError("--ray-batch-size must be positive")
    if args.num_scenarios_per_type <= 0:
        raise ValueError("--num-scenarios-per-type must be positive")
    if args.fast_standard_min_types <= 0:
        raise ValueError("--fast-standard-min-types must be positive")
    if args.map_radius_m <= 0:
        raise ValueError("--map-radius-m must be positive")
    if args.map_query_duration_s < 0:
        raise ValueError("--map-query-duration-s must be non-negative")
    if args.map_query_points <= 0:
        raise ValueError("--map-query-points must be positive")
    if args.max_center_segments <= 0:
        raise ValueError("--max-center-segments must be positive")
    if args.max_boundary_segments <= 0:
        raise ValueError("--max-boundary-segments must be positive")
    written = export(args)
    print(f"Wrote {written} samples to {args.output}")


if __name__ == "__main__":
    main()
