"""Route-roadblock correction used by runtime nuPlan feature building."""

from __future__ import annotations

from collections import deque
from typing import Any, Iterable

import numpy as np
from nuplan.common.maps.maps_datatypes import SemanticMapLayer


def _map_object(map_api: Any, roadblock_id: str) -> Any | None:
    block = map_api.get_map_object(roadblock_id, SemanticMapLayer.ROADBLOCK)
    return block or map_api.get_map_object(
        roadblock_id, SemanticMapLayer.ROADBLOCK_CONNECTOR
    )


def _find_path(
    start: Any,
    target_ids: Iterable[str],
    *,
    forward: bool,
    max_depth: int,
) -> list[Any] | None:
    targets = set(target_ids)
    queue = deque([(start, [start])])
    visited = {start.id}
    while queue:
        current, path = queue.popleft()
        if current.id in targets:
            return path
        if len(path) >= max_depth:
            continue
        neighbors = current.outgoing_edges if forward else current.incoming_edges
        for neighbor in neighbors:
            if neighbor.id not in visited:
                visited.add(neighbor.id)
                queue.append((neighbor, [*path, neighbor]))
    return None


def _current_roadblock_candidates(
    ego_state: Any,
    map_api: Any,
    route_ids: set[str],
) -> list[Any]:
    layers = [SemanticMapLayer.ROADBLOCK, SemanticMapLayer.ROADBLOCK_CONNECTOR]
    nearby = map_api.get_proximal_map_objects(
        point=ego_state.rear_axle.point, radius=1.0, layers=layers
    )
    candidates = [item for layer in layers for item in nearby[layer]]
    if not candidates:
        for layer in layers:
            roadblock_id, _ = map_api.get_distance_to_nearest_map_object(
                point=ego_state.rear_axle.point, layer=layer
            )
            block = map_api.get_map_object(roadblock_id, layer)
            if block is not None:
                candidates.append(block)

    scored: list[tuple[bool, float, Any]] = []
    for block in candidates:
        best_distance = np.inf
        heading_matches = False
        for lane in block.interior_edges:
            path = lane.baseline_path.discrete_path
            points = np.asarray([state.point.array for state in path])
            distances = np.linalg.norm(points - ego_state.rear_axle.point.array, axis=1)
            index = int(np.argmin(distances))
            heading_error = abs(
                (path[index].heading - ego_state.rear_axle.heading + np.pi)
                % (2 * np.pi)
                - np.pi
            )
            best_distance = min(best_distance, float(distances[index]))
            heading_matches |= heading_error < np.pi / 4 and distances[index] < 3.0
        if heading_matches:
            scored.append((block.id not in route_ids, best_distance, block))
    if scored:
        scored.sort(key=lambda item: (item[0], item[1]))
        best_key = scored[0][:2]
        return [item[2] for item in scored if item[:2] == best_key]
    return sorted(candidates, key=lambda block: block.id)


def _remove_route_loops(route: list[Any]) -> list[Any]:
    connector_polygons: list[Any] = []
    for index, block in enumerate(route):
        if "RoadBlockConnector" not in type(block).__name__:
            continue
        polygon = block.polygon
        if any(polygon.intersection(previous).area > 1.0 for previous in connector_polygons):
            return route[:index]
        connector_polygons.append(polygon)
    return route


def correct_route_roadblock_ids(
    ego_state: Any,
    map_api: Any,
    route_roadblock_ids: list[str],
    *,
    search_depth_backward: int = 15,
    search_depth_forward: int = 30,
) -> list[str]:
    """Repair off-route starts, disconnected route segments, and route loops."""
    route = [
        block
        for roadblock_id in dict.fromkeys(map(str, route_roadblock_ids))
        if (block := _map_object(map_api, roadblock_id)) is not None
    ]
    if not route:
        return []

    candidates = _current_roadblock_candidates(
        ego_state, map_api, {block.id for block in route}
    )
    if candidates and all(candidate.id not in {block.id for block in route} for candidate in candidates):
        path = _find_path(
            route[0],
            (candidate.id for candidate in candidates),
            forward=False,
            max_depth=search_depth_backward,
        )
        if path:
            route = [*reversed(path[1:]), *route]
        else:
            path = _find_path(
                candidates[0],
                (block.id for block in route[:3]),
                forward=True,
                max_depth=search_depth_forward,
            )
            if path:
                end_index = next(i for i, block in enumerate(route) if block.id == path[-1].id)
                route = [*path, *route[end_index + 1 :]]

    corrected: list[Any] = [route[0]]
    for next_block in route[1:]:
        if corrected[-1].id not in {block.id for block in next_block.incoming_edges}:
            path = _find_path(
                corrected[-1],
                [next_block.id],
                forward=True,
                max_depth=search_depth_forward,
            )
            if path:
                corrected.extend(path[1:-1])
        corrected.append(next_block)

    return [block.id for block in _remove_route_loops(corrected)]
