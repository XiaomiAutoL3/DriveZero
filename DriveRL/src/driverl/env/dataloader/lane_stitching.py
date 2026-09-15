"""Stitch lane segments into complete polylines using group IDs and topology.

Given fragmented lane center segments (each with 2 endpoints), their group
assignments, and optional successor group information, this module reconstructs
full lane polylines by:

1. Concatenating segments within the same group into per-group polylines.
2. Chaining groups via successor topology (``next_groups``) to form longer
   polylines that span multiple lanes.

If a group has multiple successors, the predecessor polyline is duplicated — one
copy for each successor — so each resulting polyline represents a distinct path.
"""

from __future__ import annotations

import numpy as np

INVALID_GROUP_ID = np.iinfo(np.uint32).max

DEFAULT_MAX_LINES = 8
DEFAULT_MAX_POINTS_PER_LINE = 128


def _build_group_polylines(
    points: np.ndarray,
    groups: np.ndarray,
    valid_indices: np.ndarray,
    valid_groups: np.ndarray,
    unique_groups: np.ndarray,
) -> dict[int, np.ndarray]:
    """Build a polyline for each group by concatenating its segment endpoints.

    Returns a dict mapping group ID -> polyline array of shape ``[K+1, 2]``.
    """
    group_polylines: dict[int, np.ndarray] = {}
    for gid in unique_groups:
        seg_mask = valid_groups == gid
        seg_indices = valid_indices[seg_mask]
        seg_starts = points[seg_indices, 0, :]  # [K, 2]
        last_end = points[seg_indices[-1], 1, :]  # [2]
        group_polylines[int(gid)] = np.concatenate(
            [seg_starts, last_end[np.newaxis, :]], axis=0
        )
    return group_polylines


def _build_successor_map(
    next_groups: np.ndarray | None,
    unique_groups: np.ndarray,
) -> dict[int, list[int]]:
    """Extract per-group successor list from ``next_groups``.

    ``next_groups`` has shape ``[max_groups, max_successors]`` and is indexed
    directly by group ID.  Only successors that actually exist in
    ``unique_groups`` (and are not self-loops) are kept.
    """
    if next_groups is None:
        return {}

    valid_group_set = set(unique_groups.tolist())
    successor_map: dict[int, list[int]] = {}

    for gid in unique_groups:
        gid_int = int(gid)
        if gid_int >= next_groups.shape[0]:
            continue
        ng_row = next_groups[gid_int]
        valid_ng = ng_row[ng_row != INVALID_GROUP_ID]
        if len(valid_ng) == 0:
            continue
        unique_ng = set(int(v) for v in np.unique(valid_ng))
        candidates = sorted(unique_ng & valid_group_set - {gid_int})
        if candidates:
            successor_map[gid_int] = candidates

    return successor_map


def _chain_polylines(
    group_polylines: dict[int, np.ndarray],
    successor_map: dict[int, list[int]],
) -> list[np.ndarray]:
    """Chain group polylines using successor topology.

    Starting from every group that has no predecessor (i.e. is not a successor
    of any other group), walk the successor graph and concatenate polylines.

    If a group has multiple successors, the path up to that point is duplicated
    for each branch.

    Groups that form isolated nodes (no predecessors, no successors) are also
    emitted as standalone polylines.
    """
    has_predecessor: set[int] = set()
    for succs in successor_map.values():
        has_predecessor.update(succs)

    all_groups = set(group_polylines.keys())
    roots = sorted(all_groups - has_predecessor)

    result: list[np.ndarray] = []
    visited: set[int] = set()

    def _walk(gid: int, prefix_parts: list[np.ndarray]) -> None:
        if gid in visited or gid not in group_polylines:
            if prefix_parts:
                result.append(np.concatenate(prefix_parts, axis=0))
            return

        visited.add(gid)
        poly = group_polylines[gid]
        if prefix_parts:
            last_pt = prefix_parts[-1][-1]
            if np.linalg.norm(poly[0] - last_pt) < 1.0:
                poly = poly[1:]
            if len(poly) == 0:
                result.append(np.concatenate(prefix_parts, axis=0))
                return

        current_parts = prefix_parts + [poly]

        successors = successor_map.get(gid, [])
        if not successors:
            result.append(np.concatenate(current_parts, axis=0))
        else:
            for succ in successors:
                _walk(succ, list(current_parts))

    for root in roots:
        _walk(root, [])

    for gid in sorted(all_groups):
        if gid not in visited:
            result.append(group_polylines[gid])

    return result


def stitch_lane_segments(
    points: np.ndarray,
    masks: np.ndarray,
    groups: np.ndarray,
    next_groups: np.ndarray | None = None,
    *,
    max_lines: int = DEFAULT_MAX_LINES,
    max_points: int = DEFAULT_MAX_POINTS_PER_LINE,
) -> tuple[np.ndarray, np.ndarray]:
    """Stitch lane segments into complete polylines using group IDs and topology.

    Args:
        points: Lane segment endpoints, shape ``[L_seg, 2, 2]`` (float32).
        masks: Validity mask for segments, shape ``[L_seg]`` (bool).
        groups: Group ID per segment, shape ``[L_seg]`` (uint32).
        next_groups: Optional successor group IDs per group,
            shape ``[max_groups, max_successors]`` (uint32).
            ``INVALID_GROUP_ID`` marks unused slots.  Indexed by group ID.
        max_lines: Maximum number of output polylines.
        max_points: Maximum number of points per polyline.

    Returns:
        lines: Stitched polylines, shape ``[max_lines, max_points, 2]``.
        lines_mask: Validity mask per polyline, shape ``[max_lines]``.
    """
    lines = np.zeros((max_lines, max_points, 2), dtype=np.float32)
    lines_mask = np.zeros(max_lines, dtype=np.bool_)

    valid = masks & (groups != INVALID_GROUP_ID)
    if not valid.any():
        return lines, lines_mask

    valid_indices = np.nonzero(valid)[0]
    valid_groups = groups[valid_indices]

    _, first_occurrence = np.unique(valid_groups, return_index=True)
    unique_groups = valid_groups[np.sort(first_occurrence)]

    group_polylines = _build_group_polylines(
        points, groups, valid_indices, valid_groups, unique_groups
    )

    successor_map = _build_successor_map(next_groups, unique_groups)

    if successor_map:
        polylines = _chain_polylines(group_polylines, successor_map)
    else:
        polylines = [group_polylines[int(gid)] for gid in unique_groups]

    polyline_dists = [float(np.abs(poly[:, 1]).min()) for poly in polylines]
    sort_order = np.argsort(polyline_dists)
    selected = sort_order[:max_lines]

    for out_idx, poly_idx in enumerate(selected):
        polyline = polylines[poly_idx]
        num_pts = polyline.shape[0]

        if num_pts >= max_points:
            indices = np.linspace(0, num_pts - 1, max_points, dtype=int)
            lines[out_idx] = polyline[indices]
        else:
            lines[out_idx, :num_pts] = polyline
            lines[out_idx, num_pts:] = polyline[-1]

        lines_mask[out_idx] = True

    return lines, lines_mask
