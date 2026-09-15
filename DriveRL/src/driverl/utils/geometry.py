"""Geometry utils to compute distances between boxes and handle 2D transformations."""

from typing import Tuple

import numpy as np
import torch

# We only consider 2D boxes.
NUM_VERTICES_IN_BOX = 4
NUPLAN_EGO_REAR_AXLE_TO_CENTER_M = 1.461


def shift_ego_rear_axle_to_box_center(
    posx: torch.Tensor,
    posy: torch.Tensor,
    heading: torch.Tensor,
    rear_axle_to_center: float = NUPLAN_EGO_REAR_AXLE_TO_CENTER_M,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Shift ego slot coordinates from rear axle to box center.

    nuPlan stores ego state at the rear axle while NPC states are already box
    centers. This helper only adjusts slot 0 and leaves the input tensors
    untouched.
    """
    if posx.shape[1] == 0:
        return posx, posy

    shifted_posx = posx.clone()
    shifted_posy = posy.clone()
    offset = torch.as_tensor(rear_axle_to_center, dtype=posx.dtype, device=posx.device)
    shifted_posx[:, 0] = shifted_posx[:, 0] + offset * torch.cos(heading[:, 0])
    shifted_posy[:, 0] = shifted_posy[:, 0] + offset * torch.sin(heading[:, 0])
    return shifted_posx, shifted_posy


def minkowski_sum_of_box_and_box_points(
    box1_points: torch.Tensor, box2_points: torch.Tensor
) -> torch.Tensor:
    """Batched Minkowski sum of two boxes (counter-clockwise corners in xy).

    The last dimensions of the input and return store the x and y coordinates of
    the points. Both box1_points and box2_points needs to be stored in
    counter-clockwise order. Otherwise the function will return incorrect results
    silently.

    Args:
        box1_points: Tensor of vertices for box 1, with shape:
          (num_boxes, num_points_per_box, 2).
        box2_points: Tensor of vertices for box 2, with shape:
          (num_boxes, num_points_per_box, 2).

    Returns:
        The Minkowski sum of the two boxes, of size (num_boxes,
        num_points_per_box * 2, 2). The points will be stored in counter-clockwise
        order.
    """
    # Hard coded order to pick points from the two boxes. This is a simplification
    # of the generic convex polygons case. For boxes, the adjacent edges are
    # always 90 degrees apart from each other, so the index of vertices can be
    # hard coded.
    point_order_1 = torch.tensor(
        [0, 0, 1, 1, 2, 2, 3, 3], dtype=torch.int64, device=box1_points.device
    )
    point_order_2 = torch.tensor(
        [0, 1, 1, 2, 2, 3, 3, 0], dtype=torch.int64, device=box1_points.device
    )

    box1_start_idx, downmost_box1_edge_direction = _get_downmost_edge_in_box(
        box1_points
    )
    box2_start_idx, downmost_box2_edge_direction = _get_downmost_edge_in_box(
        box2_points
    )

    # The cross-product of the unit vectors indicates whether the downmost edge
    # in box2 is pointing to the left side (the inward side of the resulting
    # Minkowski sum) of the downmost edge in box1. If this is the case, pick
    # points from box1 in the order `point_order_2`, and pick points from box2 in
    # the order of `point_order_1`. Otherwise, we switch the order to pick points
    # from the two boxes, pick points from box1 in the order of `point_order_1`,
    # and pick points from box2 in the order of `point_order_2`.
    # Shape: (num_boxes, 1)
    condition = (
        cross_product_2d(downmost_box1_edge_direction, downmost_box2_edge_direction)
        >= 0.0
    )
    # Tile condition to shape: (num_boxes, num_points_per_box * 2 = 8).
    condition = condition.repeat(1, 8)

    # box1_point_order of size [num_boxes, num_points_per_box * 2 = 8].
    box1_point_order = torch.where(condition, point_order_2, point_order_1)
    # Shift box1_point_order by box1_start_idx, so that the first index in
    # box1_point_order is the downmost vertex in the box.
    box1_point_order = torch.remainder(
        box1_point_order + box1_start_idx, NUM_VERTICES_IN_BOX
    )
    # Gather points from box1 in order.
    # ordered_box1_points is of size [num_boxes, num_points_per_box * 2, 2].
    ordered_box1_points = box1_points.gather(
        -2, box1_point_order.unsqueeze(-1).expand(-1, -1, 2)
    )

    # Gather points from box2 as well.
    box2_point_order = torch.where(condition, point_order_1, point_order_2)
    box2_point_order = torch.remainder(
        box2_point_order + box2_start_idx, NUM_VERTICES_IN_BOX
    )
    ordered_box2_points = box2_points.gather(
        -2, box2_point_order.unsqueeze(-1).expand(-1, -1, 2)
    )
    minkowski_sum = ordered_box1_points + ordered_box2_points
    return minkowski_sum


def signed_distance_from_point_to_convex_polygon(
    query_points: torch.Tensor, polygon_points: torch.Tensor
) -> torch.Tensor:
    """Finds the signed distances from query points to convex polygons.

    Each polygon is represented by a 2d tensor storing the coordinates of its
    vertices. The vertices must be ordered in counter-clockwise order. An
    arbitrary number of pairs (point, polygon) can be batched on the 1st
    dimension.

    Note: Each polygon is associated to a single query point.

    Args:
        query_points: (batch_size, 2). The last dimension is the x and y
          coordinates of points.
        polygon_points: (batch_size, num_points_per_polygon, 2). The last
          dimension is the x and y coordinates of vertices.

    Returns:
        A tensor containing the signed distances of the query points to the
        polygons. Shape: (batch_size,).
    """
    tangent_unit_vectors, normal_unit_vectors, edge_lengths = _get_edge_info(
        polygon_points
    )

    # Expand the shape of `query_points` to (num_polygons, 1, 2), so that
    # it matches the dimension of `polygons_points` for broadcasting.
    query_points = query_points.unsqueeze(1)
    # Compute query points to polygon points distances.
    # Shape (num_polygons, num_points_per_polygon, 2).
    vertices_to_query_vectors = query_points - polygon_points
    # Shape (num_polygons, num_points_per_polygon).
    vertices_distances = torch.norm(vertices_to_query_vectors, dim=-1)

    # Query point to edge distances are measured as the perpendicular distance
    # of the point from the edge. If the projection of this point on to the edge
    # falls outside the edge itself, this distance is not considered (as there)
    # will be a lower distance with the vertices of this specific edge.

    # Make distances negative if the query point is in the inward side of the
    # edge. Shape: (num_polygons, num_points_per_polygon).
    edge_signed_perp_distances = (-normal_unit_vectors * vertices_to_query_vectors).sum(
        dim=-1
    )

    # If `edge_signed_perp_distances` are all less than 0 for a
    # polygon-query_point pair, then the query point is inside the convex polygon.
    is_inside = (edge_signed_perp_distances <= 0).all(dim=-1)

    # Project the distances over the tangents of the edge, and verify where the
    # projections fall on the edge.
    # Shape: (num_polygons, num_edges_per_polygon).
    projection_along_tangent = (tangent_unit_vectors * vertices_to_query_vectors).sum(
        dim=-1
    )
    projection_along_tangent_proportion = projection_along_tangent / edge_lengths
    # Shape: (num_polygons, num_edges_per_polygon).
    is_projection_on_edge = (projection_along_tangent_proportion >= 0.0) & (
        projection_along_tangent_proportion <= 1.0
    )

    # If the point projection doesn't lay on the edge, set the distance to inf.
    edge_perp_distances = edge_signed_perp_distances.abs()
    edge_distances = torch.where(
        is_projection_on_edge, edge_perp_distances, torch.tensor(np.inf)
    )

    # Aggregate vertex and edge distances.
    # Shape: (num_polyons, 2 * num_edges_per_polygon).
    edge_and_vertex_distance = torch.cat([edge_distances, vertices_distances], dim=-1)
    # Aggregate distances per polygon and change the sign if the point lays inside
    # the polygon. Shape: (num_polygons,).
    min_distance = edge_and_vertex_distance.min(dim=-1)[0]
    signed_distances = torch.where(is_inside, -min_distance, min_distance)
    return signed_distances


def _get_downmost_edge_in_box(box: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Finds the downmost (lowest y-coordinate) edge in the box.

    Note: We assume box edges are given in a counter-clockwise order, so that
    the edge which starts with the downmost vertex (i.e. the downmost edge) is
    uniquely identified.

    Args:
        box: (num_boxes, num_points_per_box, 2). The last dimension contains the x-y
          coordinates of corners in boxes.

    Returns:
        A tuple of two tensors:
          downmost_vertex_idx: The index of the downmost vertex, which is also the
            index of the downmost edge. Shape: (num_boxes, 1).
          downmost_edge_direction: The tangent unit vector of the downmost edge,
            pointing in the counter-clockwise direction of the box.
            Shape: (num_boxes, 1, 2).
    """
    # The downmost vertex is the lowest in the y dimension.
    # Shape: (num_boxes, 1).
    downmost_vertex_idx = box[..., 1].argmin(dim=-1).unsqueeze(-1)

    # Find the counter-clockwise point edge from the downmost vertex.
    edge_start_vertex = box.gather(
        1, downmost_vertex_idx.unsqueeze(-1).expand(-1, -1, 2)
    )
    edge_end_idx = torch.remainder(downmost_vertex_idx + 1, NUM_VERTICES_IN_BOX)
    edge_end_vertex = box.gather(1, edge_end_idx.unsqueeze(-1).expand(-1, -1, 2))

    # Compute the direction of this downmost edge.
    downmost_edge = edge_end_vertex - edge_start_vertex
    downmost_edge_length = torch.norm(downmost_edge, dim=-1)
    downmost_edge_direction = downmost_edge / downmost_edge_length.unsqueeze(
        -1
    ).unsqueeze(-1)

    downmost_edge_length = torch.norm(downmost_edge, dim=-1)
    downmost_edge_direction = downmost_edge / downmost_edge_length.unsqueeze(-1)

    return downmost_vertex_idx, downmost_edge_direction


def _get_edge_info(
    polygon_points: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Computes properties about the edges of a polygon.

    Args:
        polygon_points: Tensor containing the vertices of each polygon, with
          shape (num_polygons, num_points_per_polygon, 2). Each polygon is assumed
          to have an equal number of vertices.

    Returns:
        tangent_unit_vectors: A unit vector in (x,y) with the same direction as
          the tangent to the edge. Shape: (num_polygons, num_points_per_polygon, 2).
        normal_unit_vectors: A unit vector in (x,y) with the same direction as
          the normal to the edge.
          Shape: (num_polygons, num_points_per_polygon, 2).
        edge_lengths: Lengths of the edges.
          Shape (num_polygons, num_points_per_polygon).
    """
    # Shift the polygon points by 1 position to get the edges.
    # Shape: (num_polygons, 1, 2).
    first_point_in_polygon = polygon_points[:, 0:1, :]
    # Shape: (num_polygons, num_points_per_polygon, 2).
    shifted_polygon_points = torch.cat(
        [polygon_points[:, 1:, :], first_point_in_polygon], dim=-2
    )
    # Shape: (num_polygons, num_points_per_polygon, 2).
    edge_vectors = shifted_polygon_points - polygon_points

    # Shape: (num_polygons, num_points_per_polygon).
    edge_lengths = torch.norm(edge_vectors, dim=-1)
    # Shape: (num_polygons, num_points_per_polygon, 2).
    tangent_unit_vectors = edge_vectors / edge_lengths.unsqueeze(-1)
    # Shape: (num_polygons, num_points_per_polygon, 2).
    normal_unit_vectors = torch.stack(
        [-tangent_unit_vectors[..., 1], tangent_unit_vectors[..., 0]], dim=-1
    )
    return tangent_unit_vectors, normal_unit_vectors, edge_lengths


def cross_product_2d(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Computes the signed magnitude of cross product of 2d vectors.

    Args:
        a: A tensor with shape (..., 2).
        b: A tensor with the same shape as `a`.

    Returns:
        An (n-1)-rank tensor that stores the cross products of paired 2d vectors in
        `a` and `b`.
    """
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def dot_product_2d(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Computes the dot product of 2d vectors.

    Args:
        a: A tensor with shape (..., 2).
        b: A tensor with the same shape as `a`.

    Returns:
        An (n-1)-rank tensor that stores the cross products of 2d vectors in a and
        b.
    """
    return a[..., 0] * b[..., 0] + a[..., 1] * b[..., 1]


def rotate_2d_points(xys: torch.Tensor, rotation_yaws: torch.Tensor) -> torch.Tensor:
    """Rotates `xys` counter-clockwise using the `rotation_yaws`.

    Rotates about the origin counter-clockwise in the x-y plane.

    Arguments may have differing shapes as long as they are broadcastable to a
    common shape.

    Args:
        xys: A float Tensor with shape (..., 2) containing xy coordinates.
        rotation_yaws: A float Tensor with shape (...) containing angles in
          radians.

    Returns:
        A float Tensor with shape (..., 2) containing the rotated `xys`.
    """
    rel_cos_yaws = torch.cos(rotation_yaws)
    rel_sin_yaws = torch.sin(rotation_yaws)
    xs_out = rel_cos_yaws * xys[..., 0] - rel_sin_yaws * xys[..., 1]
    ys_out = rel_sin_yaws * xys[..., 0] + rel_cos_yaws * xys[..., 1]
    return torch.stack([xs_out, ys_out], dim=-1)


def project_points_to_segments(
    points: torch.Tensor,
    seg_starts: torch.Tensor,
    seg_ends: torch.Tensor,
    eps: float = 1e-7,
    return_dist: bool = False,
) -> (
    tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
):
    """Project points onto line segments.

    Args:
        points: Tensor with shape (..., 2).
        seg_starts: Tensor with shape (..., 2), broadcastable with points.
        seg_ends: Tensor with shape (..., 2), broadcastable with points.
        eps: Small value to avoid division by zero.

    Returns:
        proj: Projected points on segments, shape (..., 2).
        t: Projection factors in [0, 1], shape (...).
        dist (optional): Distance from points to projected points, shape (...).
    """
    seg_vec = seg_ends - seg_starts
    w = points - seg_starts
    v_sq = (seg_vec * seg_vec).sum(dim=-1)
    t = (w * seg_vec).sum(dim=-1) / (v_sq + eps)
    t = torch.clamp(t, 0.0, 1.0)
    proj = seg_starts + t.unsqueeze(-1) * seg_vec
    if not return_dist:
        return proj, t
    dist = torch.linalg.norm(proj - points, dim=-1)
    return proj, t, dist


def translate_and_rotate_2d_points(
    points: torch.Tensor, translation: torch.Tensor, rotation: torch.Tensor
) -> torch.Tensor:
    """Translates and rotates a set of 2D points."""
    return rotate_2d_points(points - translation, -rotation)


def velocity_body_frame_components(
    velocities: torch.Tensor, yaws: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decompose world-frame velocities into body-frame longitudinal/lateral components."""
    cos_yaw = torch.cos(yaws)
    sin_yaw = torch.sin(yaws)
    v_long = velocities[..., 0] * cos_yaw + velocities[..., 1] * sin_yaw
    v_lat = -velocities[..., 0] * sin_yaw + velocities[..., 1] * cos_yaw
    return v_long, v_lat


def get_box_corners(boxes: torch.Tensor) -> torch.Tensor:
    """Computes the 3D corners of boxes given their parameters.

    Args:
        boxes: Tensor of shape (..., 7) containing box parameters
          [center_x, center_y, center_z, length, width, height, heading].

    Returns:
        Tensor of shape (..., 8, 3) containing the 3D corners of each box.
    """
    center_x = boxes[..., 0]
    center_y = boxes[..., 1]
    center_z = boxes[..., 2]
    length = boxes[..., 3]
    width = boxes[..., 4]
    height = boxes[..., 5]
    heading = boxes[..., 6]

    base_corners = torch.tensor(
        [
            [1, 1, 1],
            [1, 1, -1],
            [1, -1, 1],
            [1, -1, -1],
            [-1, 1, 1],
            [-1, 1, -1],
            [-1, -1, 1],
            [-1, -1, -1],
        ],
        dtype=boxes.dtype,
        device=boxes.device,
    )

    half_length = length / 2.0
    half_width = width / 2.0
    half_height = height / 2.0
    scales = torch.stack([half_length, half_width, half_height], dim=-1).unsqueeze(
        -2
    )  # (..., 1, 3)

    local_corners = base_corners * scales  # (..., 8, 3)

    c = torch.cos(heading)
    s = torch.sin(heading)
    rotation_matrix = torch.stack(
        [
            torch.stack([c, -s, torch.zeros_like(c)], dim=-1),
            torch.stack([s, c, torch.zeros_like(c)], dim=-1),
            torch.stack(
                [torch.zeros_like(c), torch.zeros_like(c), torch.ones_like(c)], dim=-1
            ),
        ],
        dim=-2,
    )  # (..., 3, 3)

    rotated_corners = torch.matmul(local_corners, rotation_matrix)  # (..., 8, 3)

    center = torch.stack([center_x, center_y, center_z], dim=-1).unsqueeze(
        -2
    )  # (..., 1, 3)
    world_corners = rotated_corners + center  # (..., 8, 3)

    return world_corners


def restore_mean(x, y, mean_x, mean_y):
    """
    In GPUDrive, everything is centered at zero by subtracting the mean.
    This function reapplies the mean to go back to the original coordinates.
    The mean (xyz) is exported per world as world_means_tensor.
    Args:
        x (torch.Tensor): x coordinates
        y (torch.Tensor): y coordinates
        mean_x (torch.Tensor): mean of x coordinates. Shape: (num_envs, 1)
        mean_y (torch.Tensor): mean of y coordinates. Shape: (num_envs, 1)
    """
    return x + mean_x, y + mean_y


def normalize_min_max(tensor, min_val, max_val):
    """Normalizes an array of values to the range [-1, 1].

    Args:
        x (np.array): Array of values to normalize.
        min_val (float): Minimum value for normalization.
        max_val (float): Maximum value for normalization.

    Returns:
        np.array: Normalized array of values.
    """
    return 2 * ((tensor - min_val) / (max_val - min_val)) - 1


def normalize_min_max_inplace(tensor, min_val, max_val):
    """Normalizes an array of values to the range [-1, 1].
    Args:
        x (np.array): Array of values to normalize.
        min_val (float): Minimum value for normalization.
        max_val (float): Maximum value for normalization.
    """
    tensor.sub_(min_val).div_(max_val - min_val).mul_(2).sub_(1)


def wrap_angle(angles: torch.Tensor) -> torch.Tensor:
    """Normalize angles to [-π, π] range.

    Args:
        angles (torch.Tensor): Input angles in radians.

    Returns:
        torch.Tensor: Normalized angles in [-π, π] range.
    """
    return (angles + torch.pi) % (2 * torch.pi) - torch.pi


def angle_difference(angles1: torch.Tensor, angles2: torch.Tensor) -> torch.Tensor:
    """Calculate the shortest angular difference between two angles, considering periodicity.

    Args:
        angles1 (torch.Tensor): First set of angles in radians.
        angles2 (torch.Tensor): Second set of angles in radians.

    Returns:
        torch.Tensor: Angular difference in [-π, π] range.
    """
    diff = angles1 - angles2
    return (diff + torch.pi) % (2 * torch.pi) - torch.pi


def get_wheelbase_from_length(length: torch.Tensor) -> torch.Tensor:
    """Estimate the wheelbase of a vehicle from its length.

    Args:
        length (torch.Tensor): The length of the vehicle.

    Returns:
        torch.Tensor: The estimated wheelbase.
    """
    return (length * 0.6006).clamp(min=1.5)


def segment_polygons_intersection2(lines: torch.Tensor, polygons: torch.Tensor):
    """
    lines: [N, 2, 2] (x, y)
    polygons: [N, A, 2] (x, y)
    eps: 浮点精度容差
    """
    N, M, _ = polygons.shape

    # 线段的两个端点是否在多边形内部
    points = lines.reshape(N * 2, 2)  # [N*2, 2]
    points_expanded = points.unsqueeze(1).expand(-1, M, -1)  # [N*2, M, 2]

    polygons_expanded = (
        polygons.unsqueeze(1).expand(-1, 2, -1, -1).reshape(N * 2, M, 2)
    )  # [N*2, M, 2]

    edges = torch.roll(polygons_expanded, -1, dims=1) - polygons_expanded  # [N*2, M, 2]

    point_to_start = points_expanded - polygons_expanded  # [N*2, M, 2]

    def cross(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]

    cross_products = cross(point_to_start, edges)

    points_inside = torch.all(cross_products >= 0, dim=1) | torch.all(
        cross_products <= 0, dim=1
    )

    any_point_inside = torch.any(points_inside.reshape(N, 2), dim=1)

    # 2. 线段是否与多边形的任何边相交
    p1, p2 = lines[:, 0], lines[:, 1]  # [N, 2]

    p1_expanded = p1.unsqueeze(1).expand(-1, M, -1)  # [N, M, 2]
    p2_expanded = p2.unsqueeze(1).expand(-1, M, -1)  # [N, M, 2]

    v1 = polygons  # [N, M, 2]
    v2 = torch.roll(polygons, -1, dims=1)  # [N, M, 2]

    line_vec = p2_expanded - p1_expanded  # [N, M, 2]

    edge_vec = v2 - v1  # [N, M, 2]

    line_edge_cross = cross(line_vec, edge_vec)  # [N, M]

    s1 = cross(p1_expanded - v1, edge_vec)  # [N, M]
    s2 = cross(p2_expanded - v1, edge_vec)  # [N, M]

    t1 = cross(v1 - p1_expanded, line_vec)  # [N, M]
    t2 = cross(v2 - p1_expanded, line_vec)  # [N, M]

    segments_intersect = (line_edge_cross != 0) & (  # 不平行或共线
        (s1 * s2 <= 0) & (t1 * t2 <= 0)
    )  # 线段和边的端点在对方的两侧

    any_edge_intersect = torch.any(segments_intersect, dim=1)

    # 最终结果：线段的点在多边形内 或 线段与多边形的边相交
    return any_point_inside | any_edge_intersect


def segment_polygons_intersection(
    lines: torch.Tensor, polygons: torch.Tensor, eps: float = 1e-7
):
    """
    lines: [N, 2, 2] (x, y)
    polygons: [N, A, 2] (x, y)
    eps: 浮点精度容差
    """
    N, A, _ = polygons.shape

    # 1. 检查线段端点是否在多边形内部
    points = lines.reshape(N * 2, 2)  # [N*2, 2]
    polygons_rep = polygons.repeat_interleave(2, dim=0)  # [N*2, A, 2]

    # 水平向右的射线
    x0, y0 = points[:, 0], points[:, 1]
    v1 = polygons_rep
    v2 = torch.roll(polygons_rep, -1, dims=1)

    # 排除水平边
    non_horizontal = torch.abs(v1[:, :, 1] - v2[:, :, 1]) > eps  # [N*2, A]

    # 计算交点参数
    dy = v2[:, :, 1] - v1[:, :, 1]
    dx = v2[:, :, 0] - v1[:, :, 0]
    with torch.no_grad():
        t = (y0[:, None] - v1[:, :, 1]) / (dy + 1e-12)  # [N*2, A]
        x_intersect = v1[:, :, 0] + t * dx

    # 有效交点条件
    valid_up = (v1[:, :, 1] < y0[:, None]) & (v2[:, :, 1] >= y0[:, None])  # 从下到上
    valid_down = (v1[:, :, 1] >= y0[:, None]) & (v2[:, :, 1] < y0[:, None])  # 从上到下
    right_intersect = (
        (x_intersect > x0[:, None]) & (t >= 0) & (t <= 1)
    )  # 右侧且在线段上

    # 统计有效交点
    valid_intersect = non_horizontal & (valid_up | valid_down) & right_intersect
    num_intersections = valid_intersect.sum(dim=1)  # [A*2]
    point_inside = (num_intersections % 2) == 1  # 奇数个交点则在内部

    # 2. 检查线段与多边形边相交
    p1, p2 = lines[:, 0], lines[:, 1]  # [N, 2]
    v1 = polygons  # [N, A, 2]
    v2 = torch.roll(polygons, -1, dims=1)  # [N, A, 2]

    line_vec = p2[:, None] - p1[:, None]  # [N, A, 2]
    edge_vec = v2 - v1  # [N, A, 2]

    def cross(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]

    # 共线检测
    line_edge_cross = cross(line_vec, edge_vec)  # [N, A]
    collinear = torch.abs(line_edge_cross) <= eps

    # 端点相对位置
    s = cross(p1[:, None] - v1, edge_vec)  # [N, A]
    t_val = cross(v1 - p1[:, None], line_vec)  # [N, A]

    # 共线时：检查线段重叠
    # seg1_collinear = collinear & (torch.abs(cross(p1[:, None] - v1, line_vec)) <= eps)
    overlap = collinear & (
        ((v1[..., 0] - p1[:, None, 0]) * (v2[..., 0] - p1[:, None, 0]) <= eps)
        | ((v1[..., 1] - p1[:, None, 1]) * (v2[..., 1] - p1[:, None, 1]) <= eps)
    )

    # 非共线时：标准相交检测
    s2 = cross(p2[:, None] - v1, edge_vec)  # [N, A]
    non_collinear_intersect = (
        (~collinear)
        & (s * s2 <= eps)
        & (t_val * cross(v2 - p1[:, None], line_vec) <= eps)
    )

    # 合并相交情况
    segments_intersect = non_collinear_intersect | overlap
    any_edge_intersect = torch.any(segments_intersect, dim=1)

    # 3. 合并结果：点包含或边相交
    any_point_inside = torch.any(point_inside.reshape(N, 2), dim=1)
    return any_point_inside | any_edge_intersect


def batch_segment_intersection2(segments1, segments2):
    """
    批量判断线段是否相交

    参数:
        segments1: 形状为[..., 2, 2]的张量，表示一组线段，每条线段由两个点(x,y)组成
        segments2: 形状为[..., 2, 2]的张量，表示另一组线段

    返回:
        形状为[...]的布尔张量，表示对应的线段对是否相交
    """
    a1 = segments1[..., 0, :]  # [..., 2]
    a2 = segments1[..., 1, :]  # [..., 2]

    b1 = segments2[..., 0, :]  # [..., 2]
    b2 = segments2[..., 1, :]  # [..., 2]

    def cross(p, q):
        return p[..., 0] * q[..., 1] - p[..., 1] * q[..., 0]

    d1 = a2 - a1  # [..., 2]
    d2 = b2 - b1  # [..., 2]
    d12 = b1 - a1  # [..., 2]

    cross_line = cross(d1, d2)  # [...]
    # check if share a line
    collinear_mask = torch.isclose(cross_line, torch.zeros_like(cross_line))

    def projections_overlap(a1, a2, b1, b2):
        min_a = torch.min(a1, a2)  # [..., 2]
        max_a = torch.max(a1, a2)  # [..., 2]
        min_b = torch.min(b1, b2)  # [..., 2]
        max_b = torch.max(b1, b2)  # [..., 2]

        overlap_x = (max_a[..., 0] >= min_b[..., 0]) & (max_b[..., 0] >= min_a[..., 0])
        overlap_y = (max_a[..., 1] >= min_b[..., 1]) & (max_b[..., 1] >= min_a[..., 1])
        return overlap_x & overlap_y

    collinear_overlap = projections_overlap(a1, a2, b1, b2)

    non_collinear_intersect = torch.zeros_like(cross_line, dtype=torch.bool)
    t = cross(d12, d2) / cross_line.clamp(min=1e-7)
    u = -cross(d12, d1) / cross_line.clamp(min=1e-7)
    non_collinear_intersect = (t >= 0) & (t <= 1) & (u >= 0) & (u <= 1)

    result = torch.where(collinear_mask, collinear_overlap, non_collinear_intersect)

    return result


def batch_segment_intersection(segments1, segments2, tol=1e-8):
    """
    批量判断线段是否相交

    参数:
        segments1: 形状为[..., 2, 2]的张量，表示一组线段，每条线段由两个点(x,y)组成
        segments2: 形状为[..., 2, 2]的张量，表示另一组线段
        tol: 数值容差，用于处理浮点误差

    返回:
        形状为[...]的布尔张量，表示对应的线段对是否相交
    """

    a1 = segments1[..., 0, :]  # [..., 2]
    a2 = segments1[..., 1, :]  # [..., 2]
    b1 = segments2[..., 0, :]  # [..., 2]
    b2 = segments2[..., 1, :]  # [..., 2]

    def cross(p, q):
        return p[..., 0] * q[..., 1] - p[..., 1] * q[..., 0]

    d1 = a2 - a1  # 线段1的方向向量 [..., 2]
    d2 = b2 - b1  # 线段2的方向向量 [..., 2]

    a1b1 = b1 - a1  # 从a1到b1的向量 [..., 2]
    a1b2 = b2 - a1  # 从a1到b2的向量 [..., 2]
    b1a1 = a1 - b1  # 从b1到a1的向量 [..., 2]
    b1a2 = a2 - b1  # 从b1到a2的向量 [..., 2]

    cross_d1_d2 = cross(d1, d2)  # 两条线段方向向量的叉积 [...]
    cross_d1_a1b1 = cross(d1, a1b1)  # d1 × (b1-a1) [...]
    cross_d1_a1b2 = cross(d1, a1b2)  # d1 × (b2-a1) [...]
    cross_d2_b1a1 = cross(d2, b1a1)  # d2 × (a1-b1) [...]
    cross_d2_b1a2 = cross(d2, b1a2)  # d2 × (a2-b1) [...]

    # 判断是否共线
    collinear_mask = torch.abs(cross_d1_d2) < 0

    # 处理非共线情况 - 使用跨立实验
    # 条件1: (b1 - a1) 和 (b2 - a1) 在d1的两侧
    cond1 = cross_d1_a1b1 * cross_d1_a1b2 < 0

    # 条件2: (a1 - b1) 和 (a2 - b1) 在d2的两侧
    cond2 = cross_d2_b1a1 * cross_d2_b1a2 < 0

    non_collinear_intersect = cond1 & cond2

    def projections_overlap(p1, q1, p2, q2):
        """检查两条线段在x和y轴上的投影是否重叠"""
        min_p1q1 = torch.minimum(p1, q1)
        max_p1q1 = torch.maximum(p1, q1)
        min_p2q2 = torch.minimum(p2, q2)
        max_p2q2 = torch.maximum(p2, q2)

        overlap_x = (max_p1q1[..., 0] >= min_p2q2[..., 0]) & (
            max_p2q2[..., 0] >= min_p1q1[..., 0]
        )

        overlap_y = (max_p1q1[..., 1] >= min_p2q2[..., 1]) & (
            max_p2q2[..., 1] >= min_p1q1[..., 1]
        )

        return overlap_x & overlap_y

    collinear_intersect = projections_overlap(a1, a2, b1, b2)

    result = torch.where(collinear_mask, collinear_intersect, non_collinear_intersect)

    return result


def extract_recent_agent_polygons(scenario_data):
    width_buffer = scenario_data.randomized_features.get("width_buffer", True)
    _, _, T = scenario_data.agent_orientation_all.shape
    length_now = scenario_data.agent_size_all[:, :, -1, 0]
    width_now = scenario_data.agent_size_all[:, :, -1, 1]
    posx_now = scenario_data.agent_positions_all[:, :, -1, 0]
    posy_now = scenario_data.agent_positions_all[:, :, -1, 1]
    orientation_now = scenario_data.agent_orientation_all[:, :, -1]

    if T < 2:
        length_last, width_last = length_now, width_now
        posx_last, posy_last = posx_now, posy_now
        orientation_last = orientation_now
    else:
        length_last = torch.where(
            scenario_data.npc_mask_all[:, :, -2],
            scenario_data.agent_size_all[:, :, -2, 0],
            length_now,
        )
        width_last = torch.where(
            scenario_data.npc_mask_all[:, :, -2],
            scenario_data.agent_size_all[:, :, -2, 1],
            width_now,
        )
        posx_last = torch.where(
            scenario_data.npc_mask_all[:, :, -2],
            scenario_data.agent_positions_all[:, :, -2, 0],
            posx_now,
        )
        posy_last = torch.where(
            scenario_data.npc_mask_all[:, :, -2],
            scenario_data.agent_positions_all[:, :, -2, 1],
            posy_now,
        )
        orientation_last = torch.where(
            scenario_data.npc_mask_all[:, :, -2],
            scenario_data.agent_orientation_all[:, :, -2],
            orientation_now,
        )

    # Width buffer is supplied by the caller for lane-boundary checks.
    width_now = width_now + width_buffer
    width_last = width_last + width_buffer
    posx_now, posy_now = shift_ego_rear_axle_to_box_center(
        posx_now, posy_now, orientation_now
    )
    posx_last, posy_last = shift_ego_rear_axle_to_box_center(
        posx_last, posy_last, orientation_last
    )

    polygons_now = vehicle_to_polygon(
        length_now, width_now, posx_now, posy_now, orientation_now
    )
    polygons_last = vehicle_to_polygon(
        length_last, width_last, posx_last, posy_last, orientation_last
    )

    valid = scenario_data.npc_mask_all[:, :, -1]
    return polygons_now, polygons_last, valid


def vehicle_to_polygon(
    length: torch.Tensor,
    width: torch.Tensor,
    posx: torch.Tensor,
    posy: torch.Tensor,
    theta: torch.Tensor,
) -> torch.Tensor:
    half_length = length / 2
    half_width = width / 2  # [N, A]

    length_x = half_length * torch.cos(theta)
    width_x = half_width * torch.sin(theta)
    length_y = half_length * torch.sin(theta)
    width_y = half_width * torch.cos(theta)  # [N, A]

    front_left_x = posx + length_x - width_x
    front_left_y = posy + length_y + width_y  # [N, A]

    front_right_x = posx + length_x + width_x
    front_right_y = posy + length_y - width_y  # [N, A]

    rear_right_x = posx - length_x + width_x
    rear_right_y = posy - length_y - width_y  # [N, A]

    rear_left_x = posx - length_x - width_x
    rear_left_y = posy - length_y + width_y  # [N, A]

    front_left = torch.stack([front_left_x, front_left_y], dim=-1)  # [N, A, 2]
    front_right = torch.stack([front_right_x, front_right_y], dim=-1)
    rear_left = torch.stack([rear_left_x, rear_left_y], dim=-1)
    rear_right = torch.stack([rear_right_x, rear_right_y], dim=-1)

    polygons = torch.stack(
        [
            front_left,
            front_right,
            rear_right,
            rear_left,
        ],
        dim=-2,
    )  # [N, A, 4, 2]

    return polygons


def expand_polygon_with_buffer(polygon, width_buffer, length_buffer):
    width_dir = polygon[:, 1] - polygon[:, 0]
    length_dir = polygon[:, 2] - polygon[:, 1]

    width_dir = width_dir / (torch.norm(width_dir, dim=1, keepdim=True) + 1e-6)
    length_dir = length_dir / (torch.norm(length_dir, dim=1, keepdim=True) + 1e-6)

    width_buffer = width_buffer.view(-1, 1, 1)
    length_buffer = length_buffer.view(-1, 1, 1)

    # Fast path for vehicle boxes in canonical corner order:
    # [front_left, front_right, rear_right, rear_left].
    # This avoids center/projection/sign computation and is materially faster.
    if polygon.shape[1] == 4:
        width_vec = width_buffer * width_dir.unsqueeze(1)  # [N, 1, 2]
        length_vec = length_buffer * length_dir.unsqueeze(1)  # [N, 1, 2]
        expanded = torch.cat(
            [
                polygon[:, 0:1] - width_vec - length_vec,
                polygon[:, 1:2] + width_vec - length_vec,
                polygon[:, 2:3] + width_vec + length_vec,
                polygon[:, 3:4] - width_vec + length_vec,
            ],
            dim=1,
        )
        return expanded

    center = polygon.mean(dim=1, keepdim=True)
    rel = polygon - center

    width_comp = torch.sum(rel * width_dir.unsqueeze(1), dim=2, keepdim=True)
    length_comp = torch.sum(rel * length_dir.unsqueeze(1), dim=2, keepdim=True)
    width_sign = torch.sign(width_comp)
    length_sign = torch.sign(length_comp)

    expanded = (
        polygon
        + width_sign * width_buffer * width_dir.unsqueeze(1)
        + length_sign * length_buffer * length_dir.unsqueeze(1)
    )

    return expanded
