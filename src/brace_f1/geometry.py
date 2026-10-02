"""Coordinate transforms and plan-view track-corridor geometry."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from matplotlib.path import Path as PolygonPath
from numpy.typing import ArrayLike, NDArray
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from shapely import (
    Polygon,
    boundary,
    covers,
    get_coordinates,
    intersection,
    intersects_xy,
    linestrings,
    prepare,
)

from brace_f1.io import DataValidationError, DeepRacingMetadata, PointCloud


@dataclass(frozen=True)
class MapKinematics:
    """World-frame kinematics transformed into the DeepRacing map frame."""

    positions: NDArray[np.float64]
    velocities: NDArray[np.float64]
    accelerations: NDArray[np.float64]
    body_accelerations: NDArray[np.float64]
    forward_vectors: NDArray[np.float64]
    yaw_map: NDArray[np.float64]


@dataclass(frozen=True)
class BoundaryLoop:
    """One closed boundary loop and its source identity."""

    vertices: NDArray[np.float64]
    source_id: str


@dataclass(frozen=True)
class CorridorLocation:
    """Plan-view relationship between query points and a circuit corridor."""

    in_corridor: NDArray[np.bool_]
    nearest_boundary_id: NDArray[np.str_]
    nearest_boundary_role: NDArray[np.str_]
    clearance_m: NDArray[np.float64]
    outside_depth_m: NDArray[np.float64]
    local_side: NDArray[np.str_]
    centerline_segment: NDArray[np.int64]
    centerline_arclength_m: NDArray[np.float64]
    centerline_arclength_wrapped_m: NDArray[np.float64]


@dataclass(frozen=True)
class SegmentExit:
    """First corridor exit along each directed planar segment."""

    has_exit: NDArray[np.bool_]
    fraction: NDArray[np.float64]
    point: NDArray[np.float64]


@dataclass(frozen=True)
class TrackPlanarFeatures:
    """Directed centerline features evaluated at arbitrary planar points."""

    heading_rad: NDArray[np.float64]
    offset_m: NDArray[np.float64]
    curvature_per_m: NDArray[np.float64]


@dataclass(frozen=True)
class _Projection:
    distance: NDArray[np.float64]
    segment: NDArray[np.int64]
    fraction: NDArray[np.float64]
    point: NDArray[np.float64]
    tangent: NDArray[np.float64]


def _as_finite_vectors(values: ArrayLike, name: str) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3:
        raise DataValidationError(f"{name} must have shape (N, 3)")
    if not np.isfinite(array).all():
        raise DataValidationError(f"{name} contains non-finite values")
    return array


def transform_world_kinematics(
    positions: ArrayLike,
    velocities: ArrayLike,
    body_accelerations: ArrayLike,
    vehicle_quaternions_xyzw: ArrayLike,
    metadata: DeepRacingMetadata,
) -> MapKinematics:
    """Transform world motion and body acceleration into the circuit map frame.

    Per-frame vehicle quaternions are active body-to-world rotations. Metadata
    supplies the world-to-map row-vector transform. Therefore
    ``a_map = a_body @ R_vehicle.T @ R_map``.
    """

    position_array = _as_finite_vectors(positions, "positions")
    velocity_array = _as_finite_vectors(velocities, "velocities")
    acceleration_array = _as_finite_vectors(body_accelerations, "body_accelerations")
    if not (position_array.shape[0] == velocity_array.shape[0] == acceleration_array.shape[0]):
        raise DataValidationError("position, velocity, and acceleration lengths differ")
    sample_count = position_array.shape[0]
    vehicle_quaternions = np.asarray(vehicle_quaternions_xyzw, dtype=np.float64)
    if vehicle_quaternions.shape == (4,):
        vehicle_quaternions = np.broadcast_to(vehicle_quaternions, (sample_count, 4)).copy()
    if vehicle_quaternions.shape != (sample_count, 4):
        raise DataValidationError(
            f"vehicle_quaternions_xyzw must have shape (4,) or ({sample_count}, 4)"
        )
    if not np.isfinite(vehicle_quaternions).all():
        raise DataValidationError("vehicle quaternions contain non-finite values")
    quaternion_norms = np.linalg.norm(vehicle_quaternions, axis=1)
    if not np.allclose(quaternion_norms, 1.0, rtol=1e-5, atol=1e-5):
        raise DataValidationError("vehicle orientations must be unit quaternions")

    map_rotation = Rotation.from_quat(metadata.quaternion_xyzw).as_matrix()
    vehicle_rotation = Rotation.from_quat(vehicle_quaternions)
    world_acceleration = vehicle_rotation.apply(acceleration_array)
    world_forward = vehicle_rotation.apply(
        np.broadcast_to(np.asarray([1.0, 0.0, 0.0]), (sample_count, 3))
    )
    origin = np.asarray(metadata.origin, dtype=np.float64)
    map_forward = world_forward @ map_rotation
    yaw_map = np.unwrap(np.arctan2(map_forward[:, 1], map_forward[:, 0]))
    return MapKinematics(
        positions=(position_array - origin) @ map_rotation,
        velocities=velocity_array @ map_rotation,
        accelerations=world_acceleration @ map_rotation,
        body_accelerations=acceleration_array,
        forward_vectors=map_forward,
        yaw_map=yaw_map,
    )


def _as_planar_loop(values: ArrayLike, name: str) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] < 2 or array.shape[0] < 3:
        raise DataValidationError(f"{name} must contain at least three planar vertices")
    array = array[:, :2]
    if not np.isfinite(array).all():
        raise DataValidationError(f"{name} contains non-finite vertices")
    if np.allclose(array[0], array[-1]):
        array = array[:-1]
    if array.shape[0] < 3:
        raise DataValidationError(f"{name} has fewer than three distinct vertices")
    segment_lengths = np.linalg.norm(np.roll(array, -1, axis=0) - array, axis=1)
    if np.any(segment_lengths <= 1e-12):
        raise DataValidationError(f"{name} contains a zero-length segment")
    return array


def _signed_area(vertices: NDArray[np.float64]) -> float:
    next_vertices = np.roll(vertices, -1, axis=0)
    return 0.5 * float(
        np.sum(vertices[:, 0] * next_vertices[:, 1] - next_vertices[:, 0] * vertices[:, 1])
    )


def _polygon_path(vertices: NDArray[np.float64]) -> PolygonPath:
    oriented = vertices if _signed_area(vertices) > 0.0 else vertices[::-1]
    return PolygonPath(np.vstack((oriented, oriented[0])), closed=True)


def _nearest_polyline_projection(
    points: NDArray[np.float64],
    vertices: NDArray[np.float64],
    candidate_count: int = 32,
    *,
    midpoint_tree: cKDTree | None = None,
) -> _Projection:
    starts = vertices
    ends = np.roll(vertices, -1, axis=0)
    vectors = ends - starts
    squared_lengths = np.einsum("ij,ij->i", vectors, vectors)
    midpoints = (starts + ends) * 0.5
    tree = cKDTree(midpoints) if midpoint_tree is None else midpoint_tree
    k = min(candidate_count, starts.shape[0])
    # Each outer fold runs in its own process. Two query workers per fold use
    # the available cores without the severe oversubscription of workers=-1.
    _, candidates = tree.query(points, k=k, workers=2)
    if k == 1:
        candidates = np.asarray(candidates)[:, None]
    candidate_starts = starts[candidates]
    candidate_vectors = vectors[candidates]
    offsets = points[:, None, :] - candidate_starts
    fractions = np.einsum("nki,nki->nk", offsets, candidate_vectors) / squared_lengths[candidates]
    fractions = np.clip(fractions, 0.0, 1.0)
    projections = candidate_starts + fractions[..., None] * candidate_vectors
    deltas = points[:, None, :] - projections
    distance_squared = np.einsum("nki,nki->nk", deltas, deltas)
    winners = np.argmin(distance_squared, axis=1)
    rows = np.arange(points.shape[0])
    segment = np.asarray(candidates[rows, winners], dtype=np.int64)
    fraction = fractions[rows, winners]
    projection = projections[rows, winners]
    distance = np.sqrt(distance_squared[rows, winners])
    tangent = vectors[segment] / np.sqrt(squared_lengths[segment])[:, None]
    return _Projection(distance, segment, fraction, projection, tangent)


class CircuitCorridor:
    """A closed outer polygon, inner hole, and directed centerline."""

    def __init__(
        self,
        *,
        centerline: NDArray[np.float64],
        outer_loop: BoundaryLoop,
        inner_hole: BoundaryLoop,
        boundary_tolerance_m: float = 1e-7,
    ) -> None:
        self.centerline = centerline
        self.outer_loop = outer_loop
        self.inner_hole = inner_hole
        self.boundary_tolerance_m = boundary_tolerance_m
        self._outer_path = _polygon_path(outer_loop.vertices)
        self._inner_path = _polygon_path(inner_hole.vertices)
        corridor_polygon = Polygon(
            shell=outer_loop.vertices,
            holes=[inner_hole.vertices],
        )
        if corridor_polygon.is_empty or not corridor_polygon.is_valid:
            raise DataValidationError("outer loop and inner hole do not form a valid corridor")
        if boundary_tolerance_m > 0.0:
            corridor_polygon = corridor_polygon.buffer(boundary_tolerance_m)
        prepare(corridor_polygon)
        self._prepared_corridor_polygon = corridor_polygon
        self._corridor_boundary = boundary(corridor_polygon)
        center_segments = np.roll(centerline, -1, axis=0) - centerline
        self._center_segment_lengths = np.linalg.norm(center_segments, axis=1)
        self._center_segment_tangents = center_segments / self._center_segment_lengths[:, None]
        self._center_segment_headings = np.arctan2(
            self._center_segment_tangents[:, 1], self._center_segment_tangents[:, 0]
        )
        next_heading = np.roll(self._center_segment_headings, -1)
        heading_change = (next_heading - self._center_segment_headings + np.pi) % (
            2.0 * np.pi
        ) - np.pi
        self._center_curvature = heading_change / (
            0.5 * (self._center_segment_lengths + np.roll(self._center_segment_lengths, -1))
        )
        self._center_midpoint_tree = cKDTree((centerline + np.roll(centerline, -1, axis=0)) * 0.5)
        self.track_length_m = float(np.sum(self._center_segment_lengths))
        self._center_arclength_starts = np.concatenate(
            (np.asarray([0.0]), np.cumsum(self._center_segment_lengths[:-1]))
        )

    @classmethod
    def from_arrays(
        cls,
        *,
        centerline: ArrayLike,
        boundary_a: ArrayLike,
        boundary_b: ArrayLike,
        boundary_a_id: str = "boundary_a",
        boundary_b_id: str = "boundary_b",
    ) -> CircuitCorridor:
        """Infer exterior and hole from absolute polygon area, independent of filenames."""

        center = _as_planar_loop(centerline, "centerline")
        a = _as_planar_loop(boundary_a, boundary_a_id)
        b = _as_planar_loop(boundary_b, boundary_b_id)
        area_a = abs(_signed_area(a))
        area_b = abs(_signed_area(b))
        if np.isclose(area_a, area_b, rtol=1e-10, atol=1e-10):
            raise DataValidationError("cannot infer outer loop from equal boundary areas")
        if area_a > area_b:
            outer = BoundaryLoop(a, boundary_a_id)
            inner = BoundaryLoop(b, boundary_b_id)
        else:
            outer = BoundaryLoop(b, boundary_b_id)
            inner = BoundaryLoop(a, boundary_a_id)
        return cls(centerline=center, outer_loop=outer, inner_hole=inner)

    @classmethod
    def from_point_clouds(
        cls,
        *,
        centerline: PointCloud,
        boundary_a: PointCloud,
        boundary_b: PointCloud,
        boundary_a_id: str,
        boundary_b_id: str,
    ) -> CircuitCorridor:
        """Construct a corridor from validated PCD clouds."""

        return cls.from_arrays(
            centerline=centerline.xyz,
            boundary_a=boundary_a.xyz,
            boundary_b=boundary_b.xyz,
            boundary_a_id=boundary_a_id,
            boundary_b_id=boundary_b_id,
        )

    def locate(self, points: ArrayLike) -> CorridorLocation:
        """Compute corridor membership, signed clearance, boundary, side, and arclength."""

        query = np.asarray(points, dtype=np.float64)
        if query.ndim != 2 or query.shape[1] < 2:
            raise DataValidationError("query points must have shape (N, 2+) ")
        query = query[:, :2]
        if not np.isfinite(query).all():
            raise DataValidationError("query points contain non-finite values")

        outer_projection = _nearest_polyline_projection(query, self.outer_loop.vertices)
        inner_projection = _nearest_polyline_projection(query, self.inner_hole.vertices)
        nearest_is_outer = outer_projection.distance <= inner_projection.distance
        nearest_distance = np.where(
            nearest_is_outer, outer_projection.distance, inner_projection.distance
        )
        nearest_boundary_id = np.where(
            nearest_is_outer, self.outer_loop.source_id, self.inner_hole.source_id
        )
        nearest_boundary_role = np.where(nearest_is_outer, "outer_loop", "inner_hole")

        inside_outer = self._outer_path.contains_points(query)
        inside_hole = self._inner_path.contains_points(query)
        on_boundary = nearest_distance <= self.boundary_tolerance_m
        in_corridor = (inside_outer & ~inside_hole) | on_boundary
        clearance = np.where(in_corridor, nearest_distance, -nearest_distance)
        outside_depth = np.where(in_corridor, 0.0, nearest_distance)

        center_projection = _nearest_polyline_projection(
            query, self.centerline, midpoint_tree=self._center_midpoint_tree
        )
        center_arclength = (
            self._center_arclength_starts[center_projection.segment]
            + center_projection.fraction * self._center_segment_lengths[center_projection.segment]
        )
        center_arclength_wrapped = np.mod(center_arclength, self.track_length_m)
        offset = query - center_projection.point
        cross = (
            center_projection.tangent[:, 0] * offset[:, 1]
            - center_projection.tangent[:, 1] * offset[:, 0]
        )
        local_side = np.where(
            cross > self.boundary_tolerance_m,
            "left",
            np.where(cross < -self.boundary_tolerance_m, "right", "center"),
        )
        return CorridorLocation(
            in_corridor=np.asarray(in_corridor, dtype=np.bool_),
            nearest_boundary_id=np.asarray(nearest_boundary_id, dtype=np.str_),
            nearest_boundary_role=np.asarray(nearest_boundary_role, dtype=np.str_),
            clearance_m=np.asarray(clearance, dtype=np.float64),
            outside_depth_m=np.asarray(outside_depth, dtype=np.float64),
            local_side=np.asarray(local_side, dtype=np.str_),
            centerline_segment=center_projection.segment,
            centerline_arclength_m=np.asarray(center_arclength, dtype=np.float64),
            centerline_arclength_wrapped_m=np.asarray(center_arclength_wrapped, dtype=np.float64),
        )

    def contains_planar(self, points: ArrayLike) -> NDArray[np.bool_]:
        """Return corridor membership without boundary or centerline projections.

        A prepared GEOS polygon-with-hole is buffered by the declared Euclidean
        boundary tolerance at construction.  ``intersects_xy`` includes its
        boundary and evaluates all points without allocating Point objects.
        """

        query = np.asarray(points, dtype=np.float64)
        if query.ndim != 2 or query.shape[1] < 2:
            raise DataValidationError("query points must have shape (N, 2+) ")
        query = query[:, :2]
        if not np.isfinite(query).all():
            raise DataValidationError("query points contain non-finite values")
        return np.asarray(
            intersects_xy(self._prepared_corridor_polygon, query[:, 0], query[:, 1]),
            dtype=np.bool_,
        )

    def first_exit_along_segments(
        self, start_points: ArrayLike, end_points: ArrayLike
    ) -> SegmentExit:
        """Return the first continuous corridor exit on each directed segment.

        Endpoint membership alone can miss a segment that crosses the infield
        hole and re-enters within one integration step.  This method first uses
        the prepared polygon to identify segments that are not wholly covered,
        then intersects only those segments with the polygon boundary.
        """

        start = np.asarray(start_points, dtype=np.float64)
        end = np.asarray(end_points, dtype=np.float64)
        if start.ndim != 2 or start.shape[1] < 2 or end.shape != start.shape:
            raise DataValidationError("segment endpoints must have matching shape (N, 2+)")
        start = start[:, :2]
        end = end[:, :2]
        if not np.isfinite(start).all() or not np.isfinite(end).all():
            raise DataValidationError("segment endpoints contain non-finite values")
        count = start.shape[0]
        has_exit = np.zeros(count, dtype=np.bool_)
        fraction = np.full(count, np.nan, dtype=np.float64)
        point = np.full((count, 2), np.nan, dtype=np.float64)
        if count == 0:
            return SegmentExit(has_exit, fraction, point)

        segments = linestrings(np.stack((start, end), axis=1))
        starts_inside = self.contains_planar(start)
        wholly_covered = np.asarray(
            covers(self._prepared_corridor_polygon, segments), dtype=np.bool_
        )
        candidates = np.flatnonzero(starts_inside & ~wholly_covered)
        if not candidates.size:
            return SegmentExit(has_exit, fraction, point)

        boundary_hits = intersection(segments[candidates], self._corridor_boundary)
        coordinates, owner = get_coordinates(boundary_hits, return_index=True)
        delta = end[candidates] - start[candidates]
        squared_length = np.einsum("ij,ij->i", delta, delta)
        for local_index in range(candidates.size):
            hit_coordinates = coordinates[owner == local_index]
            if not hit_coordinates.size or squared_length[local_index] <= 0.0:
                continue
            projected_fraction = (
                (hit_coordinates - start[candidates[local_index]])
                @ delta[local_index]
                / squared_length[local_index]
            )
            valid = projected_fraction[
                (projected_fraction >= -1e-12) & (projected_fraction <= 1.0 + 1e-12)
            ]
            if not valid.size:
                continue
            first_fraction = float(np.clip(valid.min(), 0.0, 1.0))
            row = candidates[local_index]
            has_exit[row] = True
            fraction[row] = first_fraction
            point[row] = start[row] + first_fraction * (end[row] - start[row])
        return SegmentExit(has_exit, fraction, point)

    def track_features_planar(self, points: ArrayLike) -> TrackPlanarFeatures:
        """Return directed centerline heading, signed offset, and curvature."""

        query = np.asarray(points, dtype=np.float64)
        if query.ndim != 2 or query.shape[1] < 2:
            raise DataValidationError("query points must have shape (N, 2+) ")
        query = query[:, :2]
        if not np.isfinite(query).all():
            raise DataValidationError("query points contain non-finite values")
        projection = _nearest_polyline_projection(
            query, self.centerline, midpoint_tree=self._center_midpoint_tree
        )
        offset = query - projection.point
        signed_offset = (
            projection.tangent[:, 0] * offset[:, 1] - projection.tangent[:, 1] * offset[:, 0]
        )
        return TrackPlanarFeatures(
            heading_rad=self._center_segment_headings[projection.segment].copy(),
            offset_m=np.asarray(signed_offset, dtype=np.float64),
            curvature_per_m=self._center_curvature[projection.segment].copy(),
        )
