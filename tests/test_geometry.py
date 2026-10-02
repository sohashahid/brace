from __future__ import annotations

import math

import numpy as np
import pytest

from brace_f1.geometry import CircuitCorridor, transform_world_kinematics
from brace_f1.io import DataValidationError, DeepRacingMetadata


def _metadata(
    origin: tuple[float, float, float], quaternion: tuple[float, float, float, float]
) -> DeepRacingMetadata:
    return DeepRacingMetadata(
        trackname="Fixture",
        clockwise=True,
        frequency_hz=20.0,
        timestep_seconds=0.05,
        origin=origin,
        quaternion_xyzw=quaternion,
    )


def test_transform_world_kinematics_identity_translation() -> None:
    positions = np.asarray([[2.0, 4.0, 6.0], [0.0, 2.0, 3.0]])
    velocities = np.asarray([[4.0, 5.0, 6.0], [1.0, 2.0, 3.0]])
    accelerations = np.asarray([[0.1, 0.2, 0.3], [-1.0, 0.0, 1.0]])
    quaternions = np.tile(np.asarray([0.0, 0.0, 0.0, 1.0]), (2, 1))

    transformed = transform_world_kinematics(
        positions,
        velocities,
        accelerations,
        quaternions,
        _metadata((1.0, 2.0, 3.0), (0.0, 0.0, 0.0, 1.0)),
    )

    np.testing.assert_allclose(transformed.positions, [[1, 2, 3], [-1, 0, 0]], atol=1e-12)
    np.testing.assert_allclose(transformed.velocities, velocities, atol=1e-12)
    np.testing.assert_allclose(transformed.accelerations, accelerations, atol=1e-12)
    np.testing.assert_allclose(transformed.body_accelerations, accelerations, atol=1e-12)
    np.testing.assert_allclose(transformed.forward_vectors, [[1, 0, 0], [1, 0, 0]], atol=1e-12)
    np.testing.assert_allclose(transformed.yaw_map, [0, 0], atol=1e-12)


def test_transform_world_kinematics_uses_documented_row_vector_rotation() -> None:
    root_half = math.sqrt(0.5)
    transformed = transform_world_kinematics(
        np.asarray([[1.0, 0.0, 0.0]]),
        np.asarray([[0.0, 1.0, 0.0]]),
        np.asarray([[1.0, 1.0, 0.0]]),
        np.asarray([0.0, 0.0, 0.0, 1.0]),
        _metadata((0.0, 0.0, 0.0), (0.0, 0.0, root_half, root_half)),
    )

    np.testing.assert_allclose(transformed.positions, [[0.0, -1.0, 0.0]], atol=1e-12)
    np.testing.assert_allclose(transformed.velocities, [[1.0, 0.0, 0.0]], atol=1e-12)
    np.testing.assert_allclose(transformed.accelerations, [[1.0, -1.0, 0.0]], atol=1e-12)


def test_body_acceleration_uses_active_vehicle_body_to_world_rotation() -> None:
    root_half = math.sqrt(0.5)
    transformed = transform_world_kinematics(
        np.asarray([[0.0, 0.0, 0.0]]),
        np.asarray([[0.0, 1.0, 0.0]]),
        np.asarray([[2.0, 0.0, 0.0]]),
        np.asarray([[0.0, 0.0, root_half, root_half]]),
        _metadata((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
    )

    np.testing.assert_allclose(transformed.accelerations, [[0.0, 2.0, 0.0]], atol=1e-12)
    np.testing.assert_allclose(transformed.forward_vectors, [[0.0, 1.0, 0.0]], atol=1e-12)
    np.testing.assert_allclose(transformed.yaw_map, [math.pi / 2], atol=1e-12)


def test_vehicle_then_metadata_rotation_is_applied_in_sequence() -> None:
    root_half = math.sqrt(0.5)
    transformed = transform_world_kinematics(
        np.asarray([[0.0, 0.0, 0.0]]),
        np.asarray([[0.0, 3.0, 0.0]]),
        np.asarray([[3.0, 0.0, 0.0]]),
        np.asarray([[0.0, 0.0, root_half, root_half]]),
        _metadata((0.0, 0.0, 0.0), (0.0, 0.0, root_half, root_half)),
    )

    np.testing.assert_allclose(transformed.velocities, [[3.0, 0.0, 0.0]], atol=1e-12)
    np.testing.assert_allclose(transformed.accelerations, [[3.0, 0.0, 0.0]], atol=1e-12)
    np.testing.assert_allclose(transformed.forward_vectors, [[1.0, 0.0, 0.0]], atol=1e-12)


def test_vehicle_quaternion_accepts_scalar_or_batch_and_rejects_nonunit() -> None:
    positions = np.zeros((2, 3))
    velocities = np.zeros((2, 3))
    accelerations = np.tile(np.asarray([1.0, 0.0, 0.0]), (2, 1))
    metadata = _metadata((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))

    scalar = transform_world_kinematics(
        positions, velocities, accelerations, np.asarray([0.0, 0.0, 0.0, 1.0]), metadata
    )
    batch = transform_world_kinematics(
        positions,
        velocities,
        accelerations,
        np.tile(np.asarray([0.0, 0.0, 0.0, 1.0]), (2, 1)),
        metadata,
    )

    np.testing.assert_allclose(scalar.accelerations, batch.accelerations)
    with pytest.raises(DataValidationError, match="unit quaternions"):
        transform_world_kinematics(
            positions,
            velocities,
            accelerations,
            np.asarray([0.0, 0.0, 0.0, 2.0]),
            metadata,
        )


def test_transformed_acceleration_matches_constructed_velocity_derivative() -> None:
    velocities = np.asarray([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.4, 0.0, 0.0]])
    transformed = transform_world_kinematics(
        np.zeros((3, 3)),
        velocities,
        np.tile(np.asarray([2.0, 0.0, 0.0]), (3, 1)),
        np.asarray([0.0, 0.0, 0.0, 1.0]),
        _metadata((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
    )

    finite_difference = np.diff(transformed.velocities[:, 0]) / 0.1
    np.testing.assert_allclose(finite_difference, transformed.accelerations[1:, 0], atol=1e-12)


def _rectangular_corridor() -> CircuitCorridor:
    centerline = np.asarray([[-3, -3], [3, -3], [3, 3], [-3, 3]], dtype=float)
    geometric_outer_but_named_inner = np.asarray([[-5, -5], [5, -5], [5, 5], [-5, 5]], dtype=float)
    geometric_hole_but_named_outer = np.asarray([[-1, -1], [-1, 1], [1, 1], [1, -1]], dtype=float)
    return CircuitCorridor.from_arrays(
        centerline=centerline,
        boundary_a=geometric_hole_but_named_outer,
        boundary_b=geometric_outer_but_named_inner,
        boundary_a_id="outer_boundary.pcd",
        boundary_b_id="inner_boundary.pcd",
    )


def test_corridor_inference_uses_geometry_not_boundary_filename() -> None:
    corridor = _rectangular_corridor()

    assert corridor.outer_loop.source_id == "inner_boundary.pcd"
    assert corridor.inner_hole.source_id == "outer_boundary.pcd"

    result = corridor.locate(np.asarray([[1.5, 0.0], [0.0, 0.0], [6.0, 0.0]]))

    assert result.in_corridor.tolist() == [True, False, False]
    assert result.nearest_boundary_id.tolist() == [
        "outer_boundary.pcd",
        "outer_boundary.pcd",
        "inner_boundary.pcd",
    ]
    np.testing.assert_allclose(result.clearance_m, [0.5, -1.0, -1.0], atol=1e-12)
    np.testing.assert_allclose(result.outside_depth_m, [0.0, 1.0, 1.0], atol=1e-12)


def test_corridor_treats_outer_and_inner_boundary_points_as_in_corridor() -> None:
    result = _rectangular_corridor().locate(np.asarray([[5.0, 0.0], [1.0, 0.0]]))

    assert result.in_corridor.tolist() == [True, True]
    np.testing.assert_allclose(result.clearance_m, [0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(result.outside_depth_m, [0.0, 0.0], atol=1e-12)


def test_fast_contains_matches_full_location_including_boundary_tolerance() -> None:
    corridor = _rectangular_corridor()
    tolerance = corridor.boundary_tolerance_m
    points = np.asarray(
        [
            [2.0, 0.0],
            [0.0, 0.0],
            [6.0, 0.0],
            [5.0, 0.0],
            [1.0, 0.0],
            [5.0 + 0.5 * tolerance, 0.0],
            [1.0 - 0.5 * tolerance, 0.0],
            [5.0 + 2.0 * tolerance, 0.0],
        ]
    )
    np.testing.assert_array_equal(
        corridor.contains_planar(points), corridor.locate(points).in_corridor
    )


def test_fast_contains_uses_euclidean_tolerance_at_polygon_corners() -> None:
    corridor = _rectangular_corridor()
    tolerance = corridor.boundary_tolerance_m
    diagonal_inside_tolerance = 0.6 * tolerance
    diagonal_outside_tolerance = 0.8 * tolerance
    points = np.asarray(
        [
            [5.0 + diagonal_inside_tolerance, 5.0 + diagonal_inside_tolerance],
            [5.0 + diagonal_outside_tolerance, 5.0 + diagonal_outside_tolerance],
            [1.0 - diagonal_inside_tolerance, 1.0 - diagonal_inside_tolerance],
            [1.0 - diagonal_outside_tolerance, 1.0 - diagonal_outside_tolerance],
        ]
    )
    np.testing.assert_array_equal(
        corridor.contains_planar(points), corridor.locate(points).in_corridor
    )


def test_swept_segment_finds_first_exit_when_both_endpoints_are_inside() -> None:
    corridor = _rectangular_corridor()
    start = np.asarray([[-4.0, 0.0], [0.0, -4.0]])
    end = np.asarray([[4.0, 0.0], [0.0, 4.0]])

    crossing = corridor.first_exit_along_segments(start, end)

    assert crossing.has_exit.tolist() == [True, True]
    np.testing.assert_allclose(crossing.fraction, [3.0 / 8.0, 3.0 / 8.0], atol=1e-7)
    np.testing.assert_allclose(crossing.point, [[-1.0, 0.0], [0.0, -1.0]], atol=1e-6)


def test_track_features_planar_follow_directed_centerline() -> None:
    corridor = _rectangular_corridor()

    track = corridor.track_features_planar(np.asarray([[0.0, -2.0], [2.0, 0.0]]))

    np.testing.assert_allclose(track.heading_rad, [0.0, np.pi / 2.0], atol=1e-12)
    np.testing.assert_allclose(track.offset_m, [1.0, 1.0], atol=1e-12)
    assert np.isfinite(track.curvature_per_m).all()


def test_corridor_reports_local_side_segment_and_arclength() -> None:
    corridor = _rectangular_corridor()
    result = corridor.locate(np.asarray([[0.0, -2.0], [0.0, -4.0]]))

    assert corridor.track_length_m == 24.0
    assert result.centerline_segment.tolist() == [0, 0]
    np.testing.assert_allclose(result.centerline_arclength_m, [3.0, 3.0], atol=1e-12)
    np.testing.assert_allclose(result.centerline_arclength_wrapped_m, [3.0, 3.0], atol=1e-12)
    assert np.all(result.centerline_arclength_wrapped_m < corridor.track_length_m)
    assert result.local_side.tolist() == ["left", "right"]
