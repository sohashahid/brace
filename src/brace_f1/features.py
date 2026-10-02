"""Causal state and circuit-relative features for BRACE models.

Only present and past measurements enter this module.  Outcome labels live in
the target table and are deliberately absent from the input allow-list below.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from brace_f1.io import DataValidationError

FEATURE_COLUMNS: tuple[str, ...] = (
    "yaw_rad",
    "yaw_rate_radps",
    "body_speed_longitudinal_mps",
    "body_speed_lateral_mps",
    "track_heading_rad",
    "heading_error_rad",
    "track_offset_m",
    "track_curvature_per_m",
    "body_acceleration_longitudinal_mps2",
    "body_acceleration_lateral_mps2",
)

_IDENTIFIER_COLUMNS: tuple[str, ...] = (
    "circuit",
    "car_id",
    "frame_index",
    "time_seconds",
    "continuous_segment_id",
    "input_valid_causal",
    "in_corridor",
    "map_x_m",
    "map_y_m",
)

_REQUIRED_COLUMNS = {
    *_IDENTIFIER_COLUMNS,
    "hard_break",
    "map_x_m",
    "map_y_m",
    "velocity_x_mps",
    "velocity_y_mps",
    "body_acceleration_longitudinal_mps2",
    "body_acceleration_lateral_mps2",
    "forward_x_map",
    "forward_y_map",
    "centerline_segment_index_pcd",
}


def _wrap_angle(angle: NDArray[np.float64]) -> NDArray[np.float64]:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def yaw_from_quaternions_xyzw(quaternions: ArrayLike) -> NDArray[np.float64]:
    """Return planar yaw from normalized quaternions in explicit ``[x,y,z,w]`` order."""

    values = np.asarray(quaternions, dtype=np.float64)
    if values.ndim == 1:
        values = values[None, :]
    if values.ndim != 2 or values.shape[1] != 4:
        raise DataValidationError("quaternions must have shape (N, 4) in xyzw order")
    if not np.isfinite(values).all():
        raise DataValidationError("quaternions contain non-finite values")
    norms = np.linalg.norm(values, axis=1)
    if np.any(norms <= 1e-12):
        raise DataValidationError("quaternions must have non-zero norm")
    x, y, z, w = (values / norms[:, None]).T
    sin_yaw = 2.0 * (w * z + x * y)
    cos_yaw = 1.0 - 2.0 * (y * y + z * z)
    return _wrap_angle(np.arctan2(sin_yaw, cos_yaw))


@dataclass(frozen=True)
class TrackReference:
    """Directed closed centerline with fixed local heading and curvature."""

    centerline_xy: NDArray[np.float64]

    def __init__(self, centerline_xy: ArrayLike) -> None:
        centerline = np.asarray(centerline_xy, dtype=np.float64)
        if centerline.ndim != 2 or centerline.shape[1] < 2 or centerline.shape[0] < 3:
            raise DataValidationError("centerline must contain at least three planar points")
        centerline = centerline[:, :2]
        if np.allclose(centerline[0], centerline[-1]):
            centerline = centerline[:-1]
        if centerline.shape[0] < 3 or not np.isfinite(centerline).all():
            raise DataValidationError("centerline is non-finite or degenerate")
        vectors = np.roll(centerline, -1, axis=0) - centerline
        lengths = np.linalg.norm(vectors, axis=1)
        if np.any(lengths <= 1e-12):
            raise DataValidationError("centerline contains a zero-length segment")
        object.__setattr__(self, "centerline_xy", centerline.copy())

    @property
    def segment_count(self) -> int:
        return int(self.centerline_xy.shape[0])

    @property
    def track_length_m(self) -> float:
        vectors = np.roll(self.centerline_xy, -1, axis=0) - self.centerline_xy
        return float(np.linalg.norm(vectors, axis=1).sum())

    def sample(
        self, *, points_xy: ArrayLike, segment_indices: ArrayLike
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Return signed offset, segment heading, and forward curvature."""

        points = np.asarray(points_xy, dtype=np.float64)
        indices = np.asarray(segment_indices, dtype=np.int64)
        if points.ndim != 2 or points.shape[1] != 2 or indices.shape != (points.shape[0],):
            raise DataValidationError(
                "points and centerline segment indices have incompatible shapes"
            )
        if (
            not np.isfinite(points).all()
            or np.any(indices < 0)
            or np.any(indices >= self.segment_count)
        ):
            raise DataValidationError("points or centerline segment indices are invalid")
        starts = self.centerline_xy
        vectors = np.roll(starts, -1, axis=0) - starts
        lengths = np.linalg.norm(vectors, axis=1)
        tangents = vectors / lengths[:, None]
        selected_tangent = tangents[indices]
        selected_start = starts[indices]
        along = np.einsum("ij,ij->i", points - selected_start, selected_tangent)
        projection = selected_start + along[:, None] * selected_tangent
        offset_vector = points - projection
        signed_offset = (
            selected_tangent[:, 0] * offset_vector[:, 1]
            - selected_tangent[:, 1] * offset_vector[:, 0]
        )
        headings = np.arctan2(tangents[:, 1], tangents[:, 0])
        next_headings = np.roll(headings, -1)
        heading_change = _wrap_angle(next_headings - headings)
        curvature = heading_change / (0.5 * (lengths + np.roll(lengths, -1)))
        return signed_offset, headings[indices], curvature[indices]


def _validate_frame_order(frames: pd.DataFrame) -> None:
    for _, group in frames.groupby(["circuit", "car_id"], sort=False, dropna=False):
        times = group["time_seconds"].to_numpy(dtype=np.float64)
        if not np.isfinite(times).all() or np.any(np.diff(times) <= 0.0):
            raise DataValidationError("frame times must be finite and strictly increasing per car")
        segment = group["continuous_segment_id"].to_numpy(dtype=np.int64)
        if np.any(np.diff(segment) < 0):
            raise DataValidationError("continuous segment identifiers must be monotone per car")


def derive_causal_features(
    frames: pd.DataFrame, track_references: Mapping[str, TrackReference]
) -> pd.DataFrame:
    """Derive prefix-invariant 20 Hz state features from an explicit input allow-list."""

    missing = _REQUIRED_COLUMNS.difference(frames.columns)
    if missing:
        raise DataValidationError(f"feature input missing columns: {sorted(missing)}")
    if frames.empty:
        raise DataValidationError("feature input is empty")
    _validate_frame_order(frames)
    output = frames.loc[:, _IDENTIFIER_COLUMNS].copy().reset_index(drop=True)
    working = frames.reset_index(drop=True)
    yaw = _wrap_angle(
        np.arctan2(
            working["forward_y_map"].to_numpy(dtype=np.float64),
            working["forward_x_map"].to_numpy(dtype=np.float64),
        )
    )
    if not np.isfinite(yaw).all():
        raise DataValidationError("quaternion-derived forward vectors are non-finite")
    output["yaw_rad"] = yaw
    yaw_rate = np.zeros(len(working), dtype=np.float64)
    for _, group in working.groupby(
        ["circuit", "car_id", "continuous_segment_id"], sort=False, dropna=False
    ):
        index = group.index.to_numpy(dtype=np.int64)
        if index.size <= 1:
            continue
        dt = np.diff(group["time_seconds"].to_numpy(dtype=np.float64))
        dyaw = _wrap_angle(np.diff(yaw[index]))
        yaw_rate[index[1:]] = dyaw / dt
    yaw_rate[working["hard_break"].to_numpy(dtype=bool)] = 0.0
    output["yaw_rate_radps"] = yaw_rate

    vx = working["velocity_x_mps"].to_numpy(dtype=np.float64)
    vy = working["velocity_y_mps"].to_numpy(dtype=np.float64)
    if not np.isfinite(vx).all() or not np.isfinite(vy).all():
        raise DataValidationError("velocity features are non-finite")
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    output["body_speed_longitudinal_mps"] = cos_yaw * vx + sin_yaw * vy
    output["body_speed_lateral_mps"] = -sin_yaw * vx + cos_yaw * vy

    offset = np.empty(len(working), dtype=np.float64)
    track_heading = np.empty(len(working), dtype=np.float64)
    curvature = np.empty(len(working), dtype=np.float64)
    for circuit, index in working.groupby("circuit", sort=False).groups.items():
        circuit_name = str(circuit)
        if circuit_name not in track_references:
            raise DataValidationError(f"missing centerline reference for circuit {circuit_name}")
        rows = np.asarray(index, dtype=np.int64)
        group_offset, group_heading, group_curvature = track_references[circuit_name].sample(
            points_xy=working.loc[rows, ["map_x_m", "map_y_m"]].to_numpy(dtype=np.float64),
            segment_indices=working.loc[rows, "centerline_segment_index_pcd"].to_numpy(
                dtype=np.int64
            ),
        )
        offset[rows] = group_offset
        track_heading[rows] = group_heading
        curvature[rows] = group_curvature
    output["track_heading_rad"] = track_heading
    output["heading_error_rad"] = _wrap_angle(yaw - track_heading)
    output["track_offset_m"] = offset
    output["track_curvature_per_m"] = curvature
    output["body_acceleration_longitudinal_mps2"] = working[
        "body_acceleration_longitudinal_mps2"
    ].to_numpy(dtype=np.float64)
    output["body_acceleration_lateral_mps2"] = working["body_acceleration_lateral_mps2"].to_numpy(
        dtype=np.float64
    )
    if not np.isfinite(output.loc[:, FEATURE_COLUMNS].to_numpy(dtype=np.float64)).all():
        raise DataValidationError("derived feature table contains non-finite values")
    return output


@dataclass(frozen=True)
class FitOnlyRobustScaler:
    """Median/IQR scaler whose parameters are estimated from fit rows only."""

    feature_columns: tuple[str, ...]
    center: NDArray[np.float64]
    scale: NDArray[np.float64]
    content_hash: str

    def __post_init__(self) -> None:
        columns = tuple(self.feature_columns)
        center = np.array(self.center, dtype=np.float64, copy=True)
        scale = np.array(self.scale, dtype=np.float64, copy=True)
        if (
            len(columns) == 0
            or len(set(columns)) != len(columns)
            or center.shape != (len(columns),)
            or scale.shape != center.shape
            or not np.isfinite(center).all()
            or not np.isfinite(scale).all()
            or np.any(scale <= 0.0)
        ):
            raise DataValidationError("robust scaler parameters are invalid")
        center.setflags(write=False)
        scale.setflags(write=False)
        object.__setattr__(self, "feature_columns", columns)
        object.__setattr__(self, "center", center)
        object.__setattr__(self, "scale", scale)

    @classmethod
    def fit_from_partition(
        cls,
        table: pd.DataFrame,
        feature_columns: Sequence[str],
        *,
        partition_column: str = "partition",
        fit_value: str = "fit",
    ) -> FitOnlyRobustScaler:
        missing = set(feature_columns).union({partition_column}).difference(table.columns)
        if missing:
            raise DataValidationError(f"scaler input missing columns: {sorted(missing)}")
        selected = table.loc[table[partition_column] == fit_value, list(feature_columns)]
        if selected.empty:
            raise DataValidationError("scaler has no fit-partition rows")
        values = selected.to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise DataValidationError("fit-partition scaler values must be finite")
        center = np.median(values, axis=0)
        scale = np.percentile(values, 75.0, axis=0) - np.percentile(values, 25.0, axis=0)
        scale = np.where(scale > 1e-12, scale, 1.0)
        payload = {
            "feature_columns": list(feature_columns),
            "center": center.tolist(),
            "scale": scale.tolist(),
        }
        content_hash = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        return cls(tuple(feature_columns), center, scale, content_hash)

    def transform(self, table: pd.DataFrame) -> pd.DataFrame:
        missing = set(self.feature_columns).difference(table.columns)
        if missing:
            raise DataValidationError(f"scaler transform missing columns: {sorted(missing)}")
        output = table.copy()
        values = output.loc[:, self.feature_columns].to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise DataValidationError("scaler transform values must be finite")
        output.loc[:, self.feature_columns] = (values - self.center) / self.scale
        return output
