#!/usr/bin/env python3
import copy
import os
from typing import Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

from minimalist_compliance_control.utils import ensure_matrix, get_damping_matrix

TODDY_LEFT_CAMERA_POSITION: npt.NDArray[np.float32] = np.array(
    [0.01, -0.004 + 0.033 / 2.0, 0.048], dtype=np.float32
)
TODDY_LEFT_CAMERA_EULER: npt.NDArray[np.float32] = np.array(
    [-np.pi / 2, 0.0, -np.pi / 2], dtype=np.float32
)
TODDY_LEFT_CAMERA_EULER = (
    (
        R.from_euler("xyz", TODDY_LEFT_CAMERA_EULER)
        * R.from_euler("yx", [5 / 180 * np.pi, 0 / 180 * np.pi])
    )
    .as_euler("xyz")
    .astype(np.float32)
)
TODDY_RIGHT_CAMERA_POSITION: npt.NDArray[np.float32] = np.array(
    [0.01, -0.004 - 0.033 / 2.0, 0.048], dtype=np.float32
)
TODDY_RIGHT_CAMERA_EULER: npt.NDArray[np.float32] = TODDY_LEFT_CAMERA_EULER.copy()

LEAP_LEFT_CAMERA_POSITION: npt.NDArray[np.float32] = np.array(
    [0.1807, 0.057 + 0.033 / 2, 0.0379], dtype=np.float32
)
LEAP_LEFT_CAMERA_EULER: npt.NDArray[np.float32] = np.array(
    [0, np.pi, np.pi / 2], dtype=np.float32
)
LEAP_LEFT_CAMERA_EULER = (
    (
        R.from_euler("xyz", LEAP_LEFT_CAMERA_EULER)
        * R.from_euler("yx", [5 / 180 * np.pi, 0 / 180 * np.pi])
    )
    .as_euler("xyz")
    .astype(np.float32)
)
LEAP_RIGHT_CAMERA_POSITION: npt.NDArray[np.float32] = np.array(
    [0.1807, 0.057 - 0.033 / 2, 0.0379], dtype=np.float32
)
LEAP_RIGHT_CAMERA_EULER: npt.NDArray[np.float32] = LEAP_LEFT_CAMERA_EULER.copy()


def build_camera_to_head(
    position: npt.NDArray[np.float32], euler_xyz: npt.NDArray[np.float32]
) -> npt.NDArray[np.float32]:
    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = R.from_euler("xyz", euler_xyz).as_matrix().astype(np.float32)
    transform[:3, 3] = position
    return transform


TODDY_CAMERA_EXTRINSICS = {
    "left": build_camera_to_head(TODDY_LEFT_CAMERA_POSITION, TODDY_LEFT_CAMERA_EULER),
    "right": build_camera_to_head(
        TODDY_RIGHT_CAMERA_POSITION, TODDY_RIGHT_CAMERA_EULER
    ),
}

LEAP_CAMERA_EXTRINSICS = {
    "left": build_camera_to_head(LEAP_LEFT_CAMERA_POSITION, LEAP_LEFT_CAMERA_EULER),
    "right": build_camera_to_head(LEAP_RIGHT_CAMERA_POSITION, LEAP_RIGHT_CAMERA_EULER),
}

DEFAULT_CONTACT_PREP_DISTANCE: float = 0.03

TODDY_TOOL_OFFSETS = {
    "pen": {
        "rot_offset": R.from_euler("xyz", [0, -np.pi / 2, 0])
        .as_matrix()
        .astype(np.float32),
        "contact_offset_vec": np.array([0.0296, 0.0, -0.012], dtype=np.float32),
    },
    "eraser": {
        "rot_offset": R.from_euler("xyz", [0, -np.pi / 2, 0])
        .as_matrix()
        .astype(np.float32),
        "contact_offset_vec": np.array([0.0, 0.0, -0.01], dtype=np.float32),
    },
}

LEAP_TOOL_OFFSETS = {
    "pen": {
        "rot_offset": R.from_euler("xyz", [np.pi / 2, 0, 0])
        .as_matrix()
        .astype(np.float32),
        "rot_offset_by_site": {
            "th_tip": R.from_euler("xyz", [np.pi, 0, 0]).as_matrix().astype(np.float32)
        },
        "contact_offset_vec": np.zeros(3, dtype=np.float32),
        # np.array([0.055, 0.0, 0.01], dtype=np.float32),
    },
    "eraser": {
        "rot_offset": R.from_euler("xyz", [np.pi / 2, 0, 0])
        .as_matrix()
        .astype(np.float32),
        "contact_offset_vec": np.zeros(3, dtype=np.float32),
        # np.array([0.065, 0.0, 0.01], dtype=np.float32),
    },
}

TOOL_OFFSETS_BY_VARIANT = {
    "toddlerbot": TODDY_TOOL_OFFSETS,
    "leap": LEAP_TOOL_OFFSETS,
}


FREESPACE_POS_STIFFNESS: npt.NDArray[np.float32] = np.diag(
    np.array([400.0, 400.0, 400.0], dtype=np.float32)
)
FREESPACE_ROT_STIFFNESS: npt.NDArray[np.float32] = np.diag(
    np.array([20.0, 20.0, 20.0], dtype=np.float32)
)


def as_homogeneous(
    position_world: np.ndarray, quaternion_wxyz: np.ndarray
) -> npt.NDArray[np.float32]:
    transform = np.eye(4, dtype=np.float32)
    rotation = (
        R.from_quat(quaternion_wxyz, scalar_first=True).as_matrix().astype(np.float32)
    )
    transform[:3, :3] = rotation
    transform[:3, 3] = np.asarray(position_world, dtype=np.float32)
    return transform


def get_tool_offsets(
    tool: str, robot_name: str, site_names: List[str]
) -> Dict[str, Union[np.ndarray, float]]:
    """Return rotation/contact offsets for the requested tool/variant."""
    variant_cfg = TOOL_OFFSETS_BY_VARIANT.get(robot_name, {})
    base_tool_cfg = variant_cfg.get(tool, {})

    rot_offsets = {}
    contact_vecs = {}
    for site_name in site_names:
        # Create a site-specific config copy
        tool_cfg = copy.deepcopy(base_tool_cfg)
        rot_by_site = tool_cfg.get("rot_offset_by_site", {})
        if site_name in rot_by_site:
            raw_rot = rot_by_site[site_name]
        else:
            raw_rot = tool_cfg.get("rot_offset", np.eye(3, dtype=np.float32))

        rot_offsets[site_name] = np.asarray(raw_rot, dtype=np.float32)
        contact_vecs[site_name] = tool_cfg.get("contact_offset_vec")

    return rot_offsets, contact_vecs


def estimate_camera_pose(
    head_position_world: np.ndarray,
    head_quaternion_wxyz: np.ndarray,
    camera_extrinsics: Dict[str, npt.NDArray[np.float32]],
    side: str = "left",
) -> npt.NDArray[np.float32]:
    camera_to_head = camera_extrinsics.get(side)
    if camera_to_head is None:
        raise ValueError(f"No camera extrinsics provided for side '{side}'.")
    return as_homogeneous(head_position_world, head_quaternion_wxyz) @ camera_to_head


def transform_points(
    points_camera: np.ndarray, world_T_camera: npt.NDArray[np.float32]
) -> npt.NDArray[np.float32]:
    points = np.asarray(points_camera, dtype=np.float32)
    if points.size == 0:
        return points.reshape(0, 3)
    homogeneous = np.concatenate(
        [points, np.ones((points.shape[0], 1), dtype=np.float32)], axis=1
    )
    world_pts = (world_T_camera @ homogeneous.T).T[:, :3]
    return world_pts.astype(np.float32)


def transform_normals(
    normals_camera: Optional[np.ndarray], world_T_camera: npt.NDArray[np.float32]
) -> npt.NDArray[np.float32]:
    if normals_camera is None:
        return np.empty((0, 3), dtype=np.float32)
    normals = np.asarray(normals_camera, dtype=np.float32)
    if normals.size == 0:
        return normals.reshape(0, 3)
    rotated = (world_T_camera[:3, :3] @ normals.T).T
    norms = np.linalg.norm(rotated, axis=1, keepdims=True) + 1e-9
    return (rotated / norms).astype(np.float32)


def normals_to_orientations(
    normals: np.ndarray, rot_offset: npt.NDArray[np.float32]
) -> npt.NDArray[np.float32]:
    normals_arr = np.asarray(normals, dtype=np.float32)
    if normals_arr.size == 0:
        return normals_arr.reshape(0, 3)
    if normals_arr.ndim == 1:
        normals_arr = normals_arr.reshape(1, 3)

    x_axis = -normals_arr
    z_axis = np.tile(
        np.array([0.0, 0.0, 1.0], dtype=np.float32), (normals_arr.shape[0], 1)
    )
    dot = np.einsum("ij,ij->i", x_axis, z_axis)
    near_parallel = np.abs(dot) > 0.99
    z_axis[near_parallel] = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    y_axis = np.cross(z_axis, x_axis)
    y_norms = np.linalg.norm(y_axis, axis=1, keepdims=True)
    valid_orientation = y_norms[:, 0] > 1e-6
    y_axis[valid_orientation] /= y_norms[valid_orientation]
    z_axis[valid_orientation] = np.cross(
        x_axis[valid_orientation], y_axis[valid_orientation]
    )

    rot_mats = np.stack([x_axis, y_axis, z_axis], axis=2) @ rot_offset

    orientations = np.zeros((normals_arr.shape[0], 3), dtype=np.float32)
    if np.any(valid_orientation):
        orientations[valid_orientation] = (
            R.from_matrix(rot_mats[valid_orientation]).as_rotvec().astype(np.float32)
        )

    return orientations


def apply_tool_contact_offset(
    contact_pos: np.ndarray,
    contact_ori: np.ndarray,
    contact_offset: Optional[np.ndarray],
) -> npt.NDArray[np.float32]:
    """
    Shift contact positions by the tool-specific offset so that the commanded
    waypoints correspond to the actual tip of the end-effector.
    """
    if contact_offset is None:
        return contact_pos

    offset_vec = -np.asarray(contact_offset, dtype=np.float32)
    rot_mats = R.from_rotvec(contact_ori).as_matrix().astype(np.float32)
    offset_world = (rot_mats @ offset_vec.reshape(3, 1)).reshape(-1, 3)
    return contact_pos + offset_world


def build_dense_trajectory(
    ee_pos_arr: np.ndarray, dt: float, segment_speed_limits: np.ndarray
) -> Dict[str, Union[np.ndarray, float]]:
    points = np.asarray(ee_pos_arr, dtype=np.float32)

    def static_profile(path_len: float) -> Dict[str, Union[np.ndarray, float]]:
        zero_vec = np.zeros((1, 3), dtype=np.float32)
        zero_scalar = np.array([0.0], dtype=np.float32)
        return {
            "t": zero_scalar,
            "pos": points[:1].copy(),
            "vel": zero_vec,
            "acc": zero_vec,
            "s": zero_scalar,
            "length": float(path_len),
            "v_profile": zero_scalar,
        }

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("ee_pos_arr must have shape (N, 3)")
    if points.shape[0] == 0:
        raise ValueError("ee_pos_arr must contain at least one waypoint")
    if dt <= 0.0:
        raise ValueError("dt must be positive")

    if points.shape[0] == 1:
        return static_profile(0.0)

    seg_speed_array = np.asarray(segment_speed_limits, dtype=np.float32).reshape(-1)
    expected_segments = points.shape[0] - 1
    if seg_speed_array.shape[0] != expected_segments:
        raise ValueError(
            "segment_speed_limits must have length equal to len(ee_pos_arr) - 1"
        )

    diffs = np.diff(points, axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1)

    keep_segments = seg_lengths > 1e-8
    keep = np.ones(points.shape[0], dtype=bool)
    keep[1:] = keep_segments
    if not np.all(keep):
        points = points[keep]
        seg_speed_array = seg_speed_array[keep_segments]
        if points.shape[0] == 1:
            return static_profile(0.0)
        diffs = np.diff(points, axis=0)
        seg_lengths = np.linalg.norm(diffs, axis=1)

    num_segments = seg_lengths.shape[0]
    seg_speeds = np.clip(seg_speed_array, 1e-5, None)
    ref_speed = float(np.max(seg_speeds))

    arc_lengths = np.concatenate(([0.0], np.cumsum(seg_lengths)))
    path_length = float(arc_lengths[-1])
    if path_length <= 1e-9:
        return static_profile(path_length)

    eff_seg_lengths = seg_lengths * (ref_speed / seg_speeds)
    eff_arc_lengths = np.concatenate(([0.0], np.cumsum(eff_seg_lengths)))
    eff_path_length = float(eff_arc_lengths[-1])
    path_time = float(np.sum(seg_lengths / seg_speeds))
    if eff_path_length <= 1e-9 or path_time <= 1e-9:
        return static_profile(path_length)

    # Trapezoidal sigma(t): linear accel to peak speed, cruise, linear decel.
    time_stretch = 1.1
    min_time = path_time * time_stretch
    total_time = max(min_time, dt)
    t_samples = np.arange(0.0, total_time, dt, dtype=np.float32)
    if t_samples.size == 0 or t_samples[-1] < total_time:
        t_samples = np.append(t_samples, total_time)
    num_samples = t_samples.shape[0]

    sigma_samples = np.empty(num_samples, dtype=np.float32)
    dsigma_dt = np.empty(num_samples, dtype=np.float32)
    d2sigma_dt2 = np.empty(num_samples, dtype=np.float32)

    max_t_acc = max(total_time - path_time, 0.0)
    nominal_t_acc = 0.2 * total_time
    t_acc = float(min(nominal_t_acc, max_t_acc, total_time * 0.5))
    t_flat = max(total_time - 2.0 * t_acc, 0.0)

    if t_acc <= 1e-9:
        v_peak = eff_path_length / max(total_time, 1e-9)
        sigma_samples[:] = v_peak * t_samples
        dsigma_dt[:] = v_peak
        d2sigma_dt2[:] = 0.0
    else:
        v_peak = eff_path_length / max(total_time - t_acc, 1e-9)
        a = v_peak / t_acc
        for idx, t_val in enumerate(t_samples):
            if t_val < t_acc:
                sigma_samples[idx] = 0.5 * a * t_val * t_val
                dsigma_dt[idx] = a * t_val
                d2sigma_dt2[idx] = a
            elif t_val < t_acc + t_flat:
                sigma_samples[idx] = 0.5 * a * t_acc * t_acc + v_peak * (t_val - t_acc)
                dsigma_dt[idx] = v_peak
                d2sigma_dt2[idx] = 0.0
            else:
                t_dec = t_val - (t_acc + t_flat)
                sigma_samples[idx] = (
                    0.5 * a * t_acc * t_acc
                    + v_peak * t_flat
                    + v_peak * t_dec
                    - 0.5 * a * t_dec * t_dec
                )
                dsigma_dt[idx] = max(v_peak - a * t_dec, 0.0)
                d2sigma_dt2[idx] = -a

    sigma_samples = np.clip(sigma_samples, 0.0, eff_path_length)
    dsigma_dt = np.maximum(dsigma_dt, 0.0)
    sigma_samples[0] = 0.0
    sigma_samples[-1] = eff_path_length
    dsigma_dt[0] = 0.0
    dsigma_dt[-1] = 0.0
    d2sigma_dt2[-1] = 0.0

    s_samples = np.interp(sigma_samples, eff_arc_lengths, arc_lengths).astype(
        np.float32
    )
    if path_length > 0.0:
        s_samples[0] = 0.0
        s_samples[-1] = path_length

    seg_speed_ratios = seg_speeds / ref_speed
    if num_segments > 0:
        seg_idx_sigma = (
            np.searchsorted(eff_arc_lengths, sigma_samples, side="right") - 1
        )
        seg_idx_sigma = np.clip(seg_idx_sigma, 0, num_segments - 1)
        ds_dsigma = seg_speed_ratios[seg_idx_sigma]
    else:
        ds_dsigma = np.zeros_like(sigma_samples)
    ds_dt = ds_dsigma * dsigma_dt
    d2s_dt2 = ds_dsigma * d2sigma_dt2

    tangents = np.zeros_like(points)
    tangents[0] = (points[1] - points[0]) / max(seg_lengths[0], 1e-8)
    tangents[-1] = (points[-1] - points[-2]) / max(seg_lengths[-1], 1e-8)
    for idx in range(1, points.shape[0] - 1):
        denom = arc_lengths[idx + 1] - arc_lengths[idx - 1]
        if denom <= 1e-8:
            denom = max(seg_lengths[idx], 1e-8)
            tangents[idx] = (points[idx + 1] - points[idx]) / denom
        else:
            tangents[idx] = (points[idx + 1] - points[idx - 1]) / denom

    pos_samples = np.zeros((num_samples, 3), dtype=np.float32)
    dp_ds_samples = np.zeros_like(pos_samples)
    d2p_ds2_samples = np.zeros_like(pos_samples)

    for i, s_val in enumerate(s_samples):
        if s_val >= path_length:
            pos_samples[i] = points[-1]
            dp_ds_samples[i] = tangents[-1]
            d2p_ds2_samples[i] = 0.0
            continue

        seg_idx = np.searchsorted(arc_lengths, s_val, side="right") - 1
        seg_idx = int(np.clip(seg_idx, 0, points.shape[0] - 2))
        s_i = arc_lengths[seg_idx]
        s_ip1 = arc_lengths[seg_idx + 1]
        ds_val = s_ip1 - s_i
        if ds_val <= 1e-10:
            pos_samples[i] = points[seg_idx]
            dp_ds_samples[i] = tangents[seg_idx]
            d2p_ds2_samples[i] = 0.0
            continue

        u = (s_val - s_i) / ds_val
        u2 = u * u
        u3 = u2 * u

        h00 = 2.0 * u3 - 3.0 * u2 + 1.0
        h10 = u3 - 2.0 * u2 + u
        h01 = -2.0 * u3 + 3.0 * u2
        h11 = u3 - u2

        pos_samples[i] = (
            h00 * points[seg_idx]
            + h10 * ds_val * tangents[seg_idx]
            + h01 * points[seg_idx + 1]
            + h11 * ds_val * tangents[seg_idx + 1]
        )

        inv_ds = 1.0 / ds_val
        inv_ds2 = inv_ds * inv_ds

        dh00 = (6.0 * u2 - 6.0 * u) * inv_ds
        dh10 = (3.0 * u2 - 4.0 * u + 1.0) * inv_ds
        dh01 = (-6.0 * u2 + 6.0 * u) * inv_ds
        dh11 = (3.0 * u2 - 2.0 * u) * inv_ds

        dp_ds_samples[i] = (
            dh00 * points[seg_idx]
            + dh10 * ds_val * tangents[seg_idx]
            + dh01 * points[seg_idx + 1]
            + dh11 * ds_val * tangents[seg_idx + 1]
        )

        d2h00 = (12.0 * u - 6.0) * inv_ds2
        d2h10 = (6.0 * u - 4.0) * inv_ds2
        d2h01 = (-12.0 * u + 6.0) * inv_ds2
        d2h11 = (6.0 * u - 2.0) * inv_ds2

        d2p_ds2_samples[i] = (
            d2h00 * points[seg_idx]
            + d2h10 * ds_val * tangents[seg_idx]
            + d2h01 * points[seg_idx + 1]
            + d2h11 * ds_val * tangents[seg_idx + 1]
        )

    vel_samples = dp_ds_samples * ds_dt[:, None]
    acc_samples = (
        d2p_ds2_samples * (ds_dt**2)[:, None] + dp_ds_samples * d2s_dt2[:, None]
    )

    return {
        "t": t_samples,
        "pos": pos_samples,
        "vel": vel_samples,
        "acc": acc_samples,
        "s": s_samples,
        "length": path_length,
        "v_profile": ds_dt,
    }


def plan_trajectory_profile(
    contact_points_camera: np.ndarray,
    contact_normals_camera: np.ndarray,
    world_T_left_camera: np.ndarray,
    rot_offsets: np.ndarray,
    contact_offsets: np.ndarray,
    tangent_pos_stiffness: float,
    normal_pos_stiffness: float,
    tangent_rot_stiffness: float,
    normal_rot_stiffness: float,
    contact_force: float | np.ndarray,
    pose_cur: np.ndarray,
    traj_dt: float,
    traj_v_max_contact: float,
    traj_v_max_free: float,
    pause_prepare: float,
    pause_contact: float,
    mass: float | np.ndarray,
    inertia_diag: float | np.ndarray,
):
    contact_pos = transform_points(contact_points_camera, world_T_left_camera)
    contact_normals = transform_normals(contact_normals_camera, world_T_left_camera)
    contact_ori = normals_to_orientations(contact_normals, rot_offsets)
    contact_pos = apply_tool_contact_offset(contact_pos, contact_ori, contact_offsets)

    start_pos = pose_cur[None, :3]
    start_ori = pose_cur[None, 3:]

    retreat_vecs = contact_normals * DEFAULT_CONTACT_PREP_DISTANCE
    prepare_before_pos = contact_pos[:1] + retreat_vecs[:1]
    prepare_after_pos = contact_pos[-1:] + retreat_vecs[-1:]
    prepare_before_ori = contact_ori[:1]
    prepare_after_ori = contact_ori[-1:]

    positions = np.concatenate(
        [start_pos, prepare_before_pos, contact_pos, prepare_after_pos, start_pos],
        axis=0,
    )
    orientations = np.concatenate(
        [start_ori, prepare_before_ori, contact_ori, prepare_after_ori, start_ori],
        axis=0,
    )

    pos_stiffness = (
        np.eye(3, dtype=np.float32)[None] * tangent_pos_stiffness
        + (normal_pos_stiffness - tangent_pos_stiffness)
        * contact_normals[:, :, None]
        * contact_normals[:, None, :]
    )
    rot_stiffness = (
        np.eye(3, dtype=np.float32)[None] * tangent_rot_stiffness
        + (normal_rot_stiffness - tangent_rot_stiffness)
        * contact_normals[:, :, None]
        * contact_normals[:, None, :]
    )
    command_forces = -contact_normals * contact_force

    pos_stiffness_padding = np.repeat(FREESPACE_POS_STIFFNESS[None], 2, axis=0)
    rot_stiffness_padding = np.repeat(FREESPACE_ROT_STIFFNESS[None], 2, axis=0)
    pos_stiffness = np.concatenate(
        [pos_stiffness_padding, pos_stiffness, pos_stiffness_padding],
        axis=0,
    )
    rot_stiffness = np.concatenate(
        [rot_stiffness_padding, rot_stiffness, rot_stiffness_padding],
        axis=0,
    )
    command_forces = np.concatenate(
        [
            np.zeros((2, 3), dtype=np.float32),
            command_forces,
            np.zeros((2, 3), dtype=np.float32),
        ],
        axis=0,
    )

    segment_specs: List[Tuple[int, int, float, float, int]] = []
    if contact_pos.shape[0] == 0:
        segment_specs.append((0, positions.shape[0], traj_v_max_free, 0.0, 1))
    else:
        last_contact_idx = positions.shape[0] - 3
        prepare_after_idx = positions.shape[0] - 2
        segment_specs.append((0, 2, traj_v_max_free, pause_prepare, 0))
        segment_specs.append((1, 3, traj_v_max_free, pause_contact, 1))
        segment_specs.append(
            (2, prepare_after_idx, traj_v_max_contact, pause_contact, 1)
        )
        segment_specs.append(
            (
                last_contact_idx,
                positions.shape[0] - 1,
                traj_v_max_contact,
                pause_prepare,
                2,
            )
        )
        segment_specs.append(
            (prepare_after_idx, positions.shape[0], traj_v_max_free, 0.0, 2)
        )

    def interpolate_segment(
        values: np.ndarray, s_samples: np.ndarray, arc_lengths: np.ndarray
    ) -> np.ndarray:
        subset = values
        flat = subset.reshape(subset.shape[0], -1).astype(np.float32)
        dense_flat = np.empty((s_samples.size, flat.shape[1]), dtype=np.float32)
        for col in range(flat.shape[1]):
            dense_flat[:, col] = np.interp(s_samples, arc_lengths, flat[:, col])
        return dense_flat.reshape((s_samples.size,) + subset.shape[1:]).astype(
            np.float32
        )

    def densify_segment(start_idx: int, end_idx: int, speed: float):
        if end_idx - start_idx < 2:
            return None

        pos_seg = positions[start_idx:end_idx]
        ori_seg = orientations[start_idx:end_idx]
        pos_stiff_seg = pos_stiffness[start_idx:end_idx]
        rot_stiff_seg = rot_stiffness[start_idx:end_idx]
        force_seg = command_forces[start_idx:end_idx]

        speed_limits = np.full(pos_seg.shape[0] - 1, speed, dtype=np.float32)
        profile = build_dense_trajectory(
            pos_seg,
            dt=traj_dt,
            segment_speed_limits=speed_limits,
        )

        arc_lengths_seg = np.concatenate(
            ([0.0], np.cumsum(np.linalg.norm(np.diff(pos_seg, axis=0), axis=1)))
        ).astype(np.float32)
        s_samples = np.asarray(profile["s"], dtype=np.float32)
        num_samples = s_samples.size

        if arc_lengths_seg[-1] <= 1e-9:
            dense_orient = np.repeat(ori_seg[:1], num_samples, axis=0)
            dense_pos_stiff = np.repeat(pos_stiff_seg[:1], num_samples, axis=0)
            dense_rot_stiff = np.repeat(rot_stiff_seg[:1], num_samples, axis=0)
            dense_forces = np.repeat(force_seg[:1], num_samples, axis=0)
        else:
            slerp = Slerp(arc_lengths_seg, R.from_rotvec(ori_seg))
            dense_orient = (
                slerp(np.clip(s_samples, arc_lengths_seg[0], arc_lengths_seg[-1]))
                .as_rotvec()
                .astype(np.float32)
            )
            dense_pos_stiff = interpolate_segment(
                pos_stiff_seg, s_samples, arc_lengths_seg
            )
            dense_rot_stiff = interpolate_segment(
                rot_stiff_seg, s_samples, arc_lengths_seg
            )
            dense_forces = interpolate_segment(force_seg, s_samples, arc_lengths_seg)

        return {
            "profile": profile,
            "orient": dense_orient,
            "pos_stiff": dense_pos_stiff,
            "rot_stiff": dense_rot_stiff,
            "forces": dense_forces,
        }

    combined = {
        "t": [],
        "stage": [],
        "pos": [],
        "vel": [],
        "acc": [],
        "s": [],
        "v_profile": [],
        "orient": [],
        "pos_stiff": [],
        "rot_stiff": [],
        "forces": [],
    }
    time_offset = 0.0
    s_offset = 0.0
    first_segment = True

    def append_pause(
        duration: float,
        last_pos: np.ndarray,
        last_orient: np.ndarray,
        last_pos_stiff: np.ndarray,
        last_rot_stiff: np.ndarray,
        last_force: np.ndarray,
        stage_value: int,
    ):
        nonlocal time_offset
        if duration <= 0.0:
            return
        dwell_times = np.arange(traj_dt, duration + traj_dt, traj_dt, dtype=np.float32)
        num = dwell_times.shape[0]
        if num == 0:
            return
        t_seg = time_offset + dwell_times
        combined["t"].append(t_seg)
        combined["stage"].append(np.full(num, stage_value, dtype=np.int32))
        combined["pos"].append(np.repeat(last_pos[None], num, axis=0))
        combined["vel"].append(np.zeros((num, 3), dtype=np.float32))
        combined["acc"].append(np.zeros((num, 3), dtype=np.float32))
        combined["s"].append(np.full(num, s_offset, dtype=np.float32))
        combined["v_profile"].append(np.zeros(num, dtype=np.float32))
        combined["orient"].append(np.repeat(last_orient[None], num, axis=0))
        combined["pos_stiff"].append(np.repeat(last_pos_stiff[None], num, axis=0))
        combined["rot_stiff"].append(np.repeat(last_rot_stiff[None], num, axis=0))
        combined["forces"].append(np.repeat(last_force[None], num, axis=0))
        time_offset = t_seg[-1]

    for seg_start, seg_end, seg_speed, pause_after, stage_value in segment_specs:
        seg_result = densify_segment(seg_start, seg_end, seg_speed)
        if seg_result is None:
            continue
        profile = seg_result["profile"]
        start_idx = 0 if first_segment else 1
        if profile["pos"].shape[0] <= start_idx:
            continue

        profile_pos = np.asarray(profile["pos"], dtype=np.float32)
        profile_vel = np.asarray(profile["vel"], dtype=np.float32)
        profile_acc = np.asarray(profile["acc"], dtype=np.float32)
        profile_time = np.asarray(profile["t"], dtype=np.float32)
        profile_s = np.asarray(profile["s"], dtype=np.float32)
        profile_v = np.asarray(profile["v_profile"], dtype=np.float32)

        time_segment = profile_time + time_offset
        s_segment = profile_s + s_offset
        segment_length = time_segment.shape[0] - start_idx
        if segment_length <= 0:
            continue

        combined["t"].append(time_segment[start_idx:])
        combined["stage"].append(np.full(segment_length, stage_value, dtype=np.int32))
        combined["pos"].append(profile_pos[start_idx:])
        combined["vel"].append(profile_vel[start_idx:])
        combined["acc"].append(profile_acc[start_idx:])
        combined["s"].append(s_segment[start_idx:])
        combined["v_profile"].append(profile_v[start_idx:])
        combined["orient"].append(seg_result["orient"][start_idx:])
        combined["pos_stiff"].append(seg_result["pos_stiff"][start_idx:])
        combined["rot_stiff"].append(seg_result["rot_stiff"][start_idx:])
        combined["forces"].append(seg_result["forces"][start_idx:])

        time_offset = time_segment[-1]
        s_offset = s_segment[-1]
        first_segment = False
        append_pause(
            pause_after,
            profile_pos[-1],
            seg_result["orient"][-1],
            seg_result["pos_stiff"][-1],
            seg_result["rot_stiff"][-1],
            seg_result["forces"][-1],
            stage_value,
        )

    if not combined["t"]:
        combined["t"] = [np.array([0.0], dtype=np.float32)]
        combined["stage"] = [np.array([0], dtype=np.int32)]
        combined["pos"] = [positions[:1].astype(np.float32)]
        combined["vel"] = [np.zeros((1, 3), dtype=np.float32)]
        combined["acc"] = [np.zeros((1, 3), dtype=np.float32)]
        combined["s"] = [np.array([0.0], dtype=np.float32)]
        combined["v_profile"] = [np.array([0.0], dtype=np.float32)]
        combined["orient"] = [orientations[:1].astype(np.float32)]
        combined["pos_stiff"] = [pos_stiffness[:1].astype(np.float32)]
        combined["rot_stiff"] = [rot_stiffness[:1].astype(np.float32)]
        combined["forces"] = [command_forces[:1].astype(np.float32)]
        s_offset = 0.0

    traj_profile = {
        "t": np.concatenate(combined["t"], axis=0),
        "stage": np.concatenate(combined["stage"], axis=0),
        "pos": np.concatenate(combined["pos"], axis=0),
        "vel": np.concatenate(combined["vel"], axis=0),
        "acc": np.concatenate(combined["acc"], axis=0),
        "s": np.concatenate(combined["s"], axis=0),
        "length": float(s_offset),
        "v_profile": np.concatenate(combined["v_profile"], axis=0),
    }
    dense_positions = traj_profile["pos"]
    dense_orientations = np.concatenate(combined["orient"], axis=0)
    dense_pos_stiffness = np.concatenate(combined["pos_stiff"], axis=0)
    dense_rot_stiffness = np.concatenate(combined["rot_stiff"], axis=0)
    dense_command_forces = np.concatenate(combined["forces"], axis=0)
    mass_matrix = ensure_matrix(mass)
    inertia_matrix = ensure_matrix(inertia_diag)
    dense_pos_damping = np.empty_like(dense_pos_stiffness, dtype=np.float32)
    dense_rot_damping = np.empty_like(dense_rot_stiffness, dtype=np.float32)
    for idx in range(dense_pos_stiffness.shape[0]):
        dense_pos_damping[idx] = get_damping_matrix(
            dense_pos_stiffness[idx], mass_matrix
        )
        dense_rot_damping[idx] = get_damping_matrix(
            dense_rot_stiffness[idx], inertia_matrix
        )

    return (
        traj_profile["t"],
        traj_profile["stage"],
        dense_positions,
        dense_orientations,
        dense_pos_stiffness,
        dense_rot_stiffness,
        dense_pos_damping,
        dense_rot_damping,
        dense_command_forces,
    )


def plan_end_effector_poses(
    contact_points_camera: Dict[str, np.ndarray],
    contact_normals_camera: Dict[str, np.ndarray],
    head_position_world: np.ndarray,
    head_quaternion_world_wxyz: np.ndarray,
    tangent_pos_stiffness: float,
    normal_pos_stiffness: float,
    tangent_rot_stiffness: float,
    normal_rot_stiffness: float,
    contact_force: float | np.ndarray,
    pose_cur: Dict[str, np.ndarray],
    output_dir: Optional[str] = None,
    traj_dt: float = 0.02,
    traj_v_max_contact: float = 0.04,
    traj_v_max_free: float = 0.1,
    pause_prepare: float = 2.0,
    pause_contact: float = 0.2,
    tool: str = "eraser",
    robot_name: str = "toddlerbot",
    task: Optional[str] = None,
    mass: float | np.ndarray = 1.0,
    inertia_diag: float | np.ndarray = (1.0, 1.0, 1.0),
):
    site_names = list(contact_points_camera.keys())
    contact_force_by_site: Optional[Dict[str, float]] = None
    if isinstance(contact_force, np.ndarray):
        force_values = contact_force.reshape(-1).astype(np.float32)
        if force_values.size == 1:
            contact_force = float(force_values[0])
        elif len(site_names) > 1 and force_values.size == len(site_names):
            contact_force_by_site = {
                site_name: float(force_values[idx])
                for idx, site_name in enumerate(site_names)
            }
        else:
            raise ValueError(
                "contact_force must be a scalar or have one value per site."
            )
    rot_offsets, contact_offsets = get_tool_offsets(tool, robot_name, site_names)

    if task is None:
        if tool == "eraser":
            task = "wipe"
        elif tool == "pen":
            task = "draw"

    camera_extrinsics = (
        LEAP_CAMERA_EXTRINSICS if "leap" in robot_name else TODDY_CAMERA_EXTRINSICS
    )
    world_T_left_camera = estimate_camera_pose(
        head_position_world,
        head_quaternion_world_wxyz,
        camera_extrinsics,
    )
    trajectory_by_site = {}
    for site_name in site_names:
        if contact_force_by_site is not None:
            if site_name not in contact_force_by_site:
                raise ValueError(f"Missing contact force for site '{site_name}'.")
            site_contact_force = contact_force_by_site[site_name]
        else:
            site_contact_force = contact_force
        trajectory_by_site[site_name] = plan_trajectory_profile(
            contact_points_camera[site_name],
            contact_normals_camera[site_name],
            world_T_left_camera,
            rot_offsets[site_name],
            contact_offsets[site_name],
            tangent_pos_stiffness=tangent_pos_stiffness,
            normal_pos_stiffness=normal_pos_stiffness,
            tangent_rot_stiffness=tangent_rot_stiffness,
            normal_rot_stiffness=normal_rot_stiffness,
            contact_force=site_contact_force,
            pose_cur=pose_cur[site_name],
            traj_dt=traj_dt,
            traj_v_max_contact=traj_v_max_contact,
            traj_v_max_free=traj_v_max_free,
            pause_prepare=pause_prepare,
            pause_contact=pause_contact,
            mass=mass,
            inertia_diag=inertia_diag,
        )

    if output_dir is not None:
        world_T_right_camera = estimate_camera_pose(
            head_position_world,
            head_quaternion_world_wxyz,
            side="right",
            camera_extrinsics=camera_extrinsics,
        )
        os.makedirs(output_dir, exist_ok=True)
        data_path = os.path.join(output_dir, "trajectory.lz4")
        payload = {
            "world_T_left_camera": world_T_left_camera,
            "world_T_right_camera": world_T_right_camera,
            "contact_pos_camera": contact_points_camera,
            "contact_normals_camera": contact_normals_camera,
            "trajectory_by_site": trajectory_by_site,
            "task": task,
        }
        joblib.dump(payload, data_path, compress="lz4")
        print(f"Saved the end effector trajectory data to {data_path}")

    return trajectory_by_site
