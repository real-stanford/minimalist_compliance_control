from __future__ import annotations

import os
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Dict, Optional, Tuple

import joblib
import mujoco
import numpy as np
import numpy.typing as npt
import yaml
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

from hybrid_servo.algorithm.ochs import solve_ochs
from hybrid_servo.algorithm.solvehfvc import HFVC, transform_hfvc_to_global
from hybrid_servo.tasks.bimanual_ochs import compute_center_quaternion_from_hands
from hybrid_servo.tasks.multi_finger_ochs import (
    compute_hfvc_inputs,
    generate_constraint_jacobian,
    get_center_state,
)
from minimalist_compliance_control.compliance_ref import COMMAND_LAYOUT, ComplianceState
from minimalist_compliance_control.controller import (
    ComplianceController,
    ComplianceRefConfig,
    ControllerConfig,
)
from minimalist_compliance_control.utils import KeyboardControlReceiver
from minimalist_compliance_control.wrench_estimation import WrenchEstimateConfig

_tb_Array = npt.NDArray[np.float64]


def symmetrize(matrix: npt.ArrayLike) -> npt.NDArray[np.float32]:
    arr = np.asarray(matrix, dtype=np.float32)
    return (0.5 * (arr + np.swapaxes(arr, -1, -2))).astype(np.float32)


def matrix_sqrt(matrix: npt.ArrayLike) -> npt.NDArray[np.float32]:
    sym = symmetrize(matrix)
    eigvals, eigvecs = np.linalg.eigh(sym)
    eigvals_clipped = np.clip(eigvals, 0.0, None)
    sqrt_vals = np.sqrt(eigvals_clipped)[..., None, :]
    scaled_vecs = eigvecs * sqrt_vals
    sqrt_matrix = np.matmul(scaled_vecs, np.swapaxes(eigvecs, -1, -2))
    return symmetrize(sqrt_matrix)


def ensure_matrix(value: float | npt.ArrayLike) -> npt.NDArray[np.float32]:
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim == 0:
        return np.eye(3, dtype=np.float32) * float(arr)
    if arr.ndim == 1:
        if arr.shape[0] != 3:
            raise ValueError("Gain vectors must have length 3.")
        return np.diag(arr.astype(np.float32))
    if arr.ndim >= 2:
        if arr.shape[-2:] != (3, 3):
            raise ValueError("Gain matrices must have trailing shape (3, 3).")
        return arr.astype(np.float32)
    raise ValueError("Unsupported gain array shape.")


def get_damping_matrix(
    stiffness: float | npt.ArrayLike,
    inertia_like: float | npt.ArrayLike,
) -> npt.NDArray[np.float32]:
    stiffness_matrix = ensure_matrix(stiffness)
    inertia_matrix = ensure_matrix(inertia_like)
    mass_sqrt = matrix_sqrt(inertia_matrix)
    stiffness_sqrt = matrix_sqrt(stiffness_matrix)
    damping = 2.0 * np.matmul(mass_sqrt, stiffness_sqrt)
    return symmetrize(damping).astype(np.float32)


PREPARE_POS = [
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    -1.57,
    0.0,
    -1.2,
    0.0,
]

# Fingertip site/geom names used for contact checks.
LEAP_FINGER_TIPS = ("if_tip", "mf_tip", "th_tip")

OBJECT_MASS_MAP = {
    "unknown": {
        "mass": 0.05,
        "init_pos": None,
        "init_quat": None,
        "min_normal_force_rotation": 6.0,
        "min_normal_force_translation": 9.0,
        "geom_size": np.array([0.04], dtype=np.float32),  # Default radius for sphere
    },
    "sphere": {
        "mass": 0.06,
        "init_pos": np.array([-0.12, -0.075, 0.16], dtype=np.float32),
        "init_quat": np.array([0.70710677, 0.70710677, 0.0, 0.0], dtype=np.float32),
        "min_normal_force_rotation": 10.0,
        "min_normal_force_translation": 10.0,
        "geom_size": np.array([0.0343], dtype=np.float32),  # Radius
    },
    "box": {
        "mass": 0.1,
        "init_pos": np.array([-0.115, -0.075, 0.164], dtype=np.float32),
        "init_quat": np.array([0.70710677, 0.70710677, 0.0, 0.0], dtype=np.float32),
        "min_normal_force_rotation": 5,
        "min_normal_force_translation": 5.0,
        "geom_size": np.array(
            [0.03, 0.03, 0.03], dtype=np.float32
        ),  # [half_x, half_y, half_z]
    },
    "cylinder_short": {
        "mass": 0.1,
        "init_pos": np.array([-0.13, -0.08, 0.145], dtype=np.float32),
        "init_quat": np.array([0.70710677, 0.70710677, 0.0, 0.0], dtype=np.float32),
        "min_normal_force_rotation": 8.0,
        "min_normal_force_translation": 5.0,
        "geom_size": np.array([0.04, 0.12], dtype=np.float32),  # [radius, half_height]
    },
    "pen": {
        "mass": 0.05,
        "init_pos": np.array([-0.13, -0.08, 0.14], dtype=np.float32),
        "init_quat": np.array([0.70710677, 0.70710677, 0.0, 0.0], dtype=np.float32),
        "min_normal_force_rotation": 6.0,
        "min_normal_force_translation": 9.0,
        "geom_size": np.array(
            [0.015, 0.08], dtype=np.float32
        ),  # [radius, half_height] for cylinder-like pen
    },
}

INIT_POSE_DATA = {
    "if_tip": {
        "pos": np.array([0.052, -0.1, 0.247], dtype=np.float32),
        "ori": np.array([-3.14, 0, 0], dtype=np.float32),
    },
    "mf_tip": {
        "pos": np.array([0.052, -0.055, 0.247], dtype=np.float32),
        "ori": np.array([-3.14, 0, 0], dtype=np.float32),
    },
    "rf_tip": {
        "pos": np.array([0.052, -0.01, 0.247], dtype=np.float32),
        "ori": np.array([-3.14, 0, 0], dtype=np.float32),
    },
    "th_tip": {
        "pos": np.array([-0.228, -0.094, 0.149], dtype=np.float32),
        "ori": np.array([-0.07, 2.42, 0.02], dtype=np.float32),
    },
}
TARGET_POSE_DATA: Dict[str, Dict[str, np.ndarray]] = {
    "if_tip": {
        "pos": np.array([-0.101, -0.099, 0.152], dtype=np.float32),
        "ori": np.array([-1.40, -0.05, 2.81], dtype=np.float32),
    },
    "mf_tip": {
        "pos": np.array([-0.101, -0.056, 0.152], dtype=np.float32),
        "ori": np.array([-1.40, 0.0, 2.81], dtype=np.float32),
    },
    "rf_tip": {
        "pos": np.array([0.042, -0.01, 0.247], dtype=np.float32),
        "ori": np.array([-3.14, 0.0, 0.0], dtype=np.float32),
    },
    "th_tip": {
        "pos": np.array([-0.145, -0.085, 0.148], dtype=np.float32),
        "ori": np.array([0.04, 1.02, 0.03], dtype=np.float32),
    },
}

###TARGET_POSE_DATA for pen rotating
# TARGET_POSE_DATA: Dict[str, Dict[str, np.ndarray]] = {
#     "if_tip": {
#         "pos": np.array([-0.121, -0.101, 0.14], dtype=np.float32),
#         "ori": np.array([1.86, -0.0, -2.53], dtype=np.float32),
#     },
#     "mf_tip": {
#         "pos": np.array([-0.121, -0.056, 0.14], dtype=np.float32),
#         "ori": np.array([1.86, 0.0, -2.53], dtype=np.float32),
#     },
#     "rf_tip": {
#         "pos": np.array([0.042, -0.01, 0.247], dtype=np.float32),
#         "ori": np.array([-3.14, -0.0, 0.0], dtype=np.float32),
#     },
#     "th_tip": {
#         "pos": np.array([-0.141, -0.077, 0.139], dtype=np.float32),
#         "ori": np.array([0.08, 1.23, 0.09], dtype=np.float32),
#     },
# }

OBJECT_INIT_POS_MAP = {
    "sphere": np.array([-0.125, -0.08, 0.145], dtype=np.float32),
    "box": np.array([-0.125, -0.08, 0.15], dtype=np.float32),
    "cylinder_short": np.array([-0.125, -0.075, 0.16], dtype=np.float32),
}

OBJECT_TYPE = "cylinder_short"
# "box"
# "sphere"
# "cylinder_short"


def _leap_init(
    policy,
    wrench_sim: Any,
    wrench_site_names: Tuple[str, ...] = LEAP_FINGER_TIPS,
    control_dt: float = 0.02,
    prep_duration: float = 0.0,
    auto_switch_target_enabled: bool = True,
):
    policy.wrench_sim = wrench_sim
    policy.wrench_site_names = list(wrench_site_names)
    policy.num_sites = len(policy.wrench_site_names)
    policy.control_dt = float(control_dt)
    policy.prep_duration = float(prep_duration)
    policy.wrenches_by_site: Dict[str, np.ndarray] = {}
    policy.wrench_command = np.zeros((policy.num_sites, 6), dtype=np.float32)
    policy.pos_stiffness = np.zeros((policy.num_sites, 9), dtype=np.float32)
    policy.rot_stiffness = np.zeros((policy.num_sites, 9), dtype=np.float32)
    policy.pos_damping = np.zeros((policy.num_sites, 9), dtype=np.float32)
    policy.rot_damping = np.zeros((policy.num_sites, 9), dtype=np.float32)

    policy.object_type = OBJECT_TYPE
    policy.object_type_detected = False
    policy.object_mass = 0.05
    policy.object_geom_size = np.array([0.04], dtype=np.float32)  # Default size

    policy.use_compliance = True
    policy.log_ik = True
    policy.pd_updated = False
    policy.desired_kp = 1500  # 450
    policy.desired_kd = 0

    policy.contact_force = 0.0
    policy.normal_pos_stiffness = 10.0
    policy.tangent_pos_stiffness = 100.0
    policy.normal_rot_stiffness = 10.0
    policy.tangent_rot_stiffness = 20.0

    policy.ref_motor_pos = np.array(PREPARE_POS, dtype=np.float32)
    policy.initial_pose_command = _leap_build_pose_command(policy, INIT_POSE_DATA)
    policy.target_pose_command = _leap_build_pose_command(policy, TARGET_POSE_DATA)
    policy.pose_command = policy.initial_pose_command.copy()
    policy.integrated_angle_thumb = np.zeros(3, dtype=np.float32)
    policy.pose_interp_pos_speed = 0.1
    policy.pose_interp_rot_speed = 1.0
    policy.pose_interp_min_duration = 0.2
    policy.pose_interp_max_duration = 2.0
    mass_matrix = ensure_matrix(1.0)
    inertia_matrix = ensure_matrix([1.0, 1.0, 1.0])

    open_pos_stiff = ensure_matrix([400.0, 400.0, 400.0])
    open_rot_stiff = ensure_matrix([20.0, 20.0, 20.0])
    open_pos_stiff_arr = np.broadcast_to(
        open_pos_stiff, (policy.num_sites, 3, 3)
    ).astype(np.float32)
    open_rot_stiff_arr = np.broadcast_to(
        open_rot_stiff, (policy.num_sites, 3, 3)
    ).astype(np.float32)
    open_pos_damp = np.stack(
        [
            get_damping_matrix(open_pos_stiff, mass_matrix)
            for _ in range(policy.num_sites)
        ],
        axis=0,
    ).astype(np.float32)
    open_rot_damp = np.stack(
        [
            get_damping_matrix(open_rot_stiff, inertia_matrix)
            for _ in range(policy.num_sites)
        ],
        axis=0,
    ).astype(np.float32)
    open_wrench = np.zeros((policy.num_sites, 6), dtype=np.float32)
    policy.open_gains = {
        "pos_stiff": open_pos_stiff_arr,
        "rot_stiff": open_rot_stiff_arr,
        "pos_damp": open_pos_damp,
        "rot_damp": open_rot_damp,
        "wrench": open_wrench,
    }
    policy.close_gains = _leap_compute_force_and_stiffness(
        policy, policy.target_pose_command
    )

    policy.forward_traj = _leap_build_command_trajectory(
        policy,
        policy.initial_pose_command,
        policy.target_pose_command,
        policy.open_gains,
        policy.close_gains,
    )
    policy.backward_traj = _leap_build_command_trajectory(
        policy,
        policy.target_pose_command,
        policy.initial_pose_command,
        policy.close_gains,
        policy.open_gains,
    )

    policy.active_traj: Optional[Dict[str, np.ndarray]] = None
    policy.traj_start_time = 0.0

    # Initialize stiffness/wrench targets from first trajectory sample.
    _leap_apply_traj_sample(policy, policy.forward_traj, 0)

    policy.phase = "close"
    policy.traj_set = False
    policy.object_body_name: str = "manip_object"
    policy.object_qpos_adr: Optional[int] = None
    policy.object_qvel_adr: Optional[int] = None
    policy.close_stage: str = "to_init"
    policy.jacobian_constraint = generate_constraint_jacobian()
    policy.target_rotation_angvel = np.array([0.0, 0.0, 0.0])
    policy.target_rotation_linvel = np.array([0.03, 0.0, 0.0])
    policy.last_angvel_flip_time: Optional[float] = None
    policy.pos_kp = 300  # High stiffness for anisotropic fingers.
    policy.force_kp = 200  # Low stiffness for anisotropic fingers.
    policy.rot_kp = 20
    policy.baseline_tip_rot: Dict[str, R] = {}
    policy.interval = 1.5
    # Store last contact position for each fingertip (used during close phase)
    policy.last_contact_pos: Dict[str, Optional[np.ndarray]] = {
        tip: None for tip in policy.wrench_site_names
    }

    # Control mode: "rotation" or "translation"
    policy.control_mode = "translation"  # Default mode
    policy.rotation_angvel_magnitude = 0.5  # ±0.5 rad/s for rotation mode
    policy.translation_linvel_magnitude = 0.03  # ±0.03 m/s for translation mode

    # HFVC parameters for different modes
    default_force = OBJECT_MASS_MAP.get(policy.object_type, OBJECT_MASS_MAP["unknown"])
    policy.min_normal_force_rotation = float(default_force["min_normal_force_rotation"])
    policy.min_normal_force_translation = float(
        default_force["min_normal_force_translation"]
    )

    # Centripetal force: additional force toward object center for each end effector
    policy.centripetal_force_magnitude = 1.0  # N, force magnitude toward object center

    # Contact detection via external wrench
    policy.contact_force_threshold = 0.1

    # Threshold mechanism (relative to initial position)
    policy.threshold_angle = np.pi / 4  # 45 degrees for rotation mode
    policy.threshold_position = 0.05  # 5cm for translation mode
    policy.threshold_angle_reverse = -policy.threshold_angle
    policy.threshold_position_reverse = -0.02
    policy.auto_switch_target_enabled = auto_switch_target_enabled
    policy.auto_switch_counter = 0  # 0=reverse, 1=mode_switch, 2=reverse, ...
    policy.limit_reached_flag = False
    policy.integrated_angle = 0.0  # Integrated angle from angvel
    policy.integrated_position = 0.0  # Integrated position from linvel
    policy.last_integration_time: Optional[float] = None

    # Mode switching state
    policy.mode_switch_pending = False
    policy.target_mode: Optional[str] = None
    policy.return_to_zero_tolerance = 0.003

    # Stdin receiver for keyboard control.
    policy.control_receiver: Optional[KeyboardControlReceiver] = None
    try:
        policy.control_receiver = KeyboardControlReceiver()
        if policy.control_receiver is not None and policy.control_receiver.enabled:
            print(
                "[LeapRotateCompliance] Keyboard control active on stdin "
                "(c=reverse, r=switch mode)."
            )
    except Exception as exc:
        policy.control_receiver = None
        print(f"[LeapRotateCompliance] Warning: control receiver disabled: {exc}")


def _leap_build_pose_command(
    policy, pose_data: Dict[str, Dict[str, np.ndarray]]
) -> npt.NDArray[np.float32]:
    pose_cmd = np.zeros((policy.num_sites, 6), dtype=np.float32)
    for idx, site in enumerate(policy.wrench_site_names):
        site_data = pose_data.get(site)
        if site_data is None:
            raise ValueError(f"No pose data provided for site '{site}'.")
        pose_cmd[idx, :3] = site_data["pos"]
        pose_cmd[idx, 3:6] = site_data["ori"]
    return pose_cmd


def _leap_compute_force_and_stiffness(
    policy, pose: npt.NDArray[np.float32]
) -> Dict[str, np.ndarray]:
    rot_mats = R.from_rotvec(pose[:, 3:6]).as_matrix().astype(np.float32)
    normals = []
    for idx, site in enumerate(policy.wrench_site_names):
        local_normal = (
            np.array([1.0, 0.0, 0.0], dtype=np.float32)
            if site == "th_tip"
            else np.array([0.0, 0.0, 1.0], dtype=np.float32)
        )
        normal = rot_mats[idx] @ local_normal
        normals.append(normal / (np.linalg.norm(normal) + 1e-9))

    normals_arr = np.asarray(normals, dtype=np.float32)
    wrench = np.zeros((policy.num_sites, 6), dtype=np.float32)
    wrench[:, :3] = normals_arr * policy.contact_force

    eye = np.eye(3, dtype=np.float32)
    pos_stiff = []
    rot_stiff = []
    for normal in normals_arr:
        outer = np.outer(normal, normal)
        pos_stiff.append(
            eye * policy.tangent_pos_stiffness
            + (policy.normal_pos_stiffness - policy.tangent_pos_stiffness) * outer
        )
        rot_stiff.append(
            eye * policy.tangent_rot_stiffness
            + (policy.normal_rot_stiffness - policy.tangent_rot_stiffness) * outer
        )
    pos_stiff_arr = np.stack(pos_stiff, axis=0).astype(np.float32)
    rot_stiff_arr = np.stack(rot_stiff, axis=0).astype(np.float32)

    mass_matrix = ensure_matrix(1.0)
    inertia_matrix = ensure_matrix([1.0, 1.0, 1.0])
    pos_damp = np.stack(
        [get_damping_matrix(mat, mass_matrix) for mat in pos_stiff_arr], axis=0
    ).astype(np.float32)
    rot_damp = np.stack(
        [get_damping_matrix(mat, inertia_matrix) for mat in rot_stiff_arr], axis=0
    ).astype(np.float32)

    return {
        "pos_stiff": pos_stiff_arr,
        "rot_stiff": rot_stiff_arr,
        "pos_damp": pos_damp,
        "rot_damp": rot_damp,
        "wrench": wrench,
    }


def _leap_build_command_trajectory(
    policy,
    pose_start: npt.NDArray[np.float32],
    pose_target: npt.NDArray[np.float32],
    gains_start: Dict[str, np.ndarray],
    gains_target: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    pose_start = np.asarray(pose_start, dtype=np.float32)
    pose_target = np.asarray(pose_target, dtype=np.float32)

    pos_delta_max = float(
        np.linalg.norm(pose_target[:, :3] - pose_start[:, :3], axis=1).max()
    )
    rot_delta_max = float(
        np.linalg.norm(
            (
                R.from_rotvec(pose_target[:, 3:6])
                * R.from_rotvec(pose_start[:, 3:6]).inv()
            ).as_rotvec(),
            axis=1,
        ).max()
    )
    duration = max(
        pos_delta_max / max(policy.pose_interp_pos_speed, 1e-6),
        rot_delta_max / max(policy.pose_interp_rot_speed, 1e-6),
    )
    duration = float(
        np.clip(
            duration,
            policy.pose_interp_min_duration,
            policy.pose_interp_max_duration,
        )
    )

    t_samples = np.arange(
        0.0, duration + policy.control_dt, policy.control_dt, dtype=np.float32
    )
    if t_samples.size == 0 or t_samples[-1] < duration:
        t_samples = np.append(t_samples, np.float32(duration))
    u = np.clip(t_samples / max(duration, 1e-6), 0.0, 1.0)
    weights = u

    pos_interp = (
        pose_start[None, :, :3]
        + (pose_target[None, :, :3] - pose_start[None, :, :3]) * weights[:, None, None]
    ).astype(np.float32)

    ori_interp = np.zeros((weights.size, policy.num_sites, 3), dtype=np.float32)
    for idx in range(policy.num_sites):
        rot_start = pose_start[idx, 3:6]
        rot_target = pose_target[idx, 3:6]
        if np.allclose(rot_start, rot_target, atol=1e-6):
            ori_interp[:, idx] = rot_target
            continue
        key_rots = R.from_rotvec(np.stack([rot_start, rot_target], axis=0))
        slerp = Slerp([0.0, 1.0], key_rots)
        interp_rots = slerp(weights)
        ori_interp[:, idx] = interp_rots.as_rotvec().astype(np.float32)

    def blend(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        w = weights.reshape((-1,) + (1,) * a.ndim)
        return (a[None] + (b - a)[None] * w).astype(np.float32)

    traj = {
        "time": t_samples,
        "pos": pos_interp,
        "ori": ori_interp,
        "pos_stiff": blend(gains_start["pos_stiff"], gains_target["pos_stiff"]),
        "rot_stiff": blend(gains_start["rot_stiff"], gains_target["rot_stiff"]),
        "pos_damp": blend(gains_start["pos_damp"], gains_target["pos_damp"]),
        "rot_damp": blend(gains_start["rot_damp"], gains_target["rot_damp"]),
        "wrench": blend(gains_start["wrench"], gains_target["wrench"]),
    }
    return traj


def _leap_apply_traj_sample(policy, traj: Dict[str, np.ndarray], idx: int) -> None:
    idx = int(np.clip(idx, 0, traj["time"].shape[0] - 1))
    policy.pose_command[:, :3] = traj["pos"][idx]
    policy.pose_command[:, 3:6] = traj["ori"][idx]
    policy.wrench_command = traj["wrench"][idx].copy()

    # Directly set stiffness arrays (already in correct shape: num_sites x 9)
    policy.pos_stiffness = np.asarray(traj["pos_stiff"][idx], dtype=np.float32).reshape(
        policy.num_sites, 9
    )
    policy.rot_stiffness = np.asarray(traj["rot_stiff"][idx], dtype=np.float32).reshape(
        policy.num_sites, 9
    )
    policy.pos_damping = np.asarray(traj["pos_damp"][idx], dtype=np.float32).reshape(
        policy.num_sites, 9
    )
    policy.rot_damping = np.asarray(traj["rot_damp"][idx], dtype=np.float32).reshape(
        policy.num_sites, 9
    )


def _leap_start_command_trajectory(
    policy, traj: Dict[str, np.ndarray], time_curr: Optional[float]
) -> None:
    policy.active_traj = traj
    policy.traj_start_time = float(time_curr if time_curr is not None else 0.0)
    _leap_apply_traj_sample(policy, traj, 0)


def _leap_advance_command_trajectory(policy, time_curr: float) -> None:
    if policy.active_traj is None:
        return
    times = policy.active_traj["time"]
    elapsed = time_curr - policy.traj_start_time
    idx = int(np.searchsorted(times, elapsed, side="right") - 1)
    _leap_apply_traj_sample(policy, policy.active_traj, idx)
    if elapsed >= float(times[-1]):
        policy.active_traj = None


def _leap_check_control_command(policy) -> str | None:
    """Check for keyboard commands via stdin receiver."""
    if policy.control_receiver is None:
        return None

    msg = policy.control_receiver.poll_command()
    if msg is None or msg.command is None:
        return None

    cmd = str(msg.command).strip().lower()
    return cmd if cmd in ("c", "r") else None


def _leap_update_goal(policy, time_curr: float) -> None:
    """Update target velocities based on keyboard commands and threshold.

    Commands:
    - 'c': Reverse current target (flip sign of angvel or linvel)
    - 'r': Switch between rotation mode and translation mode

    Threshold logic:
    - Integrates angvel/linvel to track relative position from initial state
    - When reaching threshold, sets velocity to 0
    - Pressing 'c' reverses direction to move toward reverse threshold
    """
    # Initialize integration timer on first call during rotate phase
    if policy.last_integration_time is None:
        policy.last_integration_time = time_curr

    # Calculate dt for integration
    dt = time_curr - policy.last_integration_time
    policy.last_integration_time = time_curr

    # Integrate velocities to track relative position
    if policy.control_mode == "rotation":
        # Integrate angular velocity (z-axis component)
        angvel_z = policy.target_rotation_angvel[2]
        policy.integrated_angle += angvel_z * dt
        current_metric = policy.integrated_angle
        threshold = policy.threshold_angle
        reverse_threshold = policy.threshold_angle_reverse
    else:
        # Integrate linear velocity (x-axis component)
        linvel_x = policy.target_rotation_linvel[0]
        policy.integrated_position += linvel_x * dt
        current_metric = policy.integrated_position
        threshold = policy.threshold_position
        reverse_threshold = policy.threshold_position_reverse

    # Check for keyboard commands
    cmd = _leap_check_control_command(
        policy,
    )

    if cmd == "c":
        _leap_apply_reverse_command(
            policy,
        )

    elif cmd == "r":
        _leap_request_mode_switch(
            policy,
        )

    if policy.mode_switch_pending:
        at_zero = False
        if policy.control_mode == "rotation":
            at_zero = abs(policy.integrated_angle) < policy.return_to_zero_tolerance
            if not at_zero:
                direction = -np.sign(policy.integrated_angle)
                if direction == 0:
                    direction = 1.0
                policy.target_rotation_angvel = np.array(
                    [0.0, 0.0, direction * policy.rotation_angvel_magnitude]
                )
        else:
            at_zero = abs(policy.integrated_position) < policy.return_to_zero_tolerance
            if not at_zero:
                direction = -np.sign(policy.integrated_position)
                if direction == 0:
                    direction = 1.0
                policy.target_rotation_linvel = np.array(
                    [direction * policy.translation_linvel_magnitude, 0.0, 0.0]
                )

        if at_zero:
            policy.mode_switch_pending = False
            # Reset pose_command to target pose to avoid drift
            policy.pose_command = policy.target_pose_command.copy()
            print(
                "[LeapRotateCompliance] Reset pose_command to target_pose_command to prevent drift"
            )

            if policy.target_mode == "translation":
                policy.control_mode = "translation"
                policy.integrated_position = 0.0
                policy.target_rotation_angvel = np.array([0.0, 0.0, 0.0])
                policy.target_rotation_linvel = np.array(
                    [policy.translation_linvel_magnitude, 0.0, 0.0]
                )
                print(
                    f"[LeapRotateCompliance] Switched to TRANSLATION mode: linvel = {policy.target_rotation_linvel}"
                )
            else:
                policy.control_mode = "rotation"
                policy.integrated_angle = 0.0
                policy.target_rotation_linvel = np.array([0.0, 0.0, 0.0])
                policy.target_rotation_angvel = np.array(
                    [0.0, 0.0, policy.rotation_angvel_magnitude]
                )
                print(
                    f"[LeapRotateCompliance] Switched to ROTATION mode: angvel = {policy.target_rotation_angvel}"
                )
            policy.target_mode = None
            # Skip threshold check this frame to ensure velocity is applied
            return
        return

    if policy.control_mode == "rotation":
        active_vel = policy.target_rotation_angvel[2]
    else:
        active_vel = policy.target_rotation_linvel[0]
    active_threshold = abs(reverse_threshold) if active_vel < 0.0 else threshold
    moving_outward = False
    just_reached_limit = False
    if abs(current_metric) >= abs(active_threshold):
        # Check if moving outward (away from origin)
        if policy.control_mode == "rotation":
            moving_outward = np.sign(policy.target_rotation_angvel[2]) == np.sign(
                current_metric
            )
        else:
            moving_outward = np.sign(policy.target_rotation_linvel[0]) == np.sign(
                current_metric
            )

        if moving_outward:
            just_reached_limit = not policy.limit_reached_flag
            if just_reached_limit:
                policy.limit_reached_flag = True
            # Stop at threshold
            if policy.control_mode == "rotation":
                policy.target_rotation_angvel = np.array([0.0, 0.0, 0.0])
                policy.integrated_angle = float(
                    np.clip(
                        policy.integrated_angle, -active_threshold, active_threshold
                    )
                )
                print(
                    f"[LeapRotateCompliance] Reached threshold: angle = {policy.integrated_angle:.3f}, stopped"
                )
            else:
                policy.target_rotation_linvel = np.array([0.0, 0.0, 0.0])
                policy.integrated_position = float(
                    np.clip(
                        policy.integrated_position,
                        -active_threshold,
                        active_threshold,
                    )
                )
                print(
                    f"[LeapRotateCompliance] Reached threshold: position = {policy.integrated_position:.3f}, stopped"
                )
            _leap_auto_switch_target(policy, just_reached_limit)

    if (not moving_outward) or (abs(current_metric) < abs(active_threshold) * 0.98):
        policy.limit_reached_flag = False


def _leap_apply_reverse_command(policy) -> None:
    """Reverse direction as if pressing 'c'."""
    if policy.control_mode == "rotation":
        if np.linalg.norm(policy.target_rotation_angvel) < 1e-6:
            direction = -np.sign(policy.integrated_angle)
            if direction == 0:
                direction = 1.0
        else:
            direction = -np.sign(policy.target_rotation_angvel[2])
        policy.target_rotation_angvel = np.array(
            [0.0, 0.0, direction * policy.rotation_angvel_magnitude]
        )
        print(
            f"[LeapRotateCompliance] Reversed rotation: angvel = {policy.target_rotation_angvel}, integrated_angle = {policy.integrated_angle:.3f}"
        )
    else:
        if np.linalg.norm(policy.target_rotation_linvel) < 1e-6:
            direction = -np.sign(policy.integrated_position)
            if direction == 0:
                direction = 1.0
        else:
            direction = -np.sign(policy.target_rotation_linvel[0])
        policy.target_rotation_linvel = np.array(
            [direction * policy.translation_linvel_magnitude, 0.0, 0.0]
        )
        print(
            f"[LeapRotateCompliance] Reversed translation: linvel = {policy.target_rotation_linvel}, integrated_position = {policy.integrated_position:.3f}"
        )


def _leap_request_mode_switch(policy) -> None:
    """Request a mode switch as if pressing 'r'."""
    if policy.mode_switch_pending:
        return
    policy.mode_switch_pending = True
    if policy.control_mode == "rotation":
        policy.target_mode = "translation"
        print(
            f"[LeapRotateCompliance] Mode switch requested: rotation -> translation, returning to zero first (angle={policy.integrated_angle:.3f})"
        )
    else:
        policy.target_mode = "rotation"
        print(
            f"[LeapRotateCompliance] Mode switch requested: translation -> rotation, returning to zero first (position={policy.integrated_position:.3f})"
        )


def _leap_auto_switch_target(policy, just_reached_limit: bool) -> None:
    """Auto-switch target when reaching limit: reverse -> mode_switch -> reverse -> ..."""
    if not policy.auto_switch_target_enabled:
        return
    if not just_reached_limit:
        return

    action_type = policy.auto_switch_counter % 2  # 0=reverse, 1=mode_switch
    if action_type == 0:
        _leap_apply_reverse_command(
            policy,
        )
    else:
        _leap_request_mode_switch(
            policy,
        )
    policy.auto_switch_counter += 1


def _leap__ensure_object_detected(policy) -> None:
    if policy.object_type_detected:
        return
    policy.object_type = OBJECT_TYPE
    object_info = OBJECT_MASS_MAP.get(policy.object_type, OBJECT_MASS_MAP["unknown"])
    policy.object_mass = float(object_info.get("mass", 0.05))
    policy.object_init_pos = object_info.get("init_pos")
    policy.object_init_quat = object_info.get("init_quat")
    policy.object_geom_size = object_info.get("geom_size")
    policy.min_normal_force_rotation = float(
        object_info.get(
            "min_normal_force_rotation",
            policy.min_normal_force_rotation,
        )
    )
    policy.min_normal_force_translation = float(
        object_info.get(
            "min_normal_force_translation",
            policy.min_normal_force_translation,
        )
    )
    if policy.object_init_pos is not None:
        policy.object_init_pos = policy.object_init_pos.copy()
    if policy.object_init_quat is not None:
        policy.object_init_quat = policy.object_init_quat.copy()
    policy.object_type_detected = True
    print(
        f"[LeapRotateCompliance] Detected object type: {policy.object_type}, mass: {policy.object_mass}kg"
    )


def _leap_forward_object_to_init(policy, sim_name: str = "sim") -> None:
    """Immediately place object at policy init pose and forward once."""
    _leap__ensure_object_detected(
        policy,
    )
    _leap_capture_object_init(
        policy,
    )
    _leap_fix_object(policy, policy.wrench_sim, sim_name=sim_name)
    mujoco.mj_forward(policy.wrench_sim.model, policy.wrench_sim.data)


def _leap_step(
    policy,
    time_curr: float,
    wrenches_by_site: Optional[Dict[str, np.ndarray]] = None,
    system_state: Optional[Dict[str, np.ndarray]] = None,
    *,
    sim_name: str = "sim",
    is_real_world: bool = False,
) -> Dict[str, np.ndarray | str]:
    if wrenches_by_site is not None:
        policy.wrenches_by_site = {
            key: np.asarray(val, dtype=np.float32)
            for key, val in wrenches_by_site.items()
        }

    _leap__ensure_object_detected(
        policy,
    )

    if time_curr < policy.prep_duration:
        _leap_capture_object_init(
            policy,
        )
        if not is_real_world:
            _leap_fix_object(policy, policy.wrench_sim, sim_name=sim_name)
        return _leap_get_outputs(
            policy,
        )

    if policy.phase == "close":
        _leap_capture_object_init(
            policy,
        )
        if not is_real_world:
            _leap_fix_object(policy, policy.wrench_sim, sim_name=sim_name)
        if policy.close_stage == "to_init":
            if policy.active_traj is None:
                traj = _leap_build_command_trajectory(
                    policy,
                    policy.pose_command.copy(),
                    policy.initial_pose_command,
                    policy.open_gains,
                    policy.open_gains,
                )
                _leap_start_command_trajectory(policy, traj, time_curr)
        elif policy.close_stage == "to_target":
            if policy.active_traj is None and not policy.traj_set:
                _leap_start_command_trajectory(policy, policy.forward_traj, time_curr)
                policy.traj_set = True
            elif policy.active_traj is None:
                _leap_check_switch_phase(
                    policy,
                )
        _leap_advance_command_trajectory(policy, time_curr)
    elif policy.phase == "rotate":
        # Update goal with keyboard commands and threshold checking
        _leap_update_goal(policy, time_curr)

        # Handle rotation action
        _leap_handle_rotate_action(policy, system_state)
        _leap_check_switch_phase(
            policy,
        )

    if (
        policy.phase == "close"
        and policy.close_stage == "to_init"
        and policy.active_traj is None
    ):
        policy.close_stage = "to_target"
        policy.traj_set = False
    return _leap_get_outputs(
        policy,
    )


def _leap_get_outputs(policy) -> Dict[str, np.ndarray | str]:
    return {
        "phase": policy.phase,
        "control_mode": policy.control_mode,
        "pose_command": policy.pose_command.copy(),
        "wrench_command": policy.wrench_command.copy(),
        "pos_stiffness": policy.pos_stiffness.copy(),
        "rot_stiffness": policy.rot_stiffness.copy(),
        "pos_damping": policy.pos_damping.copy(),
        "rot_damping": policy.rot_damping.copy(),
    }


def _leap_assign_stiffness(
    policy,
    left_vel: np.ndarray,
    right_vel: np.ndarray,
) -> None:
    """Set anisotropic stiffness for index/middle; others very stiff."""

    def build_diag(vel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        dir_vec = np.asarray(vel, dtype=np.float32).reshape(-1)
        norm = np.linalg.norm(dir_vec)
        pos_high = float(policy.pos_kp)
        pos_low = float(policy.force_kp)
        eye = np.eye(3, dtype=np.float32)
        if norm < 1e-6:
            diag = np.full(3, pos_low, dtype=np.float32)
        else:
            dir_unit = dir_vec / norm
            proj = np.outer(dir_unit, dir_unit)
            mat = eye * pos_low + (pos_high - pos_low) * proj
            diag = np.diag(mat)
        damp = 2.0 * np.sqrt(diag)
        return diag, damp

    # Map finger tips to velocities
    vel_map = {
        "if_tip": left_vel,
        "mf_tip": right_vel,
    }

    # High stiffness for non-anisotropic fingers
    high_stiff_diag = np.full(3, float(policy.pos_kp), dtype=np.float32)
    high_damp_diag = 2.0 * np.sqrt(high_stiff_diag)

    # Rotation stiffness/damping (same for all fingers)
    rot_stiff_diag = np.full(3, float(policy.rot_kp), dtype=np.float32)
    rot_damp_diag = 2.0 * np.sqrt(rot_stiff_diag)

    # Set stiffness for each finger
    for idx, tip in enumerate(policy.wrench_site_names):
        if tip in vel_map:
            pos_diag, pos_damp_diag = build_diag(vel_map[tip])
        else:
            pos_diag = high_stiff_diag
            pos_damp_diag = high_damp_diag

        policy.pos_stiffness[idx] = np.diag(pos_diag).flatten()
        policy.pos_damping[idx] = np.diag(pos_damp_diag).flatten()
        policy.rot_stiffness[idx] = np.diag(rot_stiff_diag).flatten()
        policy.rot_damping[idx] = np.diag(rot_damp_diag).flatten()


def _leap_set_phase(policy, phase: str) -> None:
    """Update phase and reset trajectory flag whenever phase changes."""
    if policy.phase != phase:
        policy.phase = phase
        policy.traj_set = False


def _leap_check_switch_phase(policy) -> None:
    """Switch from close to rotate once all fingertips have sufficient contact force."""
    if policy.phase == "close":
        has_contact = _leap_check_all_fingertips_contact(
            policy,
        )
        if has_contact:
            _leap_freeze_pose_to_current(
                policy,
            )
            _leap_capture_baseline_tip_rot(
                policy,
            )
            _leap_set_phase(policy, "rotate")
    else:
        return


def _leap_check_all_fingertips_contact(policy) -> bool:
    """Check if index or middle fingertip has contact based on external wrench."""
    if not hasattr(policy, "wrenches_by_site") or not policy.wrenches_by_site:
        return False

    for tip in ("if_tip", "mf_tip"):
        wrench = policy.wrenches_by_site.get(tip)
        if wrench is None:
            continue
        force_magnitude = np.linalg.norm(wrench[:3])
        if force_magnitude >= policy.contact_force_threshold:
            return True
    return False


def _leap_capture_baseline_tip_rot(policy) -> None:
    """Cache current fingertip orientations as baseline for relative quats."""
    policy.baseline_tip_rot.clear()
    model = policy.wrench_sim.model
    data = policy.wrench_sim.data
    for tip in ("th_tip", "if_tip", "mf_tip"):
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, tip)
        if sid < 0:
            continue
        mat = data.site_xmat[sid].reshape(3, 3)
        policy.baseline_tip_rot[tip] = R.from_matrix(mat)


def _leap_freeze_pose_to_current(policy) -> None:
    """Set pose_command to current site poses to avoid jumps when switching phase."""
    model = policy.wrench_sim.model
    data = policy.wrench_sim.data
    for idx, site in enumerate(policy.wrench_site_names):
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site)
        if sid < 0:
            continue
        policy.pose_command[idx, :3] = data.site_xpos[sid]
        rotvec = R.from_matrix(data.site_xmat[sid].reshape(3, 3)).as_rotvec()
        policy.pose_command[idx, 3:6] = rotvec.astype(np.float32)


def _leap_capture_object_init(policy) -> None:
    """Store object's initial pose once."""
    if policy.object_init_pos is None:
        init_pos = OBJECT_INIT_POS_MAP.get(policy.object_type)
        if init_pos is None:
            init_pos = np.zeros(3, dtype=np.float32)
        policy.object_init_pos = init_pos.copy()

    if policy.object_init_quat is None:
        object_info = OBJECT_MASS_MAP.get(
            policy.object_type, OBJECT_MASS_MAP["unknown"]
        )
        init_quat = object_info.get("init_quat")
        if init_quat is None:
            init_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        policy.object_init_quat = init_quat.copy()


def _leap_fix_object(policy, sim: Any, sim_name: str = "sim") -> None:
    """Keep object fixed at the captured pose during close phase."""
    if "real" in str(sim_name).lower():
        return
    if policy.object_init_pos is None or policy.object_init_quat is None:
        return

    body_id = mujoco.mj_name2id(
        sim.model, mujoco.mjtObj.mjOBJ_BODY, policy.object_body_name
    )
    if body_id < 0:
        if policy.object_init_pos is None or policy.object_init_quat is None:
            policy.object_init_pos = np.zeros(3, dtype=np.float32)
            policy.object_init_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        return

    # Resolve the first joint attached to this body (free joint expected).
    jnt_adr = sim.model.body_jntadr[body_id]
    policy.object_qpos_adr = int(sim.model.jnt_qposadr[jnt_adr])
    policy.object_qvel_adr = int(sim.model.jnt_dofadr[jnt_adr])
    if (
        policy.object_qpos_adr is not None
        and policy.object_qpos_adr + 7 <= sim.model.nq
    ):
        qpos_slice = slice(policy.object_qpos_adr, policy.object_qpos_adr + 7)
        sim.data.qpos[qpos_slice][0:3] = policy.object_init_pos
        sim.data.qpos[qpos_slice][3:7] = policy.object_init_quat
    if (
        policy.object_qvel_adr is not None
        and policy.object_qvel_adr + 6 <= sim.model.nv
    ):
        qvel_slice = slice(policy.object_qvel_adr, policy.object_qvel_adr + 6)
        sim.data.qvel[qvel_slice] = 0.0


def _leap_apply_pd(policy, sim: Any) -> None:
    # Kept for API parity with toddlerbot policy; no-op in standalone mode.
    return


def _leap_get_system_state(policy) -> Dict[str, np.ndarray]:
    """Return object and fingertip state using site positions.

    Thumb contact -> fix_*, index contact -> left_*, middle contact -> right_*.
    Positions come from site poses (not contact points).
    Orientation/velocities come from the fingertip sites.
    """
    model: mujoco.MjModel = policy.wrench_sim.model
    data: mujoco.MjData = policy.wrench_sim.data

    def get_sensor_data(sensor_name: str) -> Optional[np.ndarray]:
        try:
            sensor_id = model.sensor(sensor_name).id
        except Exception:
            return None
        sensor_adr = model.sensor_adr[sensor_id]
        sensor_dim = model.sensor_dim[sensor_id]
        return data.sensordata[sensor_adr : sensor_adr + sensor_dim].copy()

    def tip_state(tip_name: str) -> Dict[str, np.ndarray]:
        pos = get_sensor_data(f"{tip_name}_framepos")
        quat = get_sensor_data(f"{tip_name}_framequat")
        linvel = get_sensor_data(f"{tip_name}_framelinvel")
        angvel = get_sensor_data(f"{tip_name}_frameangvel")
        if pos is None:
            pos = np.zeros(3, dtype=np.float32)
        if quat is None:
            quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        if linvel is None:
            linvel = np.zeros(3, dtype=np.float32)
        if angvel is None:
            angvel = np.zeros(3, dtype=np.float32)
        return {
            "pos": np.asarray(pos, dtype=np.float32),
            "quat": np.asarray(quat, dtype=np.float32),
            "linvel": np.asarray(linvel, dtype=np.float32),
            "angvel": np.asarray(angvel, dtype=np.float32),
            "force": np.zeros(3, dtype=np.float32),
            "torque": np.zeros(3, dtype=np.float32),
        }

    thumb = tip_state("th_tip")
    index = tip_state("if_tip")
    middle = tip_state("mf_tip")

    def relative_quat(tip_name: str, quat_wxyz: np.ndarray) -> np.ndarray:
        if policy.phase != "rotate":
            return quat_wxyz
        base = policy.baseline_tip_rot.get(tip_name)
        if base is None:
            return quat_wxyz
        curr = R.from_quat(
            np.array(
                [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]],
                dtype=np.float32,
            )
        )
        rel = curr * base.inv()
        rel_xyzw = rel.as_quat()
        return np.array(
            [rel_xyzw[3], rel_xyzw[0], rel_xyzw[1], rel_xyzw[2]], dtype=np.float32
        )

    thumb["quat"] = relative_quat("th_tip", thumb["quat"])
    index["quat"] = relative_quat("if_tip", index["quat"])
    middle["quat"] = relative_quat("mf_tip", middle["quat"])

    # Object state from integrated target (no sensor fallback).
    if policy.object_init_pos is None:
        obj_pos = np.zeros(3, dtype=np.float32)
    else:
        obj_pos = policy.object_init_pos.copy()
    obj_pos += np.array([policy.integrated_position, 0.0, 0.0], dtype=np.float32)

    if policy.object_init_quat is None:
        base_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    else:
        base_quat = policy.object_init_quat
    rot_delta = R.from_rotvec(
        np.array([0.0, 0.0, policy.integrated_angle], dtype=np.float32)
    )
    base_rot = R.from_quat(base_quat, scalar_first=True)
    obj_quat = (rot_delta * base_rot).as_quat(scalar_first=True).astype(np.float32)
    obj_linvel = np.zeros(3, dtype=np.float32)
    obj_angvel = np.zeros(3, dtype=np.float32)

    # print(
    #     f"[Expected] pos=[{obj_pos[0]:+.3f}, {obj_pos[1]:+.3f}, {obj_pos[2]:+.3f}] m, rotvec: f{R.from_quat(obj_quat, scalar_first=True).as_rotvec()}, integrated_pos={policy.integrated_position:+.3f}, integrated_angle={policy.integrated_angle:+.3f}"
    # )

    return {
        "sliding_cube_pos": obj_pos,
        "sliding_cube_quat": obj_quat,
        "sliding_cube_linvel": obj_linvel,
        "sliding_cube_angvel": obj_angvel,
        "fix_traj_pos": thumb["pos"],
        "fix_traj_quat": thumb["quat"],
        "fix_traj_linvel": thumb["linvel"],
        "fix_traj_angvel": thumb["angvel"],
        "fix_traj_force": thumb["force"],
        "fix_traj_torque": thumb["torque"],
        "control_left_pos": index["pos"],
        "control_left_quat": index["quat"],
        "control_left_linvel": index["linvel"],
        "control_left_angvel": index["angvel"],
        "control_left_force": index["force"],
        "control_left_torque": index["torque"],
        "control_right_pos": middle["pos"],
        "control_right_quat": middle["quat"],
        "control_right_linvel": middle["linvel"],
        "control_right_angvel": middle["angvel"],
        "control_right_force": middle["force"],
        "control_right_torque": middle["torque"],
    }


def _leap_get_target_vel(policy, state):
    p_thumb_obj = state["fix_traj_pos"] - state["sliding_cube_pos"]
    thumb_linvel = policy.target_rotation_linvel + np.cross(
        policy.target_rotation_angvel, p_thumb_obj
    )
    thumb_angvel = np.zeros(3)

    v_obj_goal = np.cross(policy.target_rotation_angvel - thumb_angvel, -p_thumb_obj)
    omega_obj_goal = policy.target_rotation_angvel - thumb_angvel

    return v_obj_goal, omega_obj_goal, thumb_linvel, thumb_angvel


def _leap_handle_rotate_action(policy, state: Optional[Dict[str, np.ndarray]] = None):
    if state is None:
        state = _leap_get_system_state(
            policy,
        )
    target_linvel, target_angvel, thumb_linvel, thumb_angvel = _leap_get_target_vel(
        policy, state
    )

    min_force = (
        policy.min_normal_force_rotation
        if policy.control_mode == "rotation"
        else policy.min_normal_force_translation
    )
    # print(target_linvel, target_angvel)
    hfvc_inputs = compute_hfvc_inputs(
        state,
        goal_velocity=target_linvel.reshape(-1, 1),
        goal_angvel=target_angvel.reshape(-1, 1),
        friction_coeff_hand=0.8,
        min_normal_force=min_force,
        jac_phi_q_cube_rotating=policy.jacobian_constraint,
        object_mass=policy.object_mass,
        object_type=policy.object_type,
        geom_size=policy.object_geom_size,
    )
    hfvc_solution = solve_ochs(*hfvc_inputs, kNumSeeds=1, kPrintLevel=0)
    if hfvc_solution is None:
        return

    _leap_distribute_action(policy, hfvc_solution, thumb_linvel, thumb_angvel, state)


def _leap_ensure_rotvec_continuity(
    policy, old_rotvec: np.ndarray, new_rotvec: np.ndarray
) -> np.ndarray:
    """Ensure rotvec sign consistency to avoid jumps at pi boundary.

    When a rotation vector crosses the pi boundary, scipy can flip its sign
    (since rotvec and -rotvec represent rotations differing by 2*pi).
    This function ensures the new rotvec maintains the same sign as the old one.

    Args:
        old_rotvec: Previous rotation vector (3,)
        new_rotvec: New rotation vector that might have flipped sign (3,)

    Returns:
        Corrected rotation vector with consistent sign
    """
    # Check if the dot product is negative (opposite directions)
    if np.dot(old_rotvec, new_rotvec) < 0:
        # Flip the sign to maintain continuity
        return -new_rotvec
    return new_rotvec


def _leap_distribute_action(
    policy, hfvc_solution, thumb_linvel, thumb_angvel, state
) -> None:
    """Distribute HFVC center commands to index/middle fingertips."""
    # Convert HFVC (center) wrench/velocity to contact-level targets.
    global_vel, global_frc = transform_hfvc_to_global(hfvc_solution)
    # print(f"global_vel:{global_vel.ravel()}, global_force:{global_frc.ravel()}")

    p_H, _, _, _ = get_center_state(state)
    p_fix = state["fix_traj_pos"].reshape(3)
    r = p_H.reshape(3) - p_fix
    coriolis_term = np.cross(thumb_angvel, r)
    global_vel[:3] += (thumb_linvel + coriolis_term).reshape(3, 1)
    global_vel[3:6] += thumb_angvel.reshape(3, 1)

    v_center = global_vel[:3].reshape(-1)
    omega = global_vel[3:6].reshape(-1)
    F_center = global_frc[:3].reshape(-1, 1)
    M_center = global_frc[3:6].reshape(-1, 1)

    # Use live site poses (not cached state) for distribution.
    model = policy.wrench_sim.model
    data = policy.wrench_sim.data
    idx_if = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "if_tip")
    idx_mf = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "mf_tip")
    if idx_if < 0 or idx_mf < 0:
        return
    p_left = data.site_xpos[idx_if].reshape(3, 1)
    p_right = data.site_xpos[idx_mf].reshape(3, 1)
    center_pos = 0.5 * (p_left + p_right)
    r_left = p_left - center_pos
    r_right = p_right - center_pos

    def cross3(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.array(
            [
                a[1] * b[2] - a[2] * b[1],
                a[2] * b[0] - a[0] * b[2],
                a[0] * b[1] - a[1] * b[0],
            ],
            dtype=np.float32,
        ).reshape(-1, 1)

    v_left = v_center.reshape(3, 1) + cross3(omega, r_left.flatten())
    v_right = v_center.reshape(3, 1) + cross3(omega, r_right.flatten())

    # Set stiffness: index/middle follow their velocities; others stiff.
    _leap_assign_stiffness(policy, v_left, v_right)

    def skew(v: np.ndarray) -> np.ndarray:
        return np.array(
            [[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]],
            dtype=np.float32,
        )

    A = np.zeros((6, 6), dtype=np.float32)
    A[0:3, 0:3] = np.eye(3, dtype=np.float32)
    A[0:3, 3:6] = np.eye(3, dtype=np.float32)
    A[3:6, 0:3] = skew(r_left.flatten())
    A[3:6, 3:6] = skew(r_right.flatten())
    b = np.vstack([F_center.astype(np.float32), M_center.astype(np.float32)])
    forces, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    f_left = forces[0:3].reshape(-1)
    f_right = forces[3:6].reshape(-1)

    # Add centripetal force: force toward object center for each end effector
    object_center = p_H.reshape(3)  # Object center position

    # Calculate centripetal force for left finger (if_tip)
    dir_left_to_center = object_center - p_left.flatten()
    dist_left = np.linalg.norm(dir_left_to_center)
    if dist_left > 1e-6:
        centripetal_left = (
            dir_left_to_center / dist_left
        ) * policy.centripetal_force_magnitude
        f_left = f_left + centripetal_left

    # Calculate centripetal force for right finger (mf_tip)
    dir_right_to_center = object_center - p_right.flatten()
    dist_right = np.linalg.norm(dir_right_to_center)
    if dist_right > 1e-6:
        centripetal_right = (
            dir_right_to_center / dist_right
        ) * policy.centripetal_force_magnitude
        f_right = f_right + centripetal_right

    # Write distributed targets into wrench_command and optional velocity targets.
    # We only set forces here; torques kept zero for fingertips.
    for tip_name, force in zip(("if_tip", "mf_tip"), (f_left, f_right)):
        if tip_name not in policy.wrench_site_names:
            continue
        tip_idx = policy.wrench_site_names.index(tip_name)
        policy.wrench_command[tip_idx, :3] = force.astype(np.float32)
        policy.wrench_command[tip_idx, 3:] = 0.0
        force_mag = float(np.linalg.norm(policy.wrench_command[tip_idx, :3]))
        print(
            f"[Force] {tip_name} |wrench|={force_mag:.3f} N, force={policy.wrench_command[tip_idx, :3]}"
        )
    if "th_tip" in policy.wrench_site_names:
        thumb_idx = policy.wrench_site_names.index("th_tip")
        policy.wrench_command[thumb_idx, :3] = np.array(
            [-float(global_frc[0]), 0.0, 0.0], dtype=np.float32
        )
        policy.wrench_command[thumb_idx, 3:] = 0.0

    # Optionally, update pose_command velocities via a small feed-forward step.
    dt = policy.control_dt
    angvel = policy.target_rotation_angvel.copy()
    # angvel[2] = 0.0
    if_mf_rot_increment = R.from_rotvec(angvel * dt)
    for tip_name, vel in zip(("if_tip", "mf_tip"), (v_left, v_right)):
        if tip_name not in policy.wrench_site_names:
            continue
        tip_idx = policy.wrench_site_names.index(tip_name)
        policy.pose_command[tip_idx, :3] += (vel.reshape(-1) * dt).astype(np.float32)
        # Compose rotations correctly: R_new = R_inc * R_curr
        old_rotvec = policy.pose_command[tip_idx, 3:6].copy()
        curr_rot = R.from_rotvec(old_rotvec)
        new_rot = if_mf_rot_increment * curr_rot
        new_rotvec = new_rot.as_rotvec().astype(np.float32)
        # Ensure sign continuity to avoid jumps
        # new_rotvec = _leap_ensure_rotvec_continuity(policy, old_rotvec, new_rotvec)
        policy.pose_command[tip_idx, 3:6] = new_rotvec

    # set the thumb linvel and angvel
    thumb_idx = policy.wrench_site_names.index("th_tip")
    policy.pose_command[thumb_idx, :3] += (thumb_linvel.reshape(-1) * dt).astype(
        np.float32
    )

    old_rotvec_thumb = policy.pose_command[thumb_idx, 3:6].copy()
    curr_rot = R.from_rotvec(old_rotvec_thumb)
    rot_increment = R.from_rotvec(policy.target_rotation_angvel * dt)
    # print(policy.target_rotation_angvel)
    new_rot = rot_increment * curr_rot
    new_rotvec_thumb = new_rot.as_rotvec().astype(np.float32)
    # Ensure sign continuity to avoid jumps
    # new_rotvec_thumb = _leap_ensure_rotvec_continuity(policy,
    #     old_rotvec_thumb, new_rotvec_thumb
    # )
    # print(new_rotvec_thumb)

    # integrated_angle = R.from_rotvec(policy.integrated_angle_thumb)
    # policy.integrated_angle_thumb = (rot_increment * integrated_angle).as_rotvec()
    # print(policy.integrated_angle_thumb)
    policy.pose_command[thumb_idx, 3:6] = new_rotvec_thumb


def create_leap_rotate_policy(
    wrench_sim: Any,
    wrench_site_names: Tuple[str, ...] = LEAP_FINGER_TIPS,
    control_dt: float = 0.02,
    prep_duration: float = 0.0,
    auto_switch_target_enabled: bool = True,
) -> Any:
    policy = SimpleNamespace()
    _leap_init(
        policy,
        wrench_sim=wrench_sim,
        wrench_site_names=wrench_site_names,
        control_dt=control_dt,
        prep_duration=prep_duration,
        auto_switch_target_enabled=auto_switch_target_enabled,
    )
    return policy


def leap_rotate_policy_forward_object_to_init(
    policy: Any, sim_name: str = "sim"
) -> None:
    _leap_forward_object_to_init(policy, sim_name=sim_name)


def leap_rotate_policy_capture_object_init(policy: Any) -> None:
    _leap_capture_object_init(policy)


def leap_rotate_policy_fix_object(policy: Any, sim: Any, sim_name: str = "sim") -> None:
    _leap_fix_object(policy, sim=sim, sim_name=sim_name)


def leap_rotate_policy_step(
    policy: Any,
    time_curr: float,
    wrenches_by_site: Optional[Dict[str, np.ndarray]] = None,
    system_state: Optional[Dict[str, np.ndarray]] = None,
    sim_name: str = "sim",
    is_real_world: bool = False,
) -> Dict[str, np.ndarray | str]:
    return _leap_step(
        policy,
        time_curr=time_curr,
        wrenches_by_site=wrenches_by_site,
        system_state=system_state,
        sim_name=sim_name,
        is_real_world=is_real_world,
    )
    # print(self.pose_command[thumb_idx, 0:3])


@dataclass(frozen=True)
class _tb_PolicyConfig:
    """Policy parameters aligned with toddlerbot_internal defaults."""

    sim_dt: float = 0.001
    prep_duration: float = 7.0
    prep_hold_duration: float = 5.0
    kneel_sync_qpos: bool = False
    goal_angular_velocity: float = 0.2
    friction_coeff_ground: float = 0.8
    friction_coeff_hand: float = 0.8
    min_hand_normal_force_single: float = 1.0
    min_hand_normal_force_both: float = 1.0
    rolling_ball_mass: float = 0.2
    ball_radius: float = 0.08
    max_wrench_force: float = 3.0
    pos_stiffness_high: float = 400.0
    pos_stiffness_low: float = 100.0
    rot_stiffness_value: float = 40.0

    approach_angle_offset: float = np.pi / 5.0
    approach_interp_duration: float = 1.5
    distance_threshold_margin: float = 0.015
    approach_timeout: float = 5.0
    contact_wait_duration: float = 0.0
    pid_kp: float = 0.0
    kneel_motion_file: str = "descriptions/toddlerbot_2xm/kneel_2xm.lz4"
    initial_active_hands_mode: str = "left"
    threshold_angle: float = np.pi / 5.0
    threshold_angle_z: float = np.pi / 4.0
    print_ochs_world_velocity: bool = False
    ochs_print_interval: float = 0.2


@dataclass
class _tb_PolicyRuntime:
    """Mutable runtime state for model-based policy integration."""

    pose_command: _tb_Array
    wrench_command: _tb_Array
    pos_stiffness: _tb_Array
    pos_damping: _tb_Array
    rot_stiffness: _tb_Array
    rot_damping: _tb_Array
    delta_goal_angular_velocity: _tb_Array

    phase: str
    phase_start_time: float

    active_hands_mode: str
    active_hand_indices: Tuple[int, ...]

    goal_rotate_axis: _tb_Array
    goal_angular_velocity: float
    goal_speed: float
    goal_angle: float
    goal_time: Optional[float]

    kneel_action_arr: _tb_Array
    kneel_qpos: _tb_Array
    kneel_qpos_source_dim: int

    reach_init_state: bool = False
    contact_reach_time: Optional[float] = None
    model_based_start_time: Optional[float] = None

    approach_progress: dict[int, float] | None = None
    approach_start_pose: dict[int, Optional[_tb_Array]] | None = None

    default_left_hand_center_rotvec: Optional[_tb_Array] = None
    default_right_hand_center_rotvec: Optional[_tb_Array] = None

    rigid_body_center: Optional[_tb_Array] = None
    rigid_body_orientation: Optional[_tb_Array] = None  # wxyz
    hand_offsets_in_body_frame: Optional[_tb_Array] = None

    expected_ball_pos: Optional[_tb_Array] = None
    last_ochs_print_time: Optional[float] = None


# Shared model-based helpers moved from examples/compliance_model_based.py


def _find_repo_root(start_dir: str) -> str:
    cur = os.path.abspath(start_dir)
    while True:
        if os.path.isfile(os.path.join(cur, "pyproject.toml")):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            raise FileNotFoundError(
                "Could not locate repository root (pyproject.toml)."
            )
        cur = parent


def _deep_update(dst: dict, src: dict) -> dict:
    for key, value in src.items():
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            _deep_update(dst[key], value)
        else:
            dst[key] = value
    return dst


def _load_motor_params(
    repo_root: str,
    robot_desc_dir: str,
    model: mujoco.MjModel,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
]:
    default_path = os.path.join(repo_root, "descriptions", "default.yml")
    robot_path = os.path.join(robot_desc_dir, "robot.yml")
    motors_path = os.path.join(robot_desc_dir, "motors.yml")

    with open(default_path, "r") as f:
        config = yaml.safe_load(f)
    with open(robot_path, "r") as f:
        robot_cfg = yaml.safe_load(f)
    if robot_cfg is not None:
        _deep_update(config, robot_cfg)
    with open(motors_path, "r") as f:
        motor_cfg = yaml.safe_load(f)
    if motor_cfg is not None:
        _deep_update(config, motor_cfg)

    kp_ratio = float(config["actuators"]["kp_ratio"])
    kd_ratio = float(config["actuators"]["kd_ratio"])
    passive_active_ratio = float(config["actuators"]["passive_active_ratio"])

    names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        for i in range(model.nu)
    ]
    kp = []
    kd = []
    tau_max = []
    q_dot_max = []
    tau_q_dot_max = []
    q_dot_tau_max = []
    tau_brake_max = []
    kd_min = []

    for name in names:
        motor_key = name
        if motor_key not in config["motors"] and motor_key.endswith("_act"):
            base_key = motor_key[: -len("_act")]
            if base_key in config["motors"]:
                motor_key = base_key
        if motor_key not in config["motors"]:
            raise ValueError(f"Missing motor config for actuator '{name}'")

        motor_entry = config["motors"][motor_key]
        motor_type = motor_entry["motor"]
        act_cfg = config["actuators"][motor_type]

        kp.append(float(motor_entry.get("kp", 0.0)) / kp_ratio)
        kd.append(float(motor_entry.get("kd", 0.0)) / kd_ratio)
        tau_max.append(float(act_cfg["tau_max"]))
        q_dot_max.append(float(act_cfg["q_dot_max"]))
        tau_q_dot_max.append(float(act_cfg["tau_q_dot_max"]))
        q_dot_tau_max.append(float(act_cfg["q_dot_tau_max"]))
        tau_brake_max.append(float(act_cfg["tau_brake_max"]))
        kd_min.append(float(act_cfg["kd_min"]))

    return (
        np.asarray(kp, dtype=np.float32),
        np.asarray(kd, dtype=np.float32),
        np.asarray(tau_max, dtype=np.float32),
        np.asarray(q_dot_max, dtype=np.float32),
        np.asarray(tau_q_dot_max, dtype=np.float32),
        np.asarray(q_dot_tau_max, dtype=np.float32),
        np.asarray(tau_brake_max, dtype=np.float32),
        np.asarray(kd_min, dtype=np.float32),
        passive_active_ratio,
    )


def _build_controller(scene_xml_path: str, control_dt: float) -> ComplianceController:
    scene_xml_path = os.path.abspath(scene_xml_path)
    scene_dir = os.path.dirname(scene_xml_path)
    fixed_model_xml = os.path.join(scene_dir, "left_hand_fixed.xml")

    site_names = ("if_tip", "mf_tip", "rf_tip", "th_tip")

    controller_cfg = ControllerConfig(
        xml_path=scene_xml_path,
        site_names=site_names,
        fixed_base=True,
        base_body_name="",
        joint_indices_by_site={
            "if_tip": np.array([0, 1, 2, 3], dtype=np.int32),
            "mf_tip": np.array([4, 5, 6, 7], dtype=np.int32),
            "rf_tip": np.array([8, 9, 10, 11], dtype=np.int32),
            "th_tip": np.array([12, 13, 14, 15], dtype=np.int32),
        },
        motor_indices_by_site={
            "if_tip": np.array([0, 1, 2, 3], dtype=np.int32),
            "mf_tip": np.array([4, 5, 6, 7], dtype=np.int32),
            "rf_tip": np.array([8, 9, 10, 11], dtype=np.int32),
            "th_tip": np.array([12, 13, 14, 15], dtype=np.int32),
        },
        gear_ratios_by_site={
            "if_tip": np.ones(4, dtype=np.float32),
            "mf_tip": np.ones(4, dtype=np.float32),
            "rf_tip": np.ones(4, dtype=np.float32),
            "th_tip": np.ones(4, dtype=np.float32),
        },
    )

    estimate_cfg = WrenchEstimateConfig(
        force_reg=1e-3,
        torque_reg=1e-2,
        force_only=False,
        axis_aligned=False,
        normal_axis="+z",
    )

    ref_cfg = ComplianceRefConfig(
        dt=float(control_dt),
        q_start_idx=0,
        qd_start_idx=0,
        ik_position_only=False,
        fixed_model_xml_path=fixed_model_xml,
        mass=1.0,
        inertia_diag=(1.0, 1.0, 1.0),
        mink_num_iter=5,
        mink_damping=1e-2,
        actuator_indices=tuple(range(16)),
        joint_to_actuator_scale=tuple([1.0] * 16),
        joint_to_actuator_bias=tuple([0.0] * 16),
    )

    return ComplianceController(
        config=controller_cfg,
        estimate_config=estimate_cfg,
        ref_config=ref_cfg,
    )


def _build_command_matrix(
    site_names: tuple[str, ...],
    policy_out: Dict[str, np.ndarray | str],
    measured_wrenches: Dict[str, np.ndarray],
) -> np.ndarray:
    pose_command = np.asarray(policy_out["pose_command"], dtype=np.float32)
    wrench_command = np.asarray(policy_out["wrench_command"], dtype=np.float32)
    pos_stiffness = np.asarray(policy_out["pos_stiffness"], dtype=np.float32)
    rot_stiffness = np.asarray(policy_out["rot_stiffness"], dtype=np.float32)
    pos_damping = np.asarray(policy_out["pos_damping"], dtype=np.float32)
    rot_damping = np.asarray(policy_out["rot_damping"], dtype=np.float32)

    num_sites = len(site_names)
    if pose_command.shape[0] != num_sites:
        raise ValueError(
            f"pose_command rows ({pose_command.shape[0]}) != num_sites ({num_sites})"
        )

    command_matrix = np.zeros((num_sites, COMMAND_LAYOUT.width), dtype=np.float32)
    command_matrix[:, COMMAND_LAYOUT.position] = pose_command[:, :3]
    command_matrix[:, COMMAND_LAYOUT.orientation] = pose_command[:, 3:6]
    command_matrix[:, COMMAND_LAYOUT.kp_pos] = pos_stiffness
    command_matrix[:, COMMAND_LAYOUT.kp_rot] = rot_stiffness
    command_matrix[:, COMMAND_LAYOUT.kd_pos] = pos_damping
    command_matrix[:, COMMAND_LAYOUT.kd_rot] = rot_damping
    command_matrix[:, COMMAND_LAYOUT.force] = wrench_command[:, :3]
    command_matrix[:, COMMAND_LAYOUT.torque] = wrench_command[:, 3:]

    for idx, site_name in enumerate(site_names):
        wrench = measured_wrenches.get(site_name)
        if wrench is None:
            continue
        wrench = np.asarray(wrench, dtype=np.float32).reshape(-1)
        if wrench.shape[0] < 6:
            continue
        command_matrix[idx, COMMAND_LAYOUT.measured_force] = wrench[:3]
        command_matrix[idx, COMMAND_LAYOUT.measured_torque] = wrench[3:6]

    return command_matrix


def _get_ground_truth_wrenches(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    site_names: tuple[str, ...],
) -> Dict[str, np.ndarray]:
    """Match toddlerbot sim behavior: use body cfrc_ext for each fingertip site."""
    mujoco.mj_rnePostConstraint(model, data)
    wrenches: Dict[str, np.ndarray] = {}
    for site_name in site_names:
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if site_id < 0:
            continue
        body_id = int(model.site_bodyid[site_id])
        wrenches[site_name] = np.asarray(
            data.cfrc_ext[body_id], dtype=np.float32
        ).copy()
    return wrenches


# ---- Inlined Toddlerbot model-based helpers ----

_tb_Array = npt.NDArray[np.float64]

_tb_HAND_POS_KEYS = (
    "left_hand_1_pos",
    "left_hand_2_pos",
    "left_hand_3_pos",
    "right_hand_1_pos",
    "right_hand_2_pos",
    "right_hand_3_pos",
)
_tb_HAND_QUAT_KEYS = (
    "left_hand_1_quat",
    "left_hand_2_quat",
    "left_hand_3_quat",
    "right_hand_1_quat",
    "right_hand_2_quat",
    "right_hand_3_quat",
)


def _tb__normalize_mode(mode: str) -> str:
    mode_norm = str(mode).strip().lower()
    return mode_norm if mode_norm in ("left", "right", "both") else "left"


def _tb__active_hand_indices_from_mode(mode: str) -> Tuple[int, ...]:
    m = _tb__normalize_mode(mode)
    if m == "left":
        return (0, 1, 2)
    if m == "right":
        return (3, 4, 5)
    return (0, 1, 2, 3, 4, 5)


def _tb__active_site_indices_from_mode(mode: str) -> Tuple[int, ...]:
    m = _tb__normalize_mode(mode)
    if m == "left":
        return (0,)
    if m == "right":
        return (1,)
    return (0, 1)


def _tb__goal_axis_from_mode(mode: str) -> _tb_Array:
    m = _tb__normalize_mode(mode)
    if m == "both":
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if m == "left":
        return np.array([1.0, 0.0, 0.0], dtype=np.float64)
    return np.array([-1.0, 0.0, 0.0], dtype=np.float64)


def _tb__set_active_hands_mode(
    runtime: _tb_PolicyRuntime,
    cfg: _tb_PolicyConfig,
    mode: str,
    *,
    keep_speed: bool = True,
) -> bool:
    mode_norm = _tb__normalize_mode(mode)
    changed = mode_norm != runtime.active_hands_mode
    if not changed:
        return False

    runtime.active_hands_mode = mode_norm
    runtime.active_hand_indices = _tb__active_hand_indices_from_mode(mode_norm)
    runtime.goal_rotate_axis = _tb__goal_axis_from_mode(mode_norm)
    if keep_speed:
        runtime.goal_speed = max(runtime.goal_speed, abs(runtime.goal_angular_velocity))
    else:
        runtime.goal_speed = max(abs(cfg.goal_angular_velocity), 1e-6)
    runtime.goal_angular_velocity = np.sign(cfg.goal_angular_velocity) * max(
        runtime.goal_speed, 1e-6
    )
    runtime.goal_angle = 0.0
    runtime.goal_time = None
    runtime.expected_ball_pos = None
    runtime.delta_goal_angular_velocity = np.zeros(3, dtype=np.float64)
    return True


def _tb__interpolate_linear(
    p_start: _tb_Array, p_end: _tb_Array, duration: float, t: float
) -> _tb_Array:
    if t <= 0.0:
        return p_start
    if t >= duration:
        return p_end
    return p_start + (p_end - p_start) * (t / duration)


def _tb__binary_search(arr: _tb_Array, t: float) -> int:
    low, high = 0, len(arr) - 1
    while low <= high:
        mid = (low + high) // 2
        if arr[mid] < t:
            low = mid + 1
        elif arr[mid] > t:
            high = mid - 1
        else:
            return mid
    return low - 1


def _tb__interpolate_action(
    t: float, time_arr: _tb_Array, action_arr: _tb_Array
) -> _tb_Array:
    if t <= float(time_arr[0]):
        return np.asarray(action_arr[0], dtype=np.float64)
    if t >= float(time_arr[-1]):
        return np.asarray(action_arr[-1], dtype=np.float64)

    idx = _tb__binary_search(time_arr, t)
    idx = max(0, min(idx, len(time_arr) - 2))
    p_start = np.asarray(action_arr[idx], dtype=np.float64)
    p_end = np.asarray(action_arr[idx + 1], dtype=np.float64)
    duration = float(time_arr[idx + 1] - time_arr[idx])
    return _tb__interpolate_linear(p_start, p_end, duration, t - float(time_arr[idx]))


def _tb__build_prep_traj(
    init_motor_pos: _tb_Array,
    target_motor_pos: _tb_Array,
    prep_duration: float,
    control_dt: float,
    prep_hold_duration: float,
) -> tuple[_tb_Array, _tb_Array]:
    # Keep this implementation numerically aligned with toddlerbot_internal get_action_traj().
    duration = float(max(prep_duration, 0.0))
    dt = float(max(control_dt, 1e-6))
    n_steps = int(duration / dt)
    if n_steps < 2:
        n_steps = 2
    prep_time = np.linspace(0.0, duration, n_steps, endpoint=True, dtype=np.float32)

    init_pos = np.asarray(init_motor_pos, dtype=np.float32).reshape(-1)
    target_pos = np.asarray(target_motor_pos, dtype=np.float32).reshape(-1)
    prep_action = np.zeros((prep_time.shape[0], init_pos.shape[0]), dtype=np.float32)

    blend_duration = max(
        duration - float(np.clip(prep_hold_duration, 0.0, duration)), 0.0
    )
    for i, t in enumerate(prep_time):
        if t < blend_duration:
            prep_action[i] = _tb__interpolate_linear(
                init_pos, target_pos, max(blend_duration, 1e-6), t
            )
        else:
            prep_action[i] = target_pos
    return prep_time, prep_action


def _tb__poll_keyboard_command(control_receiver: object | None) -> str | None:
    if control_receiver is None:
        return None
    if not hasattr(control_receiver, "poll_command"):
        return None
    cmd_obj = control_receiver.poll_command()
    if cmd_obj is None:
        return None
    cmd = getattr(cmd_obj, "command", None)
    if cmd is None:
        return None
    cmd_norm = str(cmd).strip().lower()
    if cmd_norm in ("c", "l", "r", "b"):
        return cmd_norm
    return None


def _tb__update_goal_from_keyboard_and_time(
    runtime: _tb_PolicyRuntime,
    cfg: _tb_PolicyConfig,
    t: float,
    command: str | None,
) -> bool:
    mode_changed = False
    if command == "c":
        if runtime.goal_angular_velocity == 0.0:
            direction = -np.sign(runtime.goal_angle)
            if direction == 0.0:
                direction = -1.0
            runtime.goal_angular_velocity = float(
                direction * max(runtime.goal_speed, 1e-6)
            )
        else:
            runtime.goal_angular_velocity = float(-runtime.goal_angular_velocity)
        runtime.goal_speed = max(abs(runtime.goal_angular_velocity), 1e-6)
    elif command in ("l", "r", "b"):
        mode_map = {"l": "left", "r": "right", "b": "both"}
        mode_changed = _tb__set_active_hands_mode(
            runtime, cfg, mode_map[command], keep_speed=True
        )
        if mode_changed:
            print(
                "[model_based] Mode switch -> "
                f"{runtime.active_hands_mode}, axis={runtime.goal_rotate_axis.tolist()}"
            )
            return True

    if runtime.goal_time is None:
        runtime.goal_time = float(t)
        return mode_changed

    dt = max(float(t - runtime.goal_time), 0.0)
    runtime.goal_time = float(t)

    if runtime.active_hands_mode == "both":
        threshold = float(cfg.threshold_angle_z)
        next_angle = runtime.goal_angle + runtime.goal_angular_velocity * dt
        if abs(next_angle) >= threshold:
            runtime.goal_angle = 0.0
            runtime.goal_time = None
            runtime.goal_angular_velocity = -runtime.goal_angular_velocity
        else:
            runtime.goal_angle = next_angle
        return mode_changed

    threshold = (
        float(cfg.threshold_angle_z)
        if abs(runtime.goal_rotate_axis[2]) > 0.5
        else float(cfg.threshold_angle)
    )
    next_angle = runtime.goal_angle + runtime.goal_angular_velocity * dt
    if abs(next_angle) >= threshold:
        moving_outward = np.sign(runtime.goal_angular_velocity) == np.sign(next_angle)
        if moving_outward:
            runtime.goal_angle = float(np.clip(next_angle, -threshold, threshold))
            runtime.goal_speed = max(
                runtime.goal_speed, abs(runtime.goal_angular_velocity)
            )
            runtime.goal_angular_velocity = 0.0
        else:
            runtime.goal_angle = next_angle
    else:
        runtime.goal_angle = next_angle

    return mode_changed


def _tb__sensor_data(
    model: mujoco.MjModel, data: mujoco.MjData, name: str
) -> _tb_Array:
    sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
    if sensor_id < 0:
        raise KeyError(f"Sensor '{name}' not found.")
    start = int(model.sensor_adr[sensor_id])
    end = start + int(model.sensor_dim[sensor_id])
    return np.asarray(data.sensordata[start:end], dtype=np.float64).copy()


def _tb__build_contact_state(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    left_site_id: int,
    right_site_id: int,
) -> Dict[str, _tb_Array]:
    state: Dict[str, _tb_Array] = {
        "ball_pos": _tb__sensor_data(model, data, "rolling_ball_framepos"),
        # Keep consistency with toddlerbot_internal model-based policy state packing.
        "ball_quat": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
        "ball_linvel": np.array([0.0, 0.0, 0.0], dtype=np.float64),
        "ball_angvel": np.array([0.0, 0.0, 0.0], dtype=np.float64),
        "left_hand_center_pos": np.asarray(
            data.site_xpos[left_site_id], dtype=np.float64
        ).copy(),
        "right_hand_center_pos": np.asarray(
            data.site_xpos[right_site_id], dtype=np.float64
        ).copy(),
    }

    for i in range(1, 4):
        left_prefix = f"left_contact_point_{i}"
        right_prefix = f"right_contact_point_{i}"

        state[f"left_hand_{i}_pos"] = _tb__sensor_data(
            model, data, f"{left_prefix}_pos"
        )
        state[f"left_hand_{i}_quat"] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        state[f"left_hand_{i}_linvel"] = _tb__sensor_data(
            model, data, f"{left_prefix}_linvel"
        )
        state[f"left_hand_{i}_angvel"] = _tb__sensor_data(
            model, data, f"{left_prefix}_angvel"
        )

        state[f"right_hand_{i}_pos"] = _tb__sensor_data(
            model, data, f"{right_prefix}_pos"
        )
        state[f"right_hand_{i}_quat"] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        state[f"right_hand_{i}_linvel"] = _tb__sensor_data(
            model, data, f"{right_prefix}_linvel"
        )
        state[f"right_hand_{i}_angvel"] = _tb__sensor_data(
            model, data, f"{right_prefix}_angvel"
        )

    return state


def _tb__load_robot_motor_config() -> dict:
    repo_root = _find_repo_root(os.path.abspath(os.path.dirname(__file__)))
    default_path = os.path.join(repo_root, "descriptions", "default.yml")
    robot_path = os.path.join(
        repo_root,
        "descriptions",
        "toddlerbot_2xm",
        "robot.yml",
    )

    with open(default_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    with open(robot_path, "r", encoding="utf-8") as f:
        robot_cfg = yaml.safe_load(f)
    if robot_cfg is not None:
        _deep_update(config, robot_cfg)
    return config


def _tb__load_motor_params(model: mujoco.MjModel) -> tuple[_tb_Array, ...]:
    config = _tb__load_robot_motor_config()

    kp_ratio = float(config["actuators"]["kp_ratio"])
    kd_ratio = float(config["actuators"]["kd_ratio"])
    passive_active_ratio = float(config["actuators"]["passive_active_ratio"])

    names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        for i in range(model.nu)
    ]
    kp = []
    kd = []
    tau_max = []
    q_dot_max = []
    tau_q_dot_max = []
    q_dot_tau_max = []
    tau_brake_max = []
    kd_min = []

    for name in names:
        if name not in config["motors"]:
            raise ValueError(f"Missing motor config for actuator '{name}'")
        motor_cfg = config["motors"][name]
        motor_type = motor_cfg["motor"]
        act_cfg = config["actuators"][motor_type]
        kp.append(float(motor_cfg["kp"]) / kp_ratio)
        kd.append(float(motor_cfg["kd"]) / kd_ratio)
        tau_max.append(float(act_cfg["tau_max"]))
        q_dot_max.append(float(act_cfg["q_dot_max"]))
        tau_q_dot_max.append(float(act_cfg["tau_q_dot_max"]))
        q_dot_tau_max.append(float(act_cfg["q_dot_tau_max"]))
        tau_brake_max.append(float(act_cfg["tau_brake_max"]))
        kd_min.append(float(act_cfg["kd_min"]))

    return (
        np.asarray(kp, dtype=np.float64),
        np.asarray(kd, dtype=np.float64),
        np.asarray(tau_max, dtype=np.float64),
        np.asarray(q_dot_max, dtype=np.float64),
        np.asarray(tau_q_dot_max, dtype=np.float64),
        np.asarray(q_dot_tau_max, dtype=np.float64),
        np.asarray(tau_brake_max, dtype=np.float64),
        np.asarray(kd_min, dtype=np.float64),
        np.asarray(passive_active_ratio, dtype=np.float64),
    )


def _tb__load_motor_group_indices(
    model: mujoco.MjModel,
) -> dict[str, npt.NDArray[np.int32]]:
    config = _tb__load_robot_motor_config()
    groups: dict[str, list[int]] = {}
    for i in range(model.nu):
        actuator_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        if actuator_name is None:
            continue
        motor_cfg = config.get("motors", {}).get(actuator_name, {})
        group = motor_cfg.get("group")
        if isinstance(group, str) and len(group) > 0:
            groups.setdefault(group, []).append(i)

    return {k: np.asarray(v, dtype=np.int32) for k, v in groups.items()}


def _tb__load_kneel_trajectory(
    example_dir: str,
    cfg: _tb_PolicyConfig,
    default_motor_pos: _tb_Array,
    default_qpos: _tb_Array,
    motor_dim: int,
    qpos_dim: int,
) -> tuple[_tb_Array, _tb_Array, int]:
    kneel_path = os.environ.get("MCC_KNEEL_TRAJ", str(cfg.kneel_motion_file))
    if not os.path.isabs(kneel_path):
        kneel_path = os.path.join(example_dir, kneel_path)
    kneel_path = os.path.abspath(kneel_path)

    if os.path.exists(kneel_path):
        try:
            data = joblib.load(kneel_path)
            action_arr = np.asarray(data["action"], dtype=np.float64)
            if action_arr.ndim == 1:
                action_arr = action_arr.reshape(1, -1)
            if action_arr.shape[1] != motor_dim:
                raise ValueError(
                    f"kneel action dim {action_arr.shape[1]} != motor_dim {motor_dim}"
                )

            qpos_raw = np.asarray(data["qpos"], dtype=np.float64)
            qpos_last_raw = qpos_raw if qpos_raw.ndim == 1 else qpos_raw[-1]
            qpos_last_raw = np.asarray(qpos_last_raw, dtype=np.float64).reshape(-1)
            source_qpos_dim = int(qpos_last_raw.shape[0])

            qpos_last = np.asarray(default_qpos, dtype=np.float64).reshape(-1).copy()
            if qpos_last.shape[0] != qpos_dim:
                raise ValueError(
                    f"default_qpos dim {qpos_last.shape[0]} != qpos_dim {qpos_dim}"
                )
            copied_dim = min(source_qpos_dim, qpos_dim)
            qpos_last[:copied_dim] = qpos_last_raw[:copied_dim]
            if source_qpos_dim != qpos_dim:
                print(
                    "[model_based] Adjusted kneel qpos "
                    f"{source_qpos_dim} -> {qpos_dim} from {kneel_path}"
                )

            print(f"[model_based] Loaded kneel trajectory: {kneel_path}")
            return action_arr, qpos_last, source_qpos_dim
        except Exception as exc:
            print(f"[model_based] Failed to load kneel trajectory {kneel_path}: {exc}")
    else:
        print(f"[model_based] Kneel trajectory not found: {kneel_path}")

    print("[model_based] Kneel trajectory unavailable, using single-step fallback.")
    fallback_action = np.asarray(default_motor_pos, dtype=np.float64).reshape(1, -1)
    fallback_qpos = np.asarray(default_qpos, dtype=np.float64).copy()
    return fallback_action, fallback_qpos, int(fallback_qpos.shape[0])


def _tb__skew_matrix(vec: _tb_Array) -> _tb_Array:
    vec_arr = np.asarray(vec, dtype=np.float32)
    if vec_arr.ndim == 1 and vec_arr.shape[0] == 3:
        x, y, z = vec_arr
        return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float32)
    if vec_arr.ndim == 2 and vec_arr.shape[1] == 3:
        x = vec_arr[:, 0]
        y = vec_arr[:, 1]
        z = vec_arr[:, 2]
        zeros = np.zeros_like(x)
        return np.stack(
            [
                np.stack([zeros, -z, y], axis=1),
                np.stack([z, zeros, -x], axis=1),
                np.stack([-y, x, zeros], axis=1),
            ],
            axis=1,
        ).astype(np.float32)
    raise ValueError("Input vector must have shape (3,) or (N, 3).")


def _tb__interpolate_se3_pose(
    start_pose: _tb_Array, target_pose: _tb_Array, alpha: float
) -> _tb_Array:
    # Keep interpolation identical to toddlerbot_internal utils.math_utils.interpolate_se3_pose.
    pose0_arr = np.asarray(start_pose, dtype=np.float32)
    pose1_arr = np.asarray(target_pose, dtype=np.float32)
    if pose0_arr.shape != pose1_arr.shape:
        raise ValueError("start_pose and target_pose must have the same shape.")
    if alpha <= 0.0:
        return pose0_arr.astype(np.float64)
    if alpha >= 1.0:
        return pose1_arr.astype(np.float64)

    pose0_flat = pose0_arr.reshape(-1, 6)
    pose1_flat = pose1_arr.reshape(-1, 6)
    pos0 = pose0_flat[:, :3]
    pos1 = pose1_flat[:, :3]
    rot0 = R.from_rotvec(pose0_flat[:, 3:6]).as_matrix()
    rot1 = R.from_rotvec(pose1_flat[:, 3:6]).as_matrix()
    rot0_t = np.swapaxes(rot0, -1, -2)
    rot_rel = np.einsum("nij,njk->nik", rot0_t, rot1)
    pos_rel = np.einsum("nij,nj->ni", rot0_t, pos1 - pos0)

    omega = R.from_matrix(rot_rel).as_rotvec().astype(np.float32)
    theta = np.linalg.norm(omega, axis=1)
    omega_hat = _tb__skew_matrix(omega)
    omega_hat2 = np.einsum("nij,njk->nik", omega_hat, omega_hat)
    eye = np.eye(3, dtype=np.float32)[None, :, :]
    small = theta < 1e-8
    theta_safe = np.where(small, 1.0, theta)
    theta2_safe = theta_safe * theta_safe
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    a = np.where(small, 1.0, sin_theta / theta_safe)
    b = np.where(small, 0.5, (1.0 - cos_theta) / theta2_safe)
    b_safe = np.where(np.abs(b) < 1e-8, 1.0, b)
    c = (1.0 - a / (2.0 * b_safe)) / theta2_safe
    v_inv_small = eye - 0.5 * omega_hat + (1.0 / 12.0) * omega_hat2
    v_inv_large = eye - 0.5 * omega_hat + c[:, None, None] * omega_hat2
    v_inv = np.where(small[:, None, None], v_inv_small, v_inv_large)
    v = np.einsum("nij,nj->ni", v_inv, pos_rel)

    twist = np.concatenate([v, omega], axis=1) * float(alpha)
    v = twist[:, :3]
    omega = twist[:, 3:]
    theta = np.linalg.norm(omega, axis=1)
    omega_hat = _tb__skew_matrix(omega)
    omega_hat2 = np.einsum("nij,njk->nik", omega_hat, omega_hat)
    rot_inc = R.from_rotvec(omega).as_matrix()
    small = theta < 1e-8
    theta_safe = np.where(small, 1.0, theta)
    theta2_safe = theta_safe * theta_safe
    theta3_safe = theta2_safe * theta_safe
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    a = np.where(small, 0.5, (1.0 - cos_theta) / theta2_safe)
    b = np.where(small, 1.0 / 6.0, (theta - sin_theta) / theta3_safe)
    v_mat_small = eye + 0.5 * omega_hat + (1.0 / 6.0) * omega_hat2
    v_mat_large = eye + a[:, None, None] * omega_hat + b[:, None, None] * omega_hat2
    v_mat = np.where(small[:, None, None], v_mat_small, v_mat_large)
    pos_inc = np.einsum("nij,nj->ni", v_mat, v)

    rot_interp = np.einsum("nij,njk->nik", rot0, rot_inc)
    pos_interp = pos0 + np.einsum("nij,nj->ni", rot0, pos_inc)
    rotvec_interp = R.from_matrix(rot_interp).as_rotvec().astype(np.float32)
    interp_pose = np.concatenate([pos_interp, rotvec_interp], axis=1)
    return interp_pose.reshape(pose0_arr.shape).astype(np.float64)


def _tb__ensure_default_hand_rotvec(
    runtime: _tb_PolicyRuntime,
    data: mujoco.MjData,
    left_site_id: int,
    right_site_id: int,
) -> None:
    if runtime.default_left_hand_center_rotvec is None:
        left_mat = np.asarray(data.site_xmat[left_site_id], dtype=np.float64).reshape(
            3, 3
        )
        runtime.default_left_hand_center_rotvec = R.from_matrix(left_mat).as_rotvec()
    if runtime.default_right_hand_center_rotvec is None:
        right_mat = np.asarray(data.site_xmat[right_site_id], dtype=np.float64).reshape(
            3, 3
        )
        runtime.default_right_hand_center_rotvec = R.from_matrix(right_mat).as_rotvec()


def _tb__reset_pose_command_to_current_sites(
    runtime: _tb_PolicyRuntime,
    data: mujoco.MjData,
    left_site_id: int,
    right_site_id: int,
) -> None:
    for idx, site_id in ((0, left_site_id), (1, right_site_id)):
        pos = np.asarray(data.site_xpos[site_id], dtype=np.float64)
        rotmat = np.asarray(data.site_xmat[site_id], dtype=np.float64).reshape(3, 3)
        rotvec = R.from_matrix(rotmat).as_rotvec()
        runtime.pose_command[idx, :3] = pos
        runtime.pose_command[idx, 3:] = rotvec


def _tb__initialize_runtime_from_default_state(
    default_state: ComplianceState,
    cfg: _tb_PolicyConfig,
    kneel_action_arr: _tb_Array,
    kneel_qpos: _tb_Array,
    kneel_qpos_source_dim: int,
) -> _tb_PolicyRuntime:
    pose_command = np.asarray(default_state.x_ref, dtype=np.float64).copy()

    pos_kp_default = np.diag(
        [cfg.pos_stiffness_high, cfg.pos_stiffness_high, cfg.pos_stiffness_high]
    ).reshape(-1)
    rot_kp_default = np.diag(
        [cfg.rot_stiffness_value, cfg.rot_stiffness_value, cfg.rot_stiffness_value]
    ).reshape(-1)

    pos_kd_default = np.diag(
        2.0 * np.sqrt(np.array([cfg.pos_stiffness_high] * 3, dtype=np.float64))
    ).reshape(-1)
    rot_kd_default = np.diag(
        2.0 * np.sqrt(np.array([cfg.rot_stiffness_value] * 3, dtype=np.float64))
    ).reshape(-1)

    num_sites = pose_command.shape[0]
    init_mode = _tb__normalize_mode(cfg.initial_active_hands_mode)
    init_goal_vel = float(cfg.goal_angular_velocity)
    init_goal_speed = max(abs(init_goal_vel), 1e-6)

    return _tb_PolicyRuntime(
        pose_command=pose_command,
        wrench_command=np.zeros((num_sites, 6), dtype=np.float64),
        pos_stiffness=np.tile(pos_kp_default, (num_sites, 1)),
        pos_damping=np.tile(pos_kd_default, (num_sites, 1)),
        rot_stiffness=np.tile(rot_kp_default, (num_sites, 1)),
        rot_damping=np.tile(rot_kd_default, (num_sites, 1)),
        delta_goal_angular_velocity=np.zeros(3, dtype=np.float64),
        phase="prep",
        phase_start_time=0.0,
        active_hands_mode=init_mode,
        active_hand_indices=_tb__active_hand_indices_from_mode(init_mode),
        goal_rotate_axis=_tb__goal_axis_from_mode(init_mode),
        goal_angular_velocity=init_goal_vel,
        goal_speed=init_goal_speed,
        goal_angle=0.0,
        goal_time=None,
        kneel_action_arr=kneel_action_arr,
        kneel_qpos=kneel_qpos,
        kneel_qpos_source_dim=int(kneel_qpos_source_dim),
        approach_progress={0: 0.0, 1: 0.0},
        approach_start_pose={0: None, 1: None},
    )


def _tb__reset_approach_interp(runtime: _tb_PolicyRuntime) -> None:
    if runtime.approach_progress is None:
        runtime.approach_progress = {0: 0.0, 1: 0.0}
    if runtime.approach_start_pose is None:
        runtime.approach_start_pose = {0: None, 1: None}
    for hand_idx in (0, 1):
        runtime.approach_progress[hand_idx] = 0.0
        runtime.approach_start_pose[hand_idx] = None


def _tb__compute_approach_target(
    cfg: _tb_PolicyConfig,
    ball_pos: _tb_Array,
    is_left_hand: bool,
    default_rotvec: Optional[_tb_Array],
) -> tuple[_tb_Array, R]:
    y_sign = 1.0 if is_left_hand else -1.0

    cos_angle = float(np.cos(cfg.approach_angle_offset))
    sin_angle = float(np.sin(cfg.approach_angle_offset))
    target_direction_from_center = np.array(
        [0.0, y_sign * cos_angle, sin_angle],
        dtype=np.float64,
    )
    target_direction = target_direction_from_center / (
        np.linalg.norm(target_direction_from_center) + 1e-9
    )
    target_point = (
        np.asarray(ball_pos, dtype=np.float64).reshape(3)
        + target_direction * cfg.ball_radius
    )

    if default_rotvec is None:
        base_rot = R.from_rotvec(np.zeros(3, dtype=np.float64))
    else:
        base_rot = R.from_rotvec(
            np.asarray(default_rotvec, dtype=np.float64).reshape(3)
        )

    origin_dir = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    target_dir = -target_direction
    align_rot = R.align_vectors([target_dir], [origin_dir])[0]
    target_rotation = align_rot * base_rot

    return target_point, target_rotation


def _tb__interpolate_hand_pose_to_target(
    runtime: _tb_PolicyRuntime,
    cfg: _tb_PolicyConfig,
    hand_idx: int,
    target_pos: _tb_Array,
    target_rot: R,
    control_dt: float,
) -> None:
    assert runtime.approach_progress is not None
    assert runtime.approach_start_pose is not None

    current_pose = runtime.pose_command[hand_idx].copy()
    target_pose = np.concatenate(
        [
            np.asarray(target_pos, dtype=np.float64).reshape(3),
            target_rot.as_rotvec().astype(np.float64),
        ]
    )

    duration = max(float(cfg.approach_interp_duration), 1e-6)

    start_pose = runtime.approach_start_pose.get(hand_idx)
    if start_pose is None:
        runtime.approach_start_pose[hand_idx] = current_pose
        runtime.approach_progress[hand_idx] = 0.0
        start_pose = current_pose

    progress = float(runtime.approach_progress.get(hand_idx, 0.0))
    alpha = float(np.clip(progress / duration, 0.0, 1.0))

    interp_pose = _tb__interpolate_se3_pose(start_pose, target_pose, alpha)
    runtime.pose_command[hand_idx] = interp_pose

    if alpha >= 1.0:
        runtime.approach_progress[hand_idx] = 0.0
        runtime.approach_start_pose[hand_idx] = None
    else:
        runtime.approach_progress[hand_idx] = min(progress + control_dt, duration)


def _tb__run_approach_phase(
    runtime: _tb_PolicyRuntime,
    cfg: _tb_PolicyConfig,
    state: Dict[str, _tb_Array],
    control_dt: float,
) -> bool:
    ball_pos = np.asarray(state["ball_pos"], dtype=np.float64).reshape(3)
    active_sites = set(_tb__active_site_indices_from_mode(runtime.active_hands_mode))
    threshold = float(cfg.ball_radius + cfg.distance_threshold_margin)

    reached = True
    if 0 in active_sites:
        left_pos = np.asarray(state["left_hand_center_pos"], dtype=np.float64).reshape(
            3
        )
        left_distance = float(np.linalg.norm(left_pos - ball_pos))
        reached = reached and (left_distance <= threshold)
    if 1 in active_sites:
        right_pos = np.asarray(
            state["right_hand_center_pos"], dtype=np.float64
        ).reshape(3)
        right_distance = float(np.linalg.norm(right_pos - ball_pos))
        reached = reached and (right_distance <= threshold)

    if reached:
        runtime.reach_init_state = True
        _tb__reset_approach_interp(runtime)
        return True

    left_target_pos, left_target_rot = _tb__compute_approach_target(
        cfg,
        ball_pos,
        is_left_hand=True,
        default_rotvec=runtime.default_left_hand_center_rotvec,
    )
    right_target_pos, right_target_rot = _tb__compute_approach_target(
        cfg,
        ball_pos,
        is_left_hand=False,
        default_rotvec=runtime.default_right_hand_center_rotvec,
    )

    if 0 in active_sites:
        _tb__interpolate_hand_pose_to_target(
            runtime,
            cfg,
            hand_idx=0,
            target_pos=left_target_pos,
            target_rot=left_target_rot,
            control_dt=control_dt,
        )
    if 1 in active_sites:
        _tb__interpolate_hand_pose_to_target(
            runtime,
            cfg,
            hand_idx=1,
            target_pos=right_target_pos,
            target_rot=right_target_rot,
            control_dt=control_dt,
        )

    for hand_idx in (0, 1):
        if hand_idx not in active_sites:
            runtime.approach_start_pose[hand_idx] = None
            runtime.approach_progress[hand_idx] = 0.0
        runtime.wrench_command[hand_idx, :] = 0.0

    return False


def _tb__initialize_rigid_body(
    runtime: _tb_PolicyRuntime, state: Dict[str, _tb_Array]
) -> None:
    hand_positions = []
    hand_quats = []

    for idx in runtime.active_hand_indices:
        hand_key = _tb_HAND_POS_KEYS[idx]
        quat_key = _tb_HAND_QUAT_KEYS[idx]
        hand_positions.append(np.asarray(state[hand_key], dtype=np.float64))
        hand_quats.append(np.asarray(state[quat_key], dtype=np.float64))

    hand_positions_arr = np.asarray(hand_positions, dtype=np.float64)
    runtime.rigid_body_center = np.mean(hand_positions_arr, axis=0).reshape(-1, 1)
    runtime.rigid_body_orientation = compute_center_quaternion_from_hands(hand_quats)

    r_wb = R.from_quat(
        np.asarray(runtime.rigid_body_orientation, dtype=np.float64),
        scalar_first=True,
    ).as_matrix()
    num_active_hands = len(runtime.active_hand_indices)
    runtime.hand_offsets_in_body_frame = np.zeros(
        (num_active_hands, 3), dtype=np.float64
    )

    for local_idx, hand_pos in enumerate(hand_positions_arr):
        p_world = hand_pos.reshape(-1, 1)
        p_relative = p_world - runtime.rigid_body_center
        p_body = r_wb.T @ p_relative
        runtime.hand_offsets_in_body_frame[local_idx] = p_body.flatten()


def _tb__distribute_rigid_body_motion(
    runtime: _tb_PolicyRuntime,
    hfvc_solution: HFVC,
    state: Dict[str, _tb_Array],
    dt: float,
) -> Dict[str, _tb_Array]:
    if runtime.rigid_body_center is None:
        _tb__initialize_rigid_body(runtime, state)

    assert runtime.rigid_body_center is not None
    assert runtime.rigid_body_orientation is not None
    assert runtime.hand_offsets_in_body_frame is not None

    global_vel, global_frc = transform_hfvc_to_global(hfvc_solution)

    total_dof = int(global_vel.shape[0])
    if total_dof >= 12:
        v_center = global_vel[6:9].reshape(-1, 1)
        omega = global_vel[9:12].reshape(-1, 1)
        f_center = global_frc[6:9].reshape(-1, 1)
        m_center = global_frc[9:12].reshape(-1, 1)
    elif total_dof >= 6:
        v_center = global_vel[0:3].reshape(-1, 1)
        omega = global_vel[3:6].reshape(-1, 1)
        f_center = (
            global_frc[0:3].reshape(-1, 1)
            if global_frc.shape[0] >= 3
            else np.zeros((3, 1), dtype=np.float64)
        )
        m_center = (
            global_frc[3:6].reshape(-1, 1)
            if global_frc.shape[0] >= 6
            else np.zeros((3, 1), dtype=np.float64)
        )
    else:
        v_center = np.zeros((3, 1), dtype=np.float64)
        omega = np.zeros((3, 1), dtype=np.float64)
        f_center = np.zeros((3, 1), dtype=np.float64)
        m_center = np.zeros((3, 1), dtype=np.float64)

    runtime.rigid_body_center += v_center * dt

    omega_norm = np.linalg.norm(omega)
    if omega_norm > 1e-6:
        delta_angle = omega_norm * dt
        axis = omega / omega_norm
        delta_quat_xyzw = R.from_rotvec((axis * delta_angle).reshape(-1)).as_quat()

        curr_wxyz = np.asarray(runtime.rigid_body_orientation, dtype=np.float64)
        curr_xyzw = np.array([curr_wxyz[1], curr_wxyz[2], curr_wxyz[3], curr_wxyz[0]])

        updated = R.from_quat(delta_quat_xyzw) * R.from_quat(curr_xyzw)
        uq = updated.as_quat()
        runtime.rigid_body_orientation = np.array([uq[3], uq[0], uq[1], uq[2]])

    r_wb = R.from_quat(
        np.asarray(runtime.rigid_body_orientation, dtype=np.float64),
        scalar_first=True,
    ).as_matrix()

    num_active_hands = len(runtime.active_hand_indices)
    hand_velocities = []
    hand_positions_world = []
    for local_idx in range(num_active_hands):
        offset_body = runtime.hand_offsets_in_body_frame[local_idx].reshape(-1, 1)
        offset_world = r_wb @ offset_body
        hand_pos_world = runtime.rigid_body_center + offset_world
        hand_positions_world.append(hand_pos_world)
        hand_vel = v_center + np.cross(omega.ravel(), offset_world.ravel()).reshape(
            -1, 1
        )
        hand_velocities.append(hand_vel)

    left_local_indices = [
        i
        for i, global_idx in enumerate(runtime.active_hand_indices)
        if global_idx in (0, 1, 2)
    ]
    right_local_indices = [
        i
        for i, global_idx in enumerate(runtime.active_hand_indices)
        if global_idx in (3, 4, 5)
    ]

    if left_local_indices:
        left_hand_pos = np.mean(
            [hand_positions_world[i] for i in left_local_indices], axis=0
        )
        r_left = left_hand_pos - runtime.rigid_body_center
        v_left = v_center + np.cross(omega.ravel(), r_left.ravel()).reshape(-1, 1)
    else:
        left_hand_pos = runtime.rigid_body_center
        r_left = np.zeros((3, 1), dtype=np.float64)
        v_left = v_center

    if right_local_indices:
        right_hand_pos = np.mean(
            [hand_positions_world[i] for i in right_local_indices], axis=0
        )
        r_right = right_hand_pos - runtime.rigid_body_center
        v_right = v_center + np.cross(omega.ravel(), r_right.ravel()).reshape(-1, 1)
    else:
        right_hand_pos = runtime.rigid_body_center
        r_right = np.zeros((3, 1), dtype=np.float64)
        v_right = v_center

    a_mat = np.zeros((6, 6), dtype=np.float64)
    a_mat[0:3, 0:3] = np.eye(3, dtype=np.float64)
    a_mat[0:3, 3:6] = np.eye(3, dtype=np.float64)
    a_mat[3:6, 0:3] = np.array(
        [
            [0, -r_left[2, 0], r_left[1, 0]],
            [r_left[2, 0], 0, -r_left[0, 0]],
            [-r_left[1, 0], r_left[0, 0], 0],
        ],
        dtype=np.float64,
    )
    a_mat[3:6, 3:6] = np.array(
        [
            [0, -r_right[2, 0], r_right[1, 0]],
            [r_right[2, 0], 0, -r_right[0, 0]],
            [-r_right[1, 0], r_right[0, 0], 0],
        ],
        dtype=np.float64,
    )

    b = np.vstack([f_center, m_center])
    forces, _, _, _ = np.linalg.lstsq(a_mat, b, rcond=None)
    f_left = forces[0:3].reshape(-1, 1)
    f_right = forces[3:6].reshape(-1, 1)

    result: Dict[str, _tb_Array] = {
        "left_linvel": v_left,
        "left_angvel": omega,
        "left_force": f_left,
        "right_linvel": v_right,
        "right_angvel": omega,
        "right_force": f_right,
        "center_linvel": v_center,
        "center_angvel": omega,
        "center_force": f_center,
        "center_torque": m_center,
        "rigid_body_center": runtime.rigid_body_center,
        "rigid_body_orientation": np.asarray(
            runtime.rigid_body_orientation, dtype=np.float64
        ),
    }

    for local_idx, global_idx in enumerate(runtime.active_hand_indices):
        hand_name = _tb_HAND_POS_KEYS[global_idx].replace("_pos", "")
        result[f"{hand_name}_vel"] = hand_velocities[local_idx]
        result[f"{hand_name}_pos"] = hand_positions_world[local_idx]

    return result


def _tb__maybe_print_ochs_world_velocity(
    runtime: _tb_PolicyRuntime,
    cfg: _tb_PolicyConfig,
    t: float,
    distributed_motion: Dict[str, _tb_Array],
) -> None:
    if not cfg.print_ochs_world_velocity:
        return

    last_t = runtime.last_ochs_print_time
    interval = max(float(cfg.ochs_print_interval), 0.0)
    if last_t is not None and (float(t) - float(last_t)) < interval:
        return

    runtime.last_ochs_print_time = float(t)

    center_linvel = np.asarray(
        distributed_motion["center_linvel"], dtype=np.float64
    ).reshape(3)
    center_angvel = np.asarray(
        distributed_motion["center_angvel"], dtype=np.float64
    ).reshape(3)
    left_linvel = np.asarray(
        distributed_motion["left_linvel"], dtype=np.float64
    ).reshape(3)
    left_angvel = np.asarray(
        distributed_motion["left_angvel"], dtype=np.float64
    ).reshape(3)
    right_linvel = np.asarray(
        distributed_motion["right_linvel"], dtype=np.float64
    ).reshape(3)
    right_angvel = np.asarray(
        distributed_motion["right_angvel"], dtype=np.float64
    ).reshape(3)

    print(
        "[model_based][OCHS->world] "
        f"t={float(t):.3f} "
        f"center_linvel={center_linvel.tolist()} "
        f"center_angvel={center_angvel.tolist()} "
        f"left_linvel={left_linvel.tolist()} "
        f"left_angvel={left_angvel.tolist()} "
        f"right_linvel={right_linvel.tolist()} "
        f"right_angvel={right_angvel.tolist()}"
    )


def _tb__assign_stiffness(
    runtime: _tb_PolicyRuntime,
    cfg: _tb_PolicyConfig,
    left_vel: _tb_Array,
    right_vel: _tb_Array,
) -> None:
    def build_diag(vel: _tb_Array) -> tuple[_tb_Array, _tb_Array]:
        dir_vec = np.asarray(vel, dtype=np.float64).reshape(-1)
        norm = np.linalg.norm(dir_vec)
        pos_high = cfg.pos_stiffness_high
        pos_low = cfg.pos_stiffness_low
        eye = np.eye(3, dtype=np.float64)
        if norm < 1e-6:
            diag = np.full(3, pos_low, dtype=np.float64)
        else:
            dir_unit = dir_vec / norm
            proj = np.outer(dir_unit, dir_unit)
            mat = eye * pos_low + (pos_high - pos_low) * proj
            diag = np.diag(mat)
        damp = 2.0 * np.sqrt(diag)
        return diag, damp

    left_diag, left_damp = build_diag(left_vel)
    right_diag, right_damp = build_diag(right_vel)

    runtime.pos_stiffness[0] = np.diag(left_diag).flatten()
    runtime.pos_damping[0] = np.diag(left_damp).flatten()
    if runtime.pos_stiffness.shape[0] > 1:
        runtime.pos_stiffness[1] = np.diag(right_diag).flatten()
        runtime.pos_damping[1] = np.diag(right_damp).flatten()


def _tb__integrate_pose_command(
    runtime: _tb_PolicyRuntime,
    cfg: _tb_PolicyConfig,
    distributed_motion: Dict[str, _tb_Array],
    dt: float,
) -> None:
    active_sites = set(_tb__active_site_indices_from_mode(runtime.active_hands_mode))
    for idx, prefix in [(0, "left"), (1, "right")]:
        if idx not in active_sites:
            runtime.wrench_command[idx, :] = 0.0
            continue

        linvel = distributed_motion[f"{prefix}_linvel"].reshape(-1)
        angvel = distributed_motion[f"{prefix}_angvel"].reshape(-1)
        force = distributed_motion[f"{prefix}_force"].reshape(-1)

        runtime.pose_command[idx, :3] = runtime.pose_command[idx, :3] + linvel * dt

        omega_norm = np.linalg.norm(angvel)
        if omega_norm > 1e-6:
            current_rot = R.from_rotvec(runtime.pose_command[idx, 3:])
            delta_angle = omega_norm * dt
            axis = angvel / omega_norm
            delta_rot = R.from_rotvec(axis * delta_angle)
            updated_rot = delta_rot * current_rot
            runtime.pose_command[idx, 3:] = updated_rot.as_rotvec()

        force_norm = np.linalg.norm(force)
        if force_norm > cfg.max_wrench_force:
            force = force * (cfg.max_wrench_force / force_norm)

        runtime.wrench_command[idx, :3] = force
        runtime.wrench_command[idx, 3:] = 0.0


def _tb__update_expected_ball_pos(
    runtime: _tb_PolicyRuntime,
    cfg: _tb_PolicyConfig,
    dt: float,
    state: Dict[str, _tb_Array],
) -> None:
    if runtime.expected_ball_pos is None:
        runtime.expected_ball_pos = np.asarray(
            state["ball_pos"], dtype=np.float64
        ).copy()
        return

    angular_velocity_vec = np.asarray(
        runtime.goal_rotate_axis, dtype=np.float64
    ) * float(runtime.goal_angular_velocity)
    delta_pos = (
        np.cross(
            angular_velocity_vec,
            np.array([0.0, 0.0, cfg.ball_radius], dtype=np.float64),
        )
        * dt
    )
    runtime.expected_ball_pos = runtime.expected_ball_pos + delta_pos


def _tb__update_delta_goal(
    runtime: _tb_PolicyRuntime, cfg: _tb_PolicyConfig, state: Dict[str, _tb_Array]
) -> None:
    if runtime.expected_ball_pos is None:
        return

    actual_ball_pos = np.asarray(state["ball_pos"], dtype=np.float64)
    position_error = actual_ball_pos[:2] - runtime.expected_ball_pos[:2]

    correction_xy = float(cfg.pid_kp) * position_error
    correction_vel_3d = np.array(
        [correction_xy[0], correction_xy[1], 0.0], dtype=np.float64
    )

    radius_vec = np.array([0.0, 0.0, cfg.ball_radius], dtype=np.float64)
    omega_correction = np.cross(radius_vec, correction_vel_3d) / (cfg.ball_radius**2)

    max_correction = 0.5
    omega_mag = np.linalg.norm(omega_correction)
    if omega_mag > max_correction:
        omega_correction = omega_correction * (max_correction / omega_mag)

    delta_goal = -omega_correction
    if abs(runtime.goal_angular_velocity) > 1e-6:
        target_dir = np.asarray(runtime.goal_rotate_axis, dtype=np.float64)
        target_norm = np.linalg.norm(target_dir)
        if target_norm > 1e-9:
            target_dir = target_dir / target_norm
            target_dir = target_dir * np.sign(runtime.goal_angular_velocity)
            parallel = float(np.dot(delta_goal, target_dir))
            if parallel > 0.0:
                delta_goal = delta_goal - parallel * target_dir

    runtime.delta_goal_angular_velocity = delta_goal


def _tb__build_command_matrix(
    runtime: _tb_PolicyRuntime,
    measured_wrenches: Dict[str, _tb_Array],
    site_names: Tuple[str, str],
) -> _tb_Array:
    command_matrix = np.zeros((2, COMMAND_LAYOUT.width), dtype=np.float64)
    command_matrix[:, COMMAND_LAYOUT.position] = runtime.pose_command[:, :3]
    command_matrix[:, COMMAND_LAYOUT.orientation] = runtime.pose_command[:, 3:6]
    command_matrix[:, COMMAND_LAYOUT.kp_pos] = runtime.pos_stiffness
    command_matrix[:, COMMAND_LAYOUT.kp_rot] = runtime.rot_stiffness
    command_matrix[:, COMMAND_LAYOUT.kd_pos] = runtime.pos_damping
    command_matrix[:, COMMAND_LAYOUT.kd_rot] = runtime.rot_damping

    for idx, site_name in enumerate(site_names):
        wrench = np.asarray(
            measured_wrenches.get(site_name, np.zeros(6)), dtype=np.float64
        )
        command_matrix[idx, COMMAND_LAYOUT.measured_force] = wrench[:3]
        command_matrix[idx, COMMAND_LAYOUT.measured_torque] = wrench[3:]
        command_matrix[idx, COMMAND_LAYOUT.force] = runtime.wrench_command[idx, :3]
        command_matrix[idx, COMMAND_LAYOUT.torque] = runtime.wrench_command[idx, 3:]

    return command_matrix


def _tb__run_compliance_step(
    controller: ComplianceController,
    data: mujoco.MjData,
    t: float,
    command_matrix: _tb_Array,
    target_motor_pos: _tb_Array,
    measured_wrenches: Dict[str, _tb_Array],
    site_names: Tuple[str, str],
) -> tuple[_tb_Array, Optional[ComplianceState]]:
    wrenches_out, state_ref = controller.step(
        command_matrix=command_matrix.astype(np.float32),
        motor_torques=np.asarray(data.actuator_force, dtype=np.float32),
        qpos=np.asarray(data.qpos, dtype=np.float32),
    )

    next_target = target_motor_pos
    if state_ref is not None:
        state_ref_motor_pos = np.asarray(state_ref.motor_pos, dtype=np.float64)
        next_target = np.asarray(target_motor_pos, dtype=np.float64).copy()
        controlled_actuators = np.asarray(
            controller.compliance_ref.actuator_indices, dtype=np.int32
        )
        next_target[controlled_actuators] = state_ref_motor_pos[controlled_actuators]

    for site in site_names:
        wrench = wrenches_out.get(site)
        if wrench is not None:
            measured_wrenches[site] = np.asarray(wrench, dtype=np.float64)

    return next_target, state_ref


def _tb__sync_compliance_state_to_current_pose(
    controller: ComplianceController,
    data: mujoco.MjData,
    motor_pos: _tb_Array,
) -> None:
    compliance_ref = controller.compliance_ref
    if compliance_ref is None:
        return

    ref_state = compliance_ref.get_default_state()
    site_ids = compliance_ref.site_ids
    x_ref = np.zeros((len(site_ids), 6), dtype=np.float32)
    for idx, site_id in enumerate(site_ids):
        pos = np.asarray(data.site_xpos[site_id], dtype=np.float32).copy()
        rotmat = (
            np.asarray(data.site_xmat[site_id], dtype=np.float32).reshape(3, 3).copy()
        )
        rotvec = R.from_matrix(rotmat).as_rotvec().astype(np.float32)
        x_ref[idx, :3] = pos
        x_ref[idx, 3:] = rotvec

    ref_state.x_ref = x_ref.copy()
    ref_state.v_ref = np.zeros_like(x_ref)
    ref_state.a_ref = np.zeros_like(x_ref)
    ref_state.qpos = np.asarray(data.qpos, dtype=np.float32).copy()
    ref_state.motor_pos = np.asarray(motor_pos, dtype=np.float32).copy()
    controller._last_state = ref_state


def _tb__resolve_mocap_target_ids(
    model: mujoco.MjModel,
) -> tuple[Optional[int], Optional[int]]:
    left_body_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "left_hand_target"
    )
    right_body_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "right_hand_target"
    )

    left_mocap_id: Optional[int] = None
    right_mocap_id: Optional[int] = None

    if left_body_id >= 0:
        left_mocap_raw = int(model.body_mocapid[left_body_id])
        if left_mocap_raw >= 0:
            left_mocap_id = left_mocap_raw
    if right_body_id >= 0:
        right_mocap_raw = int(model.body_mocapid[right_body_id])
        if right_mocap_raw >= 0:
            right_mocap_id = right_mocap_raw

    if left_mocap_id is None or right_mocap_id is None:
        print(
            "[model_based] Warning: mocap targets not found "
            "(expected bodies: left_hand_target/right_hand_target)."
        )
    else:
        print(
            "[model_based] Mocap targets ready "
            f"(left={left_mocap_id}, right={right_mocap_id})."
        )

    return left_mocap_id, right_mocap_id


def _tb__update_mocap_targets_from_state_ref(
    data: mujoco.MjData,
    left_mocap_id: Optional[int],
    right_mocap_id: Optional[int],
    state_ref: Optional[ComplianceState],
) -> None:
    if state_ref is None:
        return

    x_ref_arr = np.asarray(state_ref.x_ref, dtype=np.float64)
    if x_ref_arr.ndim != 2 or x_ref_arr.shape[1] < 6:
        return

    if left_mocap_id is not None and x_ref_arr.shape[0] > 0:
        pos = x_ref_arr[0, :3]
        rotvec = x_ref_arr[0, 3:6]
        quat_xyzw = R.from_rotvec(rotvec).as_quat()
        quat_wxyz = np.array(
            [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]],
            dtype=np.float64,
        )
        data.mocap_pos[left_mocap_id] = pos
        data.mocap_quat[left_mocap_id] = quat_wxyz

    if right_mocap_id is not None and x_ref_arr.shape[0] > 1:
        pos = x_ref_arr[1, :3]
        rotvec = x_ref_arr[1, 3:6]
        quat_xyzw = R.from_rotvec(rotvec).as_quat()
        quat_wxyz = np.array(
            [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]],
            dtype=np.float64,
        )
        data.mocap_pos[right_mocap_id] = pos
        data.mocap_quat[right_mocap_id] = quat_wxyz
