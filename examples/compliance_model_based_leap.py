from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Any, Dict, Optional, Tuple

import mujoco
import numpy as np
import numpy.typing as npt
import yaml
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

from examples.compliance import CompliancePolicy
from hybrid_servo.algorithm.ochs import solve_ochs
from hybrid_servo.algorithm.solvehfvc import transform_hfvc_to_global
from hybrid_servo.tasks.multi_finger_ochs import (
    compute_hfvc_inputs,
    generate_constraint_jacobian,
    get_center_state,
)
from hybrid_servo.utils import find_repo_root, sync_compliance_state_to_current_pose
from minimalist_compliance_control.compliance_ref import COMMAND_LAYOUT
from minimalist_compliance_control.controller import (
    ComplianceController,
    ControllerConfig,
    RefConfig,
)
from minimalist_compliance_control.utils import (
    KeyboardControlReceiver,
    ensure_matrix,
    get_damping_matrix,
)
from minimalist_compliance_control.wrench_estimation import WrenchEstimateConfig

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
        "min_normal_force_translation": 3.0,
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


def _init(
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
    policy.wrenches_by_site = {}
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
    policy.initial_pose_command = _build_pose_command(policy, INIT_POSE_DATA)
    policy.target_pose_command = _build_pose_command(policy, TARGET_POSE_DATA)
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
    policy.close_gains = _compute_force_and_stiffness(
        policy, policy.target_pose_command
    )

    policy.forward_traj = _build_command_trajectory(
        policy,
        policy.initial_pose_command,
        policy.target_pose_command,
        policy.open_gains,
        policy.close_gains,
    )
    policy.backward_traj = _build_command_trajectory(
        policy,
        policy.target_pose_command,
        policy.initial_pose_command,
        policy.close_gains,
        policy.open_gains,
    )

    policy.active_traj = None
    policy.traj_start_time = 0.0

    # Initialize stiffness/wrench targets from first trajectory sample.
    _apply_traj_sample(policy, policy.forward_traj, 0)

    policy.phase = "close"
    policy.traj_set = False
    policy.object_body_name = "manip_object"
    policy.object_qpos_adr = None
    policy.object_qvel_adr = None
    policy.close_stage = "to_init"
    policy.jacobian_constraint = generate_constraint_jacobian()
    policy.target_rotation_angvel = np.array([0.0, 0.0, 0.0])
    policy.target_rotation_linvel = np.array([0.03, 0.0, 0.0])
    policy.last_angvel_flip_time = None
    policy.pos_kp = 300  # High stiffness for anisotropic fingers.
    policy.force_kp = 200  # Low stiffness for anisotropic fingers.
    policy.rot_kp = 20
    policy.baseline_tip_rot = {}
    policy.interval = 1.5
    # Store last contact position for each fingertip (used during close phase)
    policy.last_contact_pos = {tip: None for tip in policy.wrench_site_names}

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
    policy.last_integration_time = None

    # Mode switching state
    policy.mode_switch_pending = False
    policy.target_mode = None
    policy.return_to_zero_tolerance = 0.003
    # Hold still briefly after close->rotate contact trigger.
    policy.rotate_start_wait = 0.5
    policy.rotate_phase_start_time = None
    policy.rotate_wait_released = False

    # Stdin receiver for keyboard control.
    policy.control_receiver = None
    try:
        policy.control_receiver = KeyboardControlReceiver(
            valid_commands={"c", "r"},
            name="LeapRotateCompliance",
            help_labels={"c": "reverse", "r": "switch mode"},
        )
    except Exception as exc:
        policy.control_receiver = None
        print(f"[LeapRotateCompliance] Warning: control receiver disabled: {exc}")

    # Close-stage thumb debug (disabled by default).
    policy.debug_close_thumb = False
    policy.debug_close_thumb_pos_jump_threshold = 0.01
    policy.debug_close_thumb_rot_jump_threshold = float(np.deg2rad(8.0))

    # Rotate-transition debug (disabled by default).
    policy.debug_rotate_transition = False
    policy.debug_rotate_transition_frames = 40
    policy.debug_rotate_transition_countdown = 0
    policy._debug_rotate_enter_thumb_cmd_pos = None
    policy._debug_rotate_enter_thumb_site_pos = None
    policy._debug_last_thumb_cmd_pos = None
    policy._debug_last_thumb_site_pos = None

    # Mode-switch debug (disabled by default).
    policy.debug_mode_switch = False


def _build_pose_command(
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


def _compute_force_and_stiffness(
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


def _build_command_trajectory(
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


def _apply_traj_sample(policy, traj: Dict[str, np.ndarray], idx: int) -> None:
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


def _start_command_trajectory(
    policy, traj: Dict[str, np.ndarray], time_curr: Optional[float]
) -> None:
    policy.active_traj = traj
    policy.traj_start_time = float(time_curr if time_curr is not None else 0.0)
    _apply_traj_sample(policy, traj, 0)


def _advance_command_trajectory(policy, time_curr: float) -> None:
    if policy.active_traj is None:
        return
    times = policy.active_traj["time"]
    elapsed = time_curr - policy.traj_start_time
    idx = int(np.searchsorted(times, elapsed, side="right") - 1)
    _apply_traj_sample(policy, policy.active_traj, idx)
    if elapsed >= float(times[-1]):
        policy.active_traj = None


def __active_traj_idx(policy, time_curr: float) -> int:
    if policy.active_traj is None:
        return -1
    times = policy.active_traj["time"]
    elapsed = time_curr - policy.traj_start_time
    idx = int(np.searchsorted(times, elapsed, side="right") - 1)
    return int(np.clip(idx, 0, times.shape[0] - 1))


def __rot_error_angle(rotvec_from: np.ndarray, rotvec_to: np.ndarray) -> float:
    rot_from = R.from_rotvec(np.asarray(rotvec_from, dtype=np.float64).reshape(3))
    rot_to = R.from_rotvec(np.asarray(rotvec_to, dtype=np.float64).reshape(3))
    return float(np.linalg.norm((rot_to * rot_from.inv()).as_rotvec()))


def __world_vec_to_object_frame(
    vec_world: np.ndarray, obj_quat_wxyz: np.ndarray
) -> np.ndarray:
    obj_rot = R.from_quat(
        np.asarray(obj_quat_wxyz, dtype=np.float64).reshape(4), scalar_first=True
    )
    vec = np.asarray(vec_world, dtype=np.float64).reshape(3)
    return obj_rot.inv().apply(vec)


def __lat_vert_components(vec: np.ndarray) -> tuple[float, float]:
    vec_arr = np.asarray(vec, dtype=np.float64).reshape(3)
    return float(np.linalg.norm(vec_arr[:2])), float(vec_arr[2])


def _debug_close_thumb(
    policy,
    time_curr: float,
    thumb_cmd_before: np.ndarray,
    stage_before: str,
    traj_idx_before: int,
) -> None:
    if not getattr(policy, "debug_close_thumb", False):
        return
    if "th_tip" not in policy.wrench_site_names:
        return
    thumb_idx = policy.wrench_site_names.index("th_tip")
    thumb_cmd_after = np.asarray(
        policy.pose_command[thumb_idx], dtype=np.float64
    ).copy()
    dpos = float(np.linalg.norm(thumb_cmd_after[:3] - thumb_cmd_before[:3]))
    drot = __rot_error_angle(thumb_cmd_before[3:6], thumb_cmd_after[3:6])
    sign_flip = bool(np.dot(thumb_cmd_before[3:6], thumb_cmd_after[3:6]) < 0.0)
    traj_idx_after = __active_traj_idx(policy, time_curr)

    site_pos_err = float("nan")
    site_rot_err = float("nan")
    thumb_site_id = mujoco.mj_name2id(
        policy.wrench_sim.model, mujoco.mjtObj.mjOBJ_SITE, "th_tip"
    )
    if thumb_site_id >= 0:
        site_pos = np.asarray(
            policy.wrench_sim.data.site_xpos[thumb_site_id], dtype=np.float64
        )
        site_rotvec = R.from_matrix(
            np.asarray(
                policy.wrench_sim.data.site_xmat[thumb_site_id], dtype=np.float64
            ).reshape(3, 3)
        ).as_rotvec()
        site_pos_err = float(np.linalg.norm(thumb_cmd_after[:3] - site_pos))
        site_rot_err = __rot_error_angle(site_rotvec, thumb_cmd_after[3:6])

    reasons: list[str] = []
    if stage_before != policy.close_stage:
        reasons.append("stage_change")
    if (
        traj_idx_before >= 0
        and traj_idx_after >= 0
        and traj_idx_after < traj_idx_before
    ):
        reasons.append("traj_restart")
    if sign_flip:
        reasons.append("rotvec_sign_flip")
    if dpos > float(getattr(policy, "debug_close_thumb_pos_jump_threshold", 0.01)):
        reasons.append("pos_jump")
    if drot > float(
        getattr(policy, "debug_close_thumb_rot_jump_threshold", np.deg2rad(8.0))
    ):
        reasons.append("ori_jump")
    reason_text = ",".join(reasons) if reasons else "none"

    print(
        "[leap_debug][close_thumb] "
        f"t={time_curr:.3f} "
        f"stage={stage_before}->{policy.close_stage} "
        f"traj_idx={traj_idx_before}->{traj_idx_after} "
        f"cmd_pos={np.round(thumb_cmd_after[:3], 6).tolist()} "
        f"dpos={dpos:.6f} "
        f"drot_deg={np.degrees(drot):.3f} "
        f"site_pos_err={site_pos_err:.6f} "
        f"site_rot_err_deg={np.degrees(site_rot_err):.3f} "
        f"sign_flip={int(sign_flip)} "
        f"reasons={reason_text}"
    )


def _check_control_command(policy) -> str | None:
    """Check for keyboard commands via stdin receiver."""
    if policy.control_receiver is None:
        return None

    msg = policy.control_receiver.poll_command()
    if msg is None or msg.command is None:
        return None

    cmd = str(msg.command).strip().lower()
    return cmd if cmd in ("c", "r") else None


def _update_goal(policy, time_curr: float) -> None:
    """Update target velocities based on keyboard commands and threshold.

    Commands:
    - 'c': Reverse current target (flip sign of angvel or linvel)
    - 'r': Switch between rotation mode and translation mode

    Threshold logic:
    - Integrates angvel/linvel to track relative position from initial state
    - When reaching threshold, sets velocity to 0
    - Pressing 'c' reverses direction to move toward reverse threshold
    """
    debug_mode_switch = bool(getattr(policy, "debug_mode_switch", False))

    # Initialize integration timer on first call during rotate phase
    if policy.last_integration_time is None:
        policy.last_integration_time = time_curr

    # Calculate dt for integration
    dt = time_curr - policy.last_integration_time
    policy.last_integration_time = time_curr
    prev_integrated_angle = float(policy.integrated_angle)
    prev_integrated_position = float(policy.integrated_position)

    # Dwell after close->rotate contact trigger: zero commanded motion for a short window.
    if not bool(getattr(policy, "rotate_wait_released", True)):
        start_time = getattr(policy, "rotate_phase_start_time", None)
        wait_duration = float(max(getattr(policy, "rotate_start_wait", 0.0), 0.0))
        if start_time is not None and (time_curr - float(start_time)) < wait_duration:
            policy.target_rotation_angvel = np.array([0.0, 0.0, 0.0])
            policy.target_rotation_linvel = np.array([0.0, 0.0, 0.0])
            return
        policy.rotate_wait_released = True
        if policy.control_mode == "rotation":
            policy.target_rotation_angvel = np.array(
                [0.0, 0.0, policy.rotation_angvel_magnitude]
            )
            policy.target_rotation_linvel = np.array([0.0, 0.0, 0.0])
        else:
            policy.target_rotation_linvel = np.array(
                [policy.translation_linvel_magnitude, 0.0, 0.0]
            )
            policy.target_rotation_angvel = np.array([0.0, 0.0, 0.0])

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
    cmd = _check_control_command(
        policy,
    )

    if cmd == "c":
        _apply_reverse_command(
            policy,
        )

    elif cmd == "r":
        _request_mode_switch(
            policy,
        )

    if policy.mode_switch_pending:
        at_zero = False
        direction = 0.0
        zero_crossed = False
        if policy.control_mode == "rotation":
            zero_crossed = (
                prev_integrated_angle * float(policy.integrated_angle) <= 0.0
                and abs(prev_integrated_angle - float(policy.integrated_angle)) > 0.0
            )
            at_zero = (
                abs(policy.integrated_angle) < policy.return_to_zero_tolerance
                or zero_crossed
            )
            if not at_zero:
                direction = -np.sign(policy.integrated_angle)
                if direction == 0:
                    direction = 1.0
                policy.target_rotation_angvel = np.array(
                    [0.0, 0.0, direction * policy.rotation_angvel_magnitude]
                )
        else:
            zero_crossed = (
                prev_integrated_position * float(policy.integrated_position) <= 0.0
                and abs(prev_integrated_position - float(policy.integrated_position))
                > 0.0
            )
            at_zero = (
                abs(policy.integrated_position) < policy.return_to_zero_tolerance
                or zero_crossed
            )
            if not at_zero:
                direction = -np.sign(policy.integrated_position)
                if direction == 0:
                    direction = 1.0
                policy.target_rotation_linvel = np.array(
                    [direction * policy.translation_linvel_magnitude, 0.0, 0.0]
                )

        if debug_mode_switch:
            if policy.control_mode == "rotation":
                metric = float(policy.integrated_angle)
                cmd_vel = float(policy.target_rotation_angvel[2])
            else:
                metric = float(policy.integrated_position)
                cmd_vel = float(policy.target_rotation_linvel[0])
            print(
                "[leaphand][mode_switch] "
                f"t={float(time_curr):.3f} "
                "event=pending_check "
                f"mode={policy.control_mode} "
                f"target_mode={policy.target_mode} "
                f"metric={metric:.6f} "
                f"tol={float(policy.return_to_zero_tolerance):.6f} "
                f"cmd_vel={cmd_vel:.6f} "
                f"direction={float(direction):.1f} "
                f"zero_crossed={int(zero_crossed)} "
                f"at_zero={int(at_zero)}"
            )

        if at_zero:
            policy.mode_switch_pending = False
            # Reset pose_command to target pose to avoid drift
            policy.pose_command = policy.target_pose_command.copy()
            print(
                "[LeapRotateCompliance] Reset pose_command to target_pose_command to prevent drift"
            )
            policy.integrated_angle = 0.0
            policy.integrated_position = 0.0

            if policy.target_mode == "translation":
                policy.control_mode = "translation"
                policy.target_rotation_angvel = np.array([0.0, 0.0, 0.0])
                policy.target_rotation_linvel = np.array(
                    [policy.translation_linvel_magnitude, 0.0, 0.0]
                )
                print(
                    f"[LeapRotateCompliance] Switched to TRANSLATION mode: linvel = {policy.target_rotation_linvel}"
                )
            else:
                policy.control_mode = "rotation"
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
            if debug_mode_switch:
                print(
                    "[leaphand][mode_switch] "
                    f"t={float(time_curr):.3f} "
                    "event=threshold_stop "
                    f"mode={policy.control_mode} "
                    f"metric={float(current_metric):.6f} "
                    f"active_threshold={float(active_threshold):.6f} "
                    f"moving_outward={int(moving_outward)} "
                    f"just_reached_limit={int(just_reached_limit)} "
                    f"auto_counter={int(policy.auto_switch_counter)}"
                )
            _auto_switch_target(policy, just_reached_limit)

    if (not moving_outward) or (abs(current_metric) < abs(active_threshold) * 0.98):
        policy.limit_reached_flag = False


def _apply_reverse_command(policy) -> None:
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


def _request_mode_switch(policy) -> None:
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


def _auto_switch_target(policy, just_reached_limit: bool) -> None:
    """Auto-switch target when reaching limit: reverse -> mode_switch -> reverse -> ..."""
    if not policy.auto_switch_target_enabled:
        return
    if not just_reached_limit:
        return

    action_type = policy.auto_switch_counter % 2  # 0=reverse, 1=mode_switch
    if bool(getattr(policy, "debug_mode_switch", False)):
        action_name = "reverse" if action_type == 0 else "mode_switch"
        print(
            "[leaphand][mode_switch] "
            f"t={float(policy.wrench_sim.data.time):.3f} "
            "event=auto_switch "
            f"action={action_name} "
            f"counter={int(policy.auto_switch_counter)}"
        )
    if action_type == 0:
        _apply_reverse_command(
            policy,
        )
    else:
        _request_mode_switch(
            policy,
        )
    policy.auto_switch_counter += 1


def __ensure_object_detected(policy) -> None:
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


def _forward_object_to_init(policy, sim_name: str = "sim") -> None:
    """Immediately place object at policy init pose and forward once."""
    __ensure_object_detected(
        policy,
    )
    _capture_object_init(
        policy,
    )
    _fix_object(policy, policy.wrench_sim, sim_name=sim_name)
    mujoco.mj_forward(policy.wrench_sim.model, policy.wrench_sim.data)


def _step(
    policy,
    time_curr: float,
    wrenches_by_site: Optional[Dict[str, np.ndarray]] = None,
    system_state: Optional[Dict[str, np.ndarray]] = None,
    *,
    sim_name: str = "sim",
    is_real_world: bool = False,
) -> Dict[str, np.ndarray | str]:
    thumb_cmd_before: Optional[np.ndarray] = None
    stage_before = str(policy.close_stage)
    traj_idx_before = -1
    if wrenches_by_site is not None:
        policy.wrenches_by_site = {
            key: np.asarray(val, dtype=np.float32)
            for key, val in wrenches_by_site.items()
        }

    __ensure_object_detected(
        policy,
    )

    if time_curr < policy.prep_duration:
        _capture_object_init(
            policy,
        )
        if not is_real_world:
            _fix_object(policy, policy.wrench_sim, sim_name=sim_name)
        return _get_outputs(
            policy,
        )

    if policy.phase == "close":
        if "th_tip" in policy.wrench_site_names:
            thumb_idx = policy.wrench_site_names.index("th_tip")
            thumb_cmd_before = np.asarray(
                policy.pose_command[thumb_idx], dtype=np.float64
            ).copy()
            traj_idx_before = __active_traj_idx(policy, time_curr)
        _capture_object_init(
            policy,
        )
        if not is_real_world:
            _fix_object(policy, policy.wrench_sim, sim_name=sim_name)
        if policy.close_stage == "to_init":
            if policy.active_traj is None:
                traj = _build_command_trajectory(
                    policy,
                    policy.pose_command.copy(),
                    policy.initial_pose_command,
                    policy.open_gains,
                    policy.open_gains,
                )
                _start_command_trajectory(policy, traj, time_curr)
        elif policy.close_stage == "to_target":
            if policy.active_traj is None and not policy.traj_set:
                _start_command_trajectory(policy, policy.forward_traj, time_curr)
                policy.traj_set = True
            elif policy.active_traj is None:
                _check_switch_phase(policy, time_curr=time_curr)
        _advance_command_trajectory(policy, time_curr)
    elif policy.phase == "rotate":
        # Update goal with keyboard commands and threshold checking
        _update_goal(policy, time_curr)

        # Handle rotation action
        _handle_rotate_action(policy, system_state)
        _check_switch_phase(policy, time_curr=time_curr)

    if (
        policy.phase == "close"
        and policy.close_stage == "to_init"
        and policy.active_traj is None
    ):
        policy.close_stage = "to_target"
        policy.traj_set = False

    if thumb_cmd_before is not None:
        _debug_close_thumb(
            policy,
            time_curr,
            thumb_cmd_before,
            stage_before,
            traj_idx_before,
        )
    return _get_outputs(
        policy,
    )


def _get_outputs(policy) -> Dict[str, np.ndarray | str]:
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


def _assign_stiffness(
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


def _set_phase(policy, phase: str) -> None:
    """Update phase and reset trajectory flag whenever phase changes."""
    if policy.phase != phase:
        policy.phase = phase
        policy.traj_set = False


def _check_switch_phase(policy, time_curr: Optional[float] = None) -> None:
    """Switch from close to rotate once all fingertips have sufficient contact force."""
    if policy.phase == "close":
        has_contact = _check_all_fingertips_contact(
            policy,
        )
        if has_contact:
            _freeze_pose_to_current(
                policy,
            )
            _capture_baseline_tip_rot(
                policy,
            )
            policy.rotate_phase_start_time = float(
                time_curr if time_curr is not None else policy.wrench_sim.data.time
            )
            policy.rotate_wait_released = False
            policy.target_rotation_linvel = np.array([0.0, 0.0, 0.0])
            policy.target_rotation_angvel = np.array([0.0, 0.0, 0.0])
            if getattr(policy, "debug_rotate_transition", False):
                policy.debug_rotate_transition_countdown = int(
                    getattr(policy, "debug_rotate_transition_frames", 40)
                )
                enter_state = _get_system_state(policy)
                obj_pos = np.asarray(enter_state["sliding_cube_pos"], dtype=np.float64)
                obj_quat = np.asarray(
                    enter_state["sliding_cube_quat"], dtype=np.float64
                )
                if "th_tip" in policy.wrench_site_names:
                    thumb_idx = policy.wrench_site_names.index("th_tip")
                    thumb_cmd = np.asarray(
                        policy.pose_command[thumb_idx], dtype=np.float64
                    )
                    policy._debug_rotate_enter_thumb_cmd_pos = thumb_cmd[:3].copy()
                    policy._debug_last_thumb_cmd_pos = thumb_cmd[:3].copy()
                    policy._debug_rotate_enter_thumb_site_pos = None
                    policy._debug_last_thumb_site_pos = None
                    thumb_site_id = mujoco.mj_name2id(
                        policy.wrench_sim.model, mujoco.mjtObj.mjOBJ_SITE, "th_tip"
                    )
                    if thumb_site_id >= 0:
                        thumb_site_pos = np.asarray(
                            policy.wrench_sim.data.site_xpos[thumb_site_id],
                            dtype=np.float64,
                        )
                        thumb_site_rot = R.from_matrix(
                            np.asarray(
                                policy.wrench_sim.data.site_xmat[thumb_site_id],
                                dtype=np.float64,
                            ).reshape(3, 3)
                        ).as_rotvec()
                        pos_err = float(np.linalg.norm(thumb_cmd[:3] - thumb_site_pos))
                        rot_err = __rot_error_angle(thumb_site_rot, thumb_cmd[3:6])
                        policy._debug_rotate_enter_thumb_site_pos = (
                            thumb_site_pos.copy()
                        )
                        policy._debug_last_thumb_site_pos = thumb_site_pos.copy()
                        thumb_cmd_rel_obj = __world_vec_to_object_frame(
                            thumb_cmd[:3] - obj_pos, obj_quat
                        )
                        thumb_site_rel_obj = __world_vec_to_object_frame(
                            thumb_site_pos - obj_pos, obj_quat
                        )
                        cmd_lat, cmd_vert = __lat_vert_components(thumb_cmd_rel_obj)
                        site_lat, site_vert = __lat_vert_components(thumb_site_rel_obj)
                        print(
                            "[leap_debug][rotate_enter] "
                            f"t={float(policy.wrench_sim.data.time):.3f} "
                            f"mode={policy.control_mode} "
                            f"target_linvel={np.asarray(policy.target_rotation_linvel, dtype=np.float64).tolist()} "
                            f"target_angvel={np.asarray(policy.target_rotation_angvel, dtype=np.float64).tolist()} "
                            f"obj_pos={obj_pos.tolist()} "
                            f"obj_quat_wxyz={obj_quat.tolist()} "
                            f"thumb_cmd_rel_obj={thumb_cmd_rel_obj.tolist()} "
                            f"thumb_site_rel_obj={thumb_site_rel_obj.tolist()} "
                            f"thumb_cmd_rel_obj_lat={cmd_lat:.6f} "
                            f"thumb_cmd_rel_obj_vert={cmd_vert:.6f} "
                            f"thumb_site_rel_obj_lat={site_lat:.6f} "
                            f"thumb_site_rel_obj_vert={site_vert:.6f} "
                            f"thumb_pos_err={pos_err:.6f} "
                            f"thumb_rot_err_deg={np.degrees(rot_err):.3f}"
                        )
            _set_phase(policy, "rotate")
    else:
        return


def _check_all_fingertips_contact(policy) -> bool:
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


def _capture_baseline_tip_rot(policy) -> None:
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


def _freeze_pose_to_current(policy) -> None:
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


def _capture_object_init(policy) -> None:
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


def _fix_object(policy, sim: Any, sim_name: str = "sim") -> None:
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


def _apply_pd(policy, sim: Any) -> None:
    # Kept for API parity with toddlerbot policy; no-op in standalone mode.
    return


def _get_system_state(policy) -> Dict[str, np.ndarray]:
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


def _get_target_vel(policy, state):
    p_thumb_obj = state["fix_traj_pos"] - state["sliding_cube_pos"]
    cross_term = np.cross(policy.target_rotation_angvel, p_thumb_obj)
    thumb_linvel = policy.target_rotation_linvel + cross_term
    thumb_angvel = np.zeros(3)

    v_obj_goal = np.cross(policy.target_rotation_angvel - thumb_angvel, -p_thumb_obj)
    omega_obj_goal = policy.target_rotation_angvel - thumb_angvel

    return (
        v_obj_goal,
        omega_obj_goal,
        thumb_linvel,
        thumb_angvel,
        p_thumb_obj,
        cross_term,
    )


def _handle_rotate_action(policy, state: Optional[Dict[str, np.ndarray]] = None):
    if state is None:
        state = _get_system_state(
            policy,
        )
    (
        target_linvel,
        target_angvel,
        thumb_linvel,
        thumb_angvel,
        p_thumb_obj,
        cross_term,
    ) = _get_target_vel(policy, state)

    if getattr(policy, "debug_rotate_transition_countdown", 0) > 0:
        print(
            "[leap_debug][rotate_target] "
            f"t={float(policy.wrench_sim.data.time):.3f} "
            f"mode={policy.control_mode} "
            f"target_lin_base={np.asarray(policy.target_rotation_linvel, dtype=np.float64).tolist()} "
            f"target_ang={np.asarray(policy.target_rotation_angvel, dtype=np.float64).tolist()} "
            f"p_thumb_obj={np.asarray(p_thumb_obj, dtype=np.float64).tolist()} "
            f"cross_term={np.asarray(cross_term, dtype=np.float64).tolist()} "
            f"thumb_linvel={np.asarray(thumb_linvel, dtype=np.float64).tolist()} "
            f"v_obj_goal={np.asarray(target_linvel, dtype=np.float64).tolist()}"
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
        if getattr(policy, "debug_rotate_transition_countdown", 0) > 0:
            print(
                "[leap_debug][rotate_target] "
                f"t={float(policy.wrench_sim.data.time):.3f} hfvc_solution=None"
            )
        return

    # Stash per-frame target decomposition for downstream debug in distribute step.
    policy._debug_target_linvel = np.asarray(target_linvel, dtype=np.float64).copy()
    policy._debug_target_angvel = np.asarray(target_angvel, dtype=np.float64).copy()
    policy._debug_thumb_linvel = np.asarray(thumb_linvel, dtype=np.float64).copy()
    policy._debug_thumb_angvel = np.asarray(thumb_angvel, dtype=np.float64).copy()
    policy._debug_p_thumb_obj = np.asarray(p_thumb_obj, dtype=np.float64).copy()
    policy._debug_cross_term = np.asarray(cross_term, dtype=np.float64).copy()
    _distribute_action(policy, hfvc_solution, thumb_linvel, thumb_angvel, state)


def _ensure_rotvec_continuity(
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


def _distribute_action(
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
    _assign_stiffness(policy, v_left, v_right)

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
    if "th_tip" in policy.wrench_site_names:
        thumb_idx = policy.wrench_site_names.index("th_tip")
        thumb_pose_before = np.asarray(
            policy.pose_command[thumb_idx], dtype=np.float64
        ).copy()
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
        # new_rotvec = _ensure_rotvec_continuity(policy, old_rotvec, new_rotvec)
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
    # new_rotvec_thumb = _ensure_rotvec_continuity(
    #     policy,
    #     old_rotvec_thumb,
    #     new_rotvec_thumb,
    # )
    # print(new_rotvec_thumb)

    # integrated_angle = R.from_rotvec(policy.integrated_angle_thumb)
    # policy.integrated_angle_thumb = (rot_increment * integrated_angle).as_rotvec()
    # print(policy.integrated_angle_thumb)
    policy.pose_command[thumb_idx, 3:6] = new_rotvec_thumb

    if getattr(policy, "debug_rotate_transition_countdown", 0) > 0:
        thumb_pose_after = np.asarray(
            policy.pose_command[thumb_idx], dtype=np.float64
        ).copy()
        thumb_pos_delta = thumb_pose_after[:3] - thumb_pose_before[:3]
        obj_pos = np.asarray(state["sliding_cube_pos"], dtype=np.float64)
        obj_quat = np.asarray(state["sliding_cube_quat"], dtype=np.float64)
        thumb_pos_delta_obj = __world_vec_to_object_frame(thumb_pos_delta, obj_quat)
        thumb_delta_lat, thumb_delta_vert = __lat_vert_components(thumb_pos_delta_obj)

        thumb_linvel_dbg = np.asarray(
            getattr(policy, "_debug_thumb_linvel", thumb_linvel), dtype=np.float64
        )
        thumb_linvel_obj = __world_vec_to_object_frame(thumb_linvel_dbg, obj_quat)
        thumb_vel_lat, thumb_vel_vert = __lat_vert_components(thumb_linvel_obj)

        prev_cmd_pos = getattr(policy, "_debug_last_thumb_cmd_pos", None)
        cmd_step_delta = (
            thumb_pose_after[:3] - np.asarray(prev_cmd_pos, dtype=np.float64)
            if prev_cmd_pos is not None
            else np.zeros(3, dtype=np.float64)
        )
        cmd_step_delta_obj = __world_vec_to_object_frame(cmd_step_delta, obj_quat)
        policy._debug_last_thumb_cmd_pos = thumb_pose_after[:3].copy()

        thumb_site_id = mujoco.mj_name2id(
            policy.wrench_sim.model, mujoco.mjtObj.mjOBJ_SITE, "th_tip"
        )
        site_step_delta = np.zeros(3, dtype=np.float64)
        site_step_delta_obj = np.zeros(3, dtype=np.float64)
        site_step_lat = 0.0
        site_step_vert = 0.0
        if thumb_site_id >= 0:
            thumb_site_pos = np.asarray(
                policy.wrench_sim.data.site_xpos[thumb_site_id], dtype=np.float64
            )
            prev_site_pos = getattr(policy, "_debug_last_thumb_site_pos", None)
            if prev_site_pos is not None:
                site_step_delta = thumb_site_pos - np.asarray(
                    prev_site_pos, dtype=np.float64
                )
                site_step_delta_obj = __world_vec_to_object_frame(
                    site_step_delta, obj_quat
                )
                site_step_lat, site_step_vert = __lat_vert_components(
                    site_step_delta_obj
                )
            policy._debug_last_thumb_site_pos = thumb_site_pos.copy()

        cmd_cum_delta = np.zeros(3, dtype=np.float64)
        cmd_cum_delta_obj = np.zeros(3, dtype=np.float64)
        cmd_cum_lat = 0.0
        cmd_cum_vert = 0.0
        enter_cmd_pos = getattr(policy, "_debug_rotate_enter_thumb_cmd_pos", None)
        if enter_cmd_pos is not None:
            cmd_cum_delta = thumb_pose_after[:3] - np.asarray(
                enter_cmd_pos, dtype=np.float64
            )
            cmd_cum_delta_obj = __world_vec_to_object_frame(cmd_cum_delta, obj_quat)
            cmd_cum_lat, cmd_cum_vert = __lat_vert_components(cmd_cum_delta_obj)

        site_cum_delta = np.zeros(3, dtype=np.float64)
        site_cum_delta_obj = np.zeros(3, dtype=np.float64)
        site_cum_lat = 0.0
        site_cum_vert = 0.0
        enter_site_pos = getattr(policy, "_debug_rotate_enter_thumb_site_pos", None)
        if thumb_site_id >= 0 and enter_site_pos is not None:
            thumb_site_pos = np.asarray(
                policy.wrench_sim.data.site_xpos[thumb_site_id], dtype=np.float64
            )
            site_cum_delta = thumb_site_pos - np.asarray(
                enter_site_pos, dtype=np.float64
            )
            site_cum_delta_obj = __world_vec_to_object_frame(site_cum_delta, obj_quat)
            site_cum_lat, site_cum_vert = __lat_vert_components(site_cum_delta_obj)

        print(
            "[leap_debug][rotate_distribute] "
            f"t={float(policy.wrench_sim.data.time):.3f} "
            f"v_center={np.asarray(v_center, dtype=np.float64).tolist()} "
            f"omega={np.asarray(omega, dtype=np.float64).tolist()} "
            f"v_left={np.asarray(v_left.reshape(-1), dtype=np.float64).tolist()} "
            f"v_right={np.asarray(v_right.reshape(-1), dtype=np.float64).tolist()} "
            f"obj_pos={obj_pos.tolist()} "
            f"obj_quat_wxyz={obj_quat.tolist()} "
            f"thumb_linvel={thumb_linvel_dbg.tolist()} "
            f"thumb_linvel_obj={thumb_linvel_obj.tolist()} "
            f"thumb_linvel_obj_lat={thumb_vel_lat:.6f} "
            f"thumb_linvel_obj_vert={thumb_vel_vert:.6f} "
            f"thumb_pos_delta={np.asarray(thumb_pos_delta, dtype=np.float64).tolist()} "
            f"thumb_pos_delta_obj={thumb_pos_delta_obj.tolist()} "
            f"thumb_pos_delta_obj_lat={thumb_delta_lat:.6f} "
            f"thumb_pos_delta_obj_vert={thumb_delta_vert:.6f} "
            f"cmd_step_delta={cmd_step_delta.tolist()} "
            f"cmd_step_delta_obj={cmd_step_delta_obj.tolist()} "
            f"site_step_delta={site_step_delta.tolist()} "
            f"site_step_delta_obj={site_step_delta_obj.tolist()} "
            f"site_step_delta_obj_lat={site_step_lat:.6f} "
            f"site_step_delta_obj_vert={site_step_vert:.6f} "
            f"cmd_cum_delta={cmd_cum_delta.tolist()} "
            f"cmd_cum_delta_obj={cmd_cum_delta_obj.tolist()} "
            f"cmd_cum_delta_obj_lat={cmd_cum_lat:.6f} "
            f"cmd_cum_delta_obj_vert={cmd_cum_vert:.6f} "
            f"site_cum_delta={site_cum_delta.tolist()} "
            f"site_cum_delta_obj={site_cum_delta_obj.tolist()} "
            f"site_cum_delta_obj_lat={site_cum_lat:.6f} "
            f"site_cum_delta_obj_vert={site_cum_vert:.6f} "
            f"target_lin={np.asarray(getattr(policy, '_debug_target_linvel', np.zeros(3)), dtype=np.float64).tolist()} "
            f"target_ang={np.asarray(getattr(policy, '_debug_target_angvel', np.zeros(3)), dtype=np.float64).tolist()} "
            f"p_thumb_obj={np.asarray(getattr(policy, '_debug_p_thumb_obj', np.zeros(3)), dtype=np.float64).tolist()} "
            f"cross_term={np.asarray(getattr(policy, '_debug_cross_term', np.zeros(3)), dtype=np.float64).tolist()}"
        )
        policy.debug_rotate_transition_countdown = (
            int(policy.debug_rotate_transition_countdown) - 1
        )


def create_model_based_policy(
    wrench_sim: Any,
    wrench_site_names: Tuple[str, ...] = LEAP_FINGER_TIPS,
    control_dt: float = 0.02,
    prep_duration: float = 0.0,
    auto_switch_target_enabled: bool = True,
) -> Any:
    policy = SimpleNamespace()
    _init(
        policy,
        wrench_sim=wrench_sim,
        wrench_site_names=wrench_site_names,
        control_dt=control_dt,
        prep_duration=prep_duration,
        auto_switch_target_enabled=auto_switch_target_enabled,
    )
    return policy


def forward_object_to_init(policy: Any, sim_name: str = "sim") -> None:
    _forward_object_to_init(policy, sim_name=sim_name)


def capture_object_init(policy: Any) -> None:
    _capture_object_init(policy)


def fix_object(policy: Any, sim: Any, sim_name: str = "sim") -> None:
    _fix_object(policy, sim=sim, sim_name=sim_name)


def step_policy(
    policy: Any,
    time_curr: float,
    wrenches_by_site: Optional[Dict[str, np.ndarray]] = None,
    system_state: Optional[Dict[str, np.ndarray]] = None,
    sim_name: str = "sim",
    is_real_world: bool = False,
) -> Dict[str, np.ndarray | str]:
    return _step(
        policy,
        time_curr=time_curr,
        wrenches_by_site=wrenches_by_site,
        system_state=system_state,
        sim_name=sim_name,
        is_real_world=is_real_world,
    )
    # print(self.pose_command[thumb_idx, 0:3])


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

    ref_cfg = RefConfig(
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


class LeapModelBasedPolicy(CompliancePolicy):
    def __init__(
        self,
        *,
        scene_xml: str = "",
        duration: float = 120.0,
        control_dt: float = 0.02,
        prep_duration: float = 7.0,
        status_interval: float = 1.0,
        vis: bool = True,
    ) -> None:
        self.vis = bool(vis)
        self.duration = float(duration)
        self.repo_root = find_repo_root(os.path.abspath(os.path.dirname(__file__)))

        if scene_xml:
            scene_xml_path = os.path.abspath(scene_xml)
        else:
            scene_xml_path = os.path.join(
                self.repo_root, "descriptions", "leap_hand", "scene_object_fixed.xml"
            )
        self.scene_xml_path = scene_xml_path

        self.controller = _build_controller(scene_xml_path, float(control_dt))
        if self.controller.compliance_ref is None:
            raise RuntimeError("Controller compliance_ref is not initialized.")
        self.site_names = tuple(self.controller.config.site_names)
        self.thumb_site_id = (
            int(
                mujoco.mj_name2id(
                    self.controller.wrench_sim.model, mujoco.mjtObj.mjOBJ_SITE, "th_tip"
                )
            )
            if "th_tip" in self.site_names
            else -1
        )

        self.controller.wrench_sim.data.qpos[:] = (
            self.controller.compliance_ref.default_qpos.copy()
        )
        mujoco.mj_forward(
            self.controller.wrench_sim.model, self.controller.wrench_sim.data
        )

        self.policy = create_model_based_policy(
            wrench_sim=self.controller.wrench_sim,
            wrench_site_names=self.site_names,
            control_dt=float(self.controller.ref_config.dt),
            prep_duration=max(float(prep_duration), 0.0),
            auto_switch_target_enabled=True,
        )
        forward_object_to_init(self.policy, sim_name="sim")

        self.motor_cmd = np.asarray(
            self.controller.compliance_ref.default_motor_pos, dtype=np.float32
        )
        self.measured_wrenches: dict[str, np.ndarray] = {}
        self.wrench_filter_alpha = 0.9
        self.measure_wrench_use_ema = False
        self.measure_wrench_force_only = True
        self.control_dt = float(self.controller.ref_config.dt)

        trnid = np.asarray(
            self.controller.wrench_sim.model.actuator_trnid[:, 0], dtype=np.int32
        )
        self.qpos_adr = self.controller.wrench_sim.model.jnt_qposadr[trnid]
        self.qvel_adr = self.controller.wrench_sim.model.jnt_dofadr[trnid]

        self.prep_start_motor_pos = np.asarray(
            self.controller.wrench_sim.data.qpos[self.qpos_adr], dtype=np.float32
        ).copy()
        self.prep_target_motor_pos = np.asarray(PREPARE_POS, dtype=np.float32)
        if self.prep_target_motor_pos.shape != self.prep_start_motor_pos.shape:
            self.prep_target_motor_pos = self.prep_start_motor_pos.copy()
        self.prep_duration = float(self.policy.prep_duration)
        prep_hold_duration = min(5.0, self.prep_duration)
        self.prep_ramp_duration = max(self.prep_duration - prep_hold_duration, 1e-6)

        robot_desc_dir = os.path.dirname(scene_xml_path)
        (
            kp,
            kd,
            tau_max,
            q_dot_max,
            tau_q_dot_max,
            q_dot_tau_max,
            tau_brake_max,
            kd_min,
            passive_active_ratio,
        ) = _load_motor_params(
            self.repo_root, robot_desc_dir, self.controller.wrench_sim.model
        )

        def _extra_substep(_data: mujoco.MjData) -> None:
            sim_time_local = float(_data.time)
            if sim_time_local < self.prep_duration or self.policy.phase == "close":
                capture_object_init(self.policy)
                fix_object(self.policy, self.controller.wrench_sim, sim_name="sim")
                mujoco.mj_forward(
                    self.controller.wrench_sim.model, self.controller.wrench_sim.data
                )

        self.substep_control = self._make_clamped_torque_substep_control(
            qpos_adr=self.qpos_adr,
            qvel_adr=self.qvel_adr,
            target_motor_pos_getter=lambda: self.motor_cmd,
            kp=kp,
            kd=kd,
            tau_max=tau_max,
            q_dot_max=q_dot_max,
            tau_q_dot_max=tau_q_dot_max,
            q_dot_tau_max=q_dot_tau_max,
            tau_brake_max=tau_brake_max,
            kd_min=kd_min,
            passive_active_ratio=float(passive_active_ratio),
            extra_substep_fn=_extra_substep,
        )

        self.status_interval = max(float(status_interval), 1e-3)
        self.next_status_time = 0.0
        self.done = False
        self._compliance_state_synced = False

    @staticmethod
    def _make_clamped_torque_substep_control(
        *,
        qpos_adr: np.ndarray,
        qvel_adr: np.ndarray,
        target_motor_pos_getter,
        kp: np.ndarray,
        kd: np.ndarray,
        tau_max: np.ndarray,
        q_dot_max: np.ndarray,
        tau_q_dot_max: np.ndarray,
        q_dot_tau_max: np.ndarray,
        tau_brake_max: np.ndarray,
        kd_min: np.ndarray,
        passive_active_ratio: float,
        extra_substep_fn=None,
    ):
        qpos_adr = np.asarray(qpos_adr, dtype=np.int32)
        qvel_adr = np.asarray(qvel_adr, dtype=np.int32)
        kp = np.asarray(kp, dtype=np.float64)
        kd = np.asarray(kd, dtype=np.float64)
        tau_max = np.asarray(tau_max, dtype=np.float64)
        q_dot_max = np.asarray(q_dot_max, dtype=np.float64)
        tau_q_dot_max = np.asarray(tau_q_dot_max, dtype=np.float64)
        q_dot_tau_max = np.asarray(q_dot_tau_max, dtype=np.float64)
        tau_brake_max = np.asarray(tau_brake_max, dtype=np.float64)
        kd_min = np.asarray(kd_min, dtype=np.float64)
        passive_active_ratio = float(passive_active_ratio)

        def _substep(data_step: mujoco.MjData) -> None:
            target_motor_pos = np.asarray(target_motor_pos_getter(), dtype=np.float64)
            q = np.asarray(data_step.qpos[qpos_adr], dtype=np.float64)
            q_dot = np.asarray(data_step.qvel[qvel_adr], dtype=np.float64)
            q_dot_dot = np.asarray(data_step.qacc[qvel_adr], dtype=np.float64)
            error = target_motor_pos - q

            real_kp = np.where(q_dot_dot * error < 0.0, kp * passive_active_ratio, kp)
            tau_m = real_kp * error - (kd_min + kd) * q_dot

            abs_q_dot = np.abs(q_dot)
            slope = (tau_q_dot_max - tau_max) / (q_dot_max - q_dot_tau_max)
            taper_limit = tau_max + slope * (abs_q_dot - q_dot_tau_max)
            tau_acc_limit = np.where(abs_q_dot <= q_dot_tau_max, tau_max, taper_limit)
            tau_m_clamped = np.where(
                np.logical_and(abs_q_dot > q_dot_max, q_dot * target_motor_pos > 0),
                np.where(
                    q_dot > 0,
                    np.ones_like(tau_m) * -tau_brake_max,
                    np.ones_like(tau_m) * tau_brake_max,
                ),
                np.where(
                    q_dot > 0,
                    np.clip(tau_m, -tau_brake_max, tau_acc_limit),
                    np.clip(tau_m, -tau_acc_limit, tau_brake_max),
                ),
            )
            data_step.ctrl[:] = tau_m_clamped.astype(np.float32)
            if extra_substep_fn is not None:
                extra_substep_fn(data_step)

        return _substep

    def _get_current_motor_pos(self) -> np.ndarray:
        qpos = np.asarray(self.controller.wrench_sim.data.qpos, dtype=np.float32)
        qpos_idx = np.asarray(self.qpos_adr, dtype=np.int32).reshape(-1)
        if qpos_idx.size == 0:
            return np.asarray(self.motor_cmd, dtype=np.float32).copy()
        if np.any(qpos_idx < 0) or np.any(qpos_idx >= qpos.shape[0]):
            return np.asarray(self.motor_cmd, dtype=np.float32).copy()
        return qpos[qpos_idx].astype(np.float32, copy=True)

    def _update_measured_wrenches(self) -> None:
        raw_wrenches = _get_ground_truth_wrenches(
            self.controller.wrench_sim.model,
            self.controller.wrench_sim.data,
            self.site_names,
        )
        alpha = float(np.clip(self.wrench_filter_alpha, 0.0, 1.0))
        measured: dict[str, np.ndarray] = {}
        for site_name in self.site_names:
            current = np.asarray(
                raw_wrenches.get(site_name, np.zeros(6, dtype=np.float32)),
                dtype=np.float32,
            ).copy()
            if bool(self.measure_wrench_force_only):
                current[3:] = 0.0

            if bool(self.measure_wrench_use_ema):
                prev = self.measured_wrenches.get(site_name)
                if prev is None:
                    value = current
                else:
                    value = alpha * current + (1.0 - alpha) * np.asarray(
                        prev, dtype=np.float32
                    )
            else:
                value = current
            measured[site_name] = np.asarray(value, dtype=np.float32)
        self.measured_wrenches = measured

    def _sync_sim_state_from_obs(self, obs: Any) -> None:
        has_update = False
        if obs.qpos is not None:
            qpos_arr = np.asarray(obs.qpos, dtype=np.float32).reshape(-1)
            if qpos_arr.shape[0] == int(self.controller.wrench_sim.model.nq):
                self.controller.wrench_sim.data.qpos[:] = qpos_arr
                has_update = True
        if obs.qvel is not None:
            qvel_arr = np.asarray(obs.qvel, dtype=np.float32).reshape(-1)
            if qvel_arr.shape[0] == int(self.controller.wrench_sim.model.nv):
                self.controller.wrench_sim.data.qvel[:] = qvel_arr
                has_update = True
        if has_update:
            mujoco.mj_forward(
                self.controller.wrench_sim.model, self.controller.wrench_sim.data
            )

    def _print_thumb_xref_and_real(self, sim_time: float) -> None:
        phase = str(getattr(self.policy, "phase", ""))
        if phase not in ("close", "rotate"):
            return
        if "th_tip" not in self.site_names:
            return
        thumb_idx = self.site_names.index("th_tip")
        thumb_xref = np.asarray(self.policy.pose_command[thumb_idx], dtype=np.float64)
        if self.thumb_site_id >= 0:
            thumb_real_pos = np.asarray(
                self.controller.wrench_sim.data.site_xpos[self.thumb_site_id],
                dtype=np.float64,
            )
            thumb_real_rotvec = R.from_matrix(
                np.asarray(
                    self.controller.wrench_sim.data.site_xmat[self.thumb_site_id],
                    dtype=np.float64,
                ).reshape(3, 3)
            ).as_rotvec()
        else:
            thumb_real_pos = np.full(3, np.nan, dtype=np.float64)
            thumb_real_rotvec = np.full(3, np.nan, dtype=np.float64)
        thumb_real = np.concatenate([thumb_real_pos, thumb_real_rotvec], axis=0)

        raw_thumb_wrench = np.zeros(6, dtype=np.float64)
        if self.thumb_site_id >= 0:
            mujoco.mj_rnePostConstraint(
                self.controller.wrench_sim.model, self.controller.wrench_sim.data
            )
            thumb_body_id = int(
                self.controller.wrench_sim.model.site_bodyid[self.thumb_site_id]
            )
            if 0 <= thumb_body_id < int(self.controller.wrench_sim.model.nbody):
                raw_thumb_wrench = np.asarray(
                    self.controller.wrench_sim.data.cfrc_ext[thumb_body_id],
                    dtype=np.float64,
                ).reshape(-1)
        measured_thumb_wrench = np.asarray(
            self.measured_wrenches.get("th_tip", np.zeros(6, dtype=np.float32)),
            dtype=np.float64,
        ).reshape(-1)
        command_thumb_wrench = np.asarray(
            self.policy.wrench_command[thumb_idx], dtype=np.float64
        ).reshape(-1)

        raw_force = raw_thumb_wrench[:3]
        measured_force = measured_thumb_wrench[:3]
        command_force = command_thumb_wrench[:3]
        mode = str(getattr(self.policy, "control_mode", "unknown"))
        print(
            "[leaphand][thumb_state] "
            f"t={sim_time:.3f} "
            f"phase={phase} "
            f"mode={mode} "
            f"xref={np.round(thumb_xref, 6).tolist()} "
            f"real={np.round(thumb_real, 6).tolist()} "
            f"f_raw={np.round(raw_force, 6).tolist()} "
            f"|f_raw|={float(np.linalg.norm(raw_force)):.6f} "
            f"f_meas={np.round(measured_force, 6).tolist()} "
            f"|f_meas|={float(np.linalg.norm(measured_force)):.6f} "
            f"f_cmd={np.round(command_force, 6).tolist()} "
            f"|f_cmd|={float(np.linalg.norm(command_force)):.6f}"
        )

    def step(self, obs: Any, sim: Any) -> np.ndarray:
        sim_time = float(obs.time)
        if self.duration > 0.0 and sim_time >= self.duration:
            print("[leaphand] Reached duration limit, exiting.")
            self.done = True
            return self.motor_cmd.copy()

        self._sync_sim_state_from_obs(obs)
        self._update_measured_wrenches()
        # Keep the object suspended in the environment sim during prep/close so the
        # hand can approach and grasp before free interaction starts.
        if (
            hasattr(sim, "model")
            and hasattr(sim, "data")
            and (sim_time < self.prep_duration or str(self.policy.phase) == "close")
        ):
            capture_object_init(self.policy)
            fix_object(self.policy, sim, sim_name="sim")
            mujoco.mj_forward(sim.model, sim.data)

        if sim_time < self.prep_duration:
            step_policy(
                self.policy,
                time_curr=sim_time,
                wrenches_by_site=self.measured_wrenches,
                sim_name="sim",
                is_real_world=False,
            )
            if sim_time < self.prep_ramp_duration:
                alpha = float(
                    np.clip(sim_time / max(self.prep_ramp_duration, 1e-6), 0.0, 1.0)
                )
                self.motor_cmd = (
                    self.prep_start_motor_pos
                    + (self.prep_target_motor_pos - self.prep_start_motor_pos) * alpha
                ).astype(np.float32)
            else:
                self.motor_cmd = self.prep_target_motor_pos.copy()
        else:
            if not self._compliance_state_synced:
                current_motor_pos = self._get_current_motor_pos()
                sync_compliance_state_to_current_pose(
                    self.controller, self.controller.wrench_sim.data, current_motor_pos
                )
                self._compliance_state_synced = True
                print(
                    "[leaphand] Synced compliance state to current pose at model-based start."
                )
            phase_before = str(self.policy.phase)
            step_policy(
                self.policy,
                time_curr=sim_time,
                wrenches_by_site=self.measured_wrenches,
                sim_name="sim",
                is_real_world=False,
            )
            phase_after = str(self.policy.phase)
            if phase_before != phase_after and phase_after == "rotate":
                current_motor_pos = self._get_current_motor_pos()
                sync_compliance_state_to_current_pose(
                    self.controller, self.controller.wrench_sim.data, current_motor_pos
                )
                if "th_tip" in self.site_names:
                    thumb_idx = self.site_names.index("th_tip")
                    thumb_xref = np.asarray(
                        self.policy.pose_command[thumb_idx], dtype=np.float64
                    )
                    print(
                        "[leaphand][rotate_start] "
                        f"thumb_xref={np.round(thumb_xref, 6).tolist()}"
                    )
                step_policy(
                    self.policy,
                    time_curr=sim_time,
                    wrenches_by_site=self.measured_wrenches,
                    sim_name="sim",
                    is_real_world=False,
                )
            self._print_thumb_xref_and_real(sim_time)
            policy_out = {
                "pose_command": np.asarray(
                    self.policy.pose_command, dtype=np.float32
                ).copy(),
                "wrench_command": np.asarray(
                    self.policy.wrench_command, dtype=np.float32
                ).copy(),
                "pos_stiffness": np.asarray(
                    self.policy.pos_stiffness, dtype=np.float32
                ).copy(),
                "rot_stiffness": np.asarray(
                    self.policy.rot_stiffness, dtype=np.float32
                ).copy(),
                "pos_damping": np.asarray(
                    self.policy.pos_damping, dtype=np.float32
                ).copy(),
                "rot_damping": np.asarray(
                    self.policy.rot_damping, dtype=np.float32
                ).copy(),
            }
            command_matrix = _build_command_matrix(
                site_names=self.site_names,
                policy_out=policy_out,
                measured_wrenches=self.measured_wrenches,
            )
            qpos_obs = (
                np.asarray(obs.qpos, dtype=np.float32)
                if obs.qpos is not None
                else np.asarray(self.controller.wrench_sim.data.qpos, dtype=np.float32)
            )
            motor_tor_obs = np.asarray(obs.motor_tor, dtype=np.float32)
            _, state_ref = self.controller.step(
                command_matrix=command_matrix,
                motor_torques=motor_tor_obs,
                qpos=qpos_obs,
                use_estimated_wrench=False,
            )
            if state_ref is not None:
                self.motor_cmd = np.asarray(state_ref.motor_pos, dtype=np.float32)

        if sim_time >= self.next_status_time:
            print(
                f"[leaphand] t={sim_time:.2f}s phase={self.policy.phase} mode={self.policy.control_mode}"
            )
            self.next_status_time = sim_time + self.status_interval

        return self.motor_cmd.copy()

    def close(self) -> None:
        if getattr(self.policy, "control_receiver", None) is not None:
            self.policy.control_receiver.close()
