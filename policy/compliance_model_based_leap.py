from __future__ import annotations

import os
from typing import Any, Dict, Optional

import mujoco
import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

from hybrid_servo.algorithm.ochs import solve_ochs
from hybrid_servo.algorithm.solvehfvc import transform_hfvc_to_global
from hybrid_servo.tasks.multi_finger_ochs import (
    compute_hfvc_inputs,
    generate_constraint_jacobian,
    get_center_state,
)
from hybrid_servo.utils import get_ground_truth_wrenches
from minimalist_compliance_control.utils import (
    KeyboardControlReceiver,
    ensure_matrix,
    get_damping_matrix,
    load_merged_motor_config,
    load_motor_params_from_config,
    make_clamped_torque_substep_control,
)
from policy.compliance import CompliancePolicy

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
        "init_pos": np.array([-0.13, -0.08, 0.15], dtype=np.float32),
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

OBJECT_INIT_POS_MAP = {
    "sphere": np.array([-0.125, -0.08, 0.145], dtype=np.float32),
    "box": np.array([-0.125, -0.08, 0.15], dtype=np.float32),
    "cylinder_short": np.array([-0.125, -0.075, 0.16], dtype=np.float32),
}

OBJECT_TYPE = "cylinder_short"


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
    cmd = _check_control_command(policy)

    if cmd == "c":
        _apply_reverse_command(policy)

    elif cmd == "r":
        _request_mode_switch(policy)

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
    if action_type == 0:
        _apply_reverse_command(policy)
    else:
        _request_mode_switch(policy)
    policy.auto_switch_counter += 1


def _step(
    policy,
    time_curr: float,
    wrenches_by_site: Optional[Dict[str, np.ndarray]] = None,
    *,
    sim_name: str = "sim",
    is_real_world: bool = False,
) -> None:
    if wrenches_by_site is not None:
        policy.wrenches_by_site = {
            key: np.asarray(val, dtype=np.float32)
            for key, val in wrenches_by_site.items()
        }

    if time_curr < policy.prep_duration:
        _capture_object_init(
            policy,
        )
        if not is_real_world:
            _fix_object(policy, policy.wrench_sim, sim_name=sim_name)
        return

    if policy.phase == "close":
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
                _check_switch_phase(policy)
        _advance_command_trajectory(policy, time_curr)
    elif policy.phase == "rotate":
        # Update goal with keyboard commands and threshold checking
        _update_goal(policy, time_curr)

        # Handle rotation action
        _handle_rotate_action(policy)
        _check_switch_phase(policy)

    if (
        policy.phase == "close"
        and policy.close_stage == "to_init"
        and policy.active_traj is None
    ):
        policy.close_stage = "to_target"
        policy.traj_set = False

    return


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


def _check_switch_phase(policy) -> None:
    """Switch from close to rotate once all fingertips have sufficient contact force."""
    if policy.phase == "close":
        has_contact = _check_all_fingertips_contact(policy)
        if has_contact:
            _freeze_pose_to_current(policy)
            _capture_baseline_tip_rot(policy)
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
        return {
            "pos": pos.astype(np.float32, copy=False),
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
    thumb_linvel = policy.target_rotation_linvel + np.cross(
        policy.target_rotation_angvel, p_thumb_obj
    )
    thumb_angvel = np.zeros(3)

    v_obj_goal = np.cross(policy.target_rotation_angvel - thumb_angvel, -p_thumb_obj)
    omega_obj_goal = policy.target_rotation_angvel - thumb_angvel

    return v_obj_goal, omega_obj_goal, thumb_linvel, thumb_angvel


def _handle_rotate_action(policy):
    state = _get_system_state(policy)
    target_linvel, target_angvel, thumb_linvel, thumb_angvel = _get_target_vel(
        policy, state
    )

    min_force = (
        policy.min_normal_force_rotation
        if policy.control_mode == "rotation"
        else policy.min_normal_force_translation
    )
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
    _distribute_action(policy, hfvc_solution, thumb_linvel, thumb_angvel, state)


def _distribute_action(
    policy, hfvc_solution, thumb_linvel, thumb_angvel, state
) -> None:
    """Distribute HFVC center commands to index/middle fingertips."""
    # Convert HFVC (center) wrench/velocity to contact-level targets.
    global_vel, global_frc = transform_hfvc_to_global(hfvc_solution)

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
        policy.pose_command[tip_idx, 3:6] = new_rotvec

    # set the thumb linvel and angvel
    thumb_idx = policy.wrench_site_names.index("th_tip")
    policy.pose_command[thumb_idx, :3] += (thumb_linvel.reshape(-1) * dt).astype(
        np.float32
    )

    old_rotvec_thumb = policy.pose_command[thumb_idx, 3:6].copy()
    curr_rot = R.from_rotvec(old_rotvec_thumb)
    rot_increment = R.from_rotvec(policy.target_rotation_angvel * dt)
    new_rot = rot_increment * curr_rot
    new_rotvec_thumb = new_rot.as_rotvec().astype(np.float32)
    policy.pose_command[thumb_idx, 3:6] = new_rotvec_thumb


class LeapModelBasedPolicy(CompliancePolicy):
    def __init__(
        self,
        *,
        name: str,
        robot: str,
        init_motor_pos: npt.ArrayLike,
        scene_xml: str = "",
        control_dt: float = 0.02,
        prep_duration: float = 7.0,
        vis: bool = True,
    ) -> None:
        self.vis = bool(vis)
        robot_name = str(robot).strip().lower()
        if robot_name != "leap":
            raise ValueError(
                f"LeapModelBasedPolicy requires robot='leap', got {robot}."
            )
        super().__init__(
            name=str(name),
            robot=robot_name,
            init_motor_pos=init_motor_pos,
            config_name="leap_model_based.gin",
            controller_xml_path=(str(scene_xml).strip() or None),
            controller_dt=float(control_dt),
            show_help=False,
            start_keyboard_listener=False,
            enable_plotter=False,
            enable_force_perturbation=False,
        )
        if self.controller.compliance_ref is None:
            raise RuntimeError("Controller compliance_ref is not initialized.")
        # For LEAP scene XML, the first free joint belongs to manip_object.
        # Disable floating-base frame transforms so IK stays in hand/world frame.
        self.controller.compliance_ref._floating_base_body_id = None
        self.controller.compliance_ref._last_base_pos = np.zeros(3, dtype=np.float32)
        self.controller.compliance_ref._last_base_quat = np.array(
            [1.0, 0.0, 0.0, 0.0], dtype=np.float32
        )
        controller_xml_path = str(self.controller.config.xml_path)
        if not os.path.isabs(controller_xml_path):
            controller_xml_path = os.path.abspath(
                os.path.join(self.repo_root, controller_xml_path)
            )
        self.scene_xml_path = controller_xml_path
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
        prep_duration_sec = max(float(prep_duration), 0.0)
        self.wrench_sim = self.controller.wrench_sim
        self.wrench_site_names = list(self.site_names)
        self.num_sites = len(self.wrench_site_names)
        self.control_dt = float(self.controller.ref_config.dt)
        self.prep_duration = prep_duration_sec
        self.wrenches_by_site = {}
        self.wrench_command = np.zeros((self.num_sites, 6), dtype=np.float32)
        self.pos_stiffness = np.zeros((self.num_sites, 9), dtype=np.float32)
        self.rot_stiffness = np.zeros((self.num_sites, 9), dtype=np.float32)
        self.pos_damping = np.zeros((self.num_sites, 9), dtype=np.float32)
        self.rot_damping = np.zeros((self.num_sites, 9), dtype=np.float32)

        self.object_type = OBJECT_TYPE
        object_info = OBJECT_MASS_MAP.get(self.object_type, OBJECT_MASS_MAP["unknown"])
        self.object_mass = float(object_info.get("mass", 0.05))
        self.object_geom_size = np.asarray(
            object_info.get("geom_size", np.array([0.04], dtype=np.float32)),
            dtype=np.float32,
        )
        init_pos = object_info.get("init_pos")
        init_quat = object_info.get("init_quat")
        self.object_init_pos = (
            np.asarray(init_pos, dtype=np.float32).copy()
            if init_pos is not None
            else None
        )
        self.object_init_quat = (
            np.asarray(init_quat, dtype=np.float32).copy()
            if init_quat is not None
            else None
        )

        self.use_compliance = True
        self.log_ik = True
        self.pd_updated = False
        self.desired_kp = 1500  # 450
        self.desired_kd = 0

        self.contact_force = 0.0
        self.normal_pos_stiffness = 10.0
        self.tangent_pos_stiffness = 100.0
        self.normal_rot_stiffness = 10.0
        self.tangent_rot_stiffness = 20.0

        self.ref_motor_pos = np.array(PREPARE_POS, dtype=np.float32)
        self.initial_pose_command = _build_pose_command(self, INIT_POSE_DATA)
        self.target_pose_command = _build_pose_command(self, TARGET_POSE_DATA)
        self.pose_command = self.initial_pose_command.copy()
        self.integrated_angle_thumb = np.zeros(3, dtype=np.float32)
        self.pose_interp_pos_speed = 0.1
        self.pose_interp_rot_speed = 1.0
        self.pose_interp_min_duration = 0.2
        self.pose_interp_max_duration = 2.0
        mass_matrix = ensure_matrix(1.0)
        inertia_matrix = ensure_matrix([1.0, 1.0, 1.0])

        open_pos_stiff = ensure_matrix([400.0, 400.0, 400.0])
        open_rot_stiff = ensure_matrix([20.0, 20.0, 20.0])
        open_pos_stiff_arr = np.broadcast_to(
            open_pos_stiff, (self.num_sites, 3, 3)
        ).astype(np.float32)
        open_rot_stiff_arr = np.broadcast_to(
            open_rot_stiff, (self.num_sites, 3, 3)
        ).astype(np.float32)
        open_pos_damp = np.stack(
            [
                get_damping_matrix(open_pos_stiff, mass_matrix)
                for _ in range(self.num_sites)
            ],
            axis=0,
        ).astype(np.float32)
        open_rot_damp = np.stack(
            [
                get_damping_matrix(open_rot_stiff, inertia_matrix)
                for _ in range(self.num_sites)
            ],
            axis=0,
        ).astype(np.float32)
        open_wrench = np.zeros((self.num_sites, 6), dtype=np.float32)
        self.open_gains = {
            "pos_stiff": open_pos_stiff_arr,
            "rot_stiff": open_rot_stiff_arr,
            "pos_damp": open_pos_damp,
            "rot_damp": open_rot_damp,
            "wrench": open_wrench,
        }
        self.close_gains = _compute_force_and_stiffness(self, self.target_pose_command)

        self.forward_traj = _build_command_trajectory(
            self,
            self.initial_pose_command,
            self.target_pose_command,
            self.open_gains,
            self.close_gains,
        )
        self.backward_traj = _build_command_trajectory(
            self,
            self.target_pose_command,
            self.initial_pose_command,
            self.close_gains,
            self.open_gains,
        )

        self.active_traj = None
        self.traj_start_time = 0.0

        # Initialize stiffness/wrench targets from first trajectory sample.
        _apply_traj_sample(self, self.forward_traj, 0)

        self.phase = "close"
        self.traj_set = False
        self.object_body_name = "manip_object"
        self.object_qpos_adr = None
        self.object_qvel_adr = None
        self.close_stage = "to_init"
        self.jacobian_constraint = generate_constraint_jacobian()
        self.target_rotation_angvel = np.array([0.0, 0.0, 0.0])
        self.target_rotation_linvel = np.array([0.03, 0.0, 0.0])
        self.last_angvel_flip_time = None
        self.pos_kp = 300  # High stiffness for anisotropic fingers.
        self.force_kp = 200  # Low stiffness for anisotropic fingers.
        self.rot_kp = 20
        self.baseline_tip_rot = {}
        self.interval = 1.5
        self.last_contact_pos = {tip: None for tip in self.wrench_site_names}

        self.control_mode = "translation"
        self.rotation_angvel_magnitude = 0.5
        self.translation_linvel_magnitude = 0.03

        self.min_normal_force_rotation = float(object_info["min_normal_force_rotation"])
        self.min_normal_force_translation = float(
            object_info["min_normal_force_translation"]
        )

        self.centripetal_force_magnitude = 1.0

        self.contact_force_threshold = 0.1

        self.threshold_angle = np.pi / 4
        self.threshold_position = 0.05
        self.threshold_angle_reverse = -self.threshold_angle
        self.threshold_position_reverse = -0.02
        self.auto_switch_target_enabled = True
        self.auto_switch_counter = 0
        self.limit_reached_flag = False
        self.integrated_angle = 0.0
        self.integrated_position = 0.0
        self.last_integration_time = None

        self.mode_switch_pending = False
        self.target_mode = None
        self.return_to_zero_tolerance = 0.003
        self.control_receiver = None
        try:
            self.control_receiver = KeyboardControlReceiver(
                valid_commands={"c", "r"},
                name="LeapRotateCompliance",
                help_labels={"c": "reverse", "r": "switch mode"},
            )
        except Exception as exc:
            self.control_receiver = None
            print(f"[LeapRotateCompliance] Warning: control receiver disabled: {exc}")
        _capture_object_init(self)
        _fix_object(self, self.controller.wrench_sim, sim_name="sim")
        mujoco.mj_forward(
            self.controller.wrench_sim.model, self.controller.wrench_sim.data
        )

        self.motor_cmd = np.asarray(self.init_motor_pos, dtype=np.float32).copy()
        self.measured_wrenches: dict[str, np.ndarray] = {}
        self.control_dt = float(self.controller.ref_config.dt)

        trnid = np.asarray(
            self.controller.wrench_sim.model.actuator_trnid[:, 0], dtype=np.int32
        )
        self.qpos_adr = self.controller.wrench_sim.model.jnt_qposadr[trnid]
        self.qvel_adr = self.controller.wrench_sim.model.jnt_dofadr[trnid]

        self.prep_start_motor_pos = np.asarray(
            self.init_motor_pos, dtype=np.float32
        ).copy()
        self.prep_target_motor_pos = np.asarray(PREPARE_POS, dtype=np.float32)
        if self.prep_target_motor_pos.shape != self.prep_start_motor_pos.shape:
            self.prep_target_motor_pos = self.prep_start_motor_pos.copy()
        self.prep_duration = prep_duration_sec
        prep_hold_duration = min(5.0, self.prep_duration)
        self.prep_ramp_duration = max(self.prep_duration - prep_hold_duration, 1e-6)
        self.prep_initialized_from_obs = False

        robot_desc_dir = os.path.dirname(self.scene_xml_path)
        motor_cfg = load_merged_motor_config(
            default_path=os.path.join(self.repo_root, "descriptions", "default.yml"),
            robot_path=os.path.join(robot_desc_dir, "robot.yml"),
            motors_path=os.path.join(robot_desc_dir, "motors.yml"),
        )
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
        ) = load_motor_params_from_config(
            model=self.controller.wrench_sim.model,
            config=motor_cfg,
            allow_act_suffix=True,
            dtype=np.float32,
        )

        def _extra_substep(_data: mujoco.MjData) -> None:
            sim_time_local = float(_data.time)
            if sim_time_local < self.prep_duration or self.phase == "close":
                _capture_object_init(self)
                _fix_object(self, self.controller.wrench_sim, sim_name="sim")
                mujoco.mj_forward(
                    self.controller.wrench_sim.model, self.controller.wrench_sim.data
                )

        self.substep_control = make_clamped_torque_substep_control(
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
        self.done = False

    def step(self, obs: Any, sim: Any) -> np.ndarray:
        sim_time = float(obs.time)
        sim_name = str(getattr(sim, "name", ""))
        is_real_world = "real" in sim_name.lower()
        if (
            is_real_world
            and not self.prep_initialized_from_obs
            and obs.motor_pos is not None
        ):
            obs_motor_pos = np.asarray(obs.motor_pos, dtype=np.float32).reshape(-1)
            if obs_motor_pos.shape == self.prep_start_motor_pos.shape:
                self.prep_start_motor_pos = obs_motor_pos.copy()
                self.motor_cmd = obs_motor_pos.copy()
                self.prep_initialized_from_obs = True
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
        if is_real_world:
            self.measured_wrenches = {
                site_name: np.asarray(
                    self.measured_wrenches.get(
                        site_name, np.zeros(6, dtype=np.float32)
                    ),
                    dtype=np.float32,
                ).copy()
                for site_name in self.site_names
            }
        else:
            raw_wrenches = get_ground_truth_wrenches(
                self.controller.wrench_sim.model,
                self.controller.wrench_sim.data,
                self.site_names,
            )
            self.measured_wrenches = {
                site_name: np.asarray(
                    raw_wrenches.get(site_name, np.zeros(6, dtype=np.float32)),
                    dtype=np.float32,
                ).copy()
                for site_name in self.site_names
            }
        if sim_time < self.prep_duration or str(self.phase) == "close":
            _capture_object_init(self)
            if not is_real_world and hasattr(sim, "model") and hasattr(sim, "data"):
                _fix_object(self, sim, sim_name=sim_name or "sim")
                mujoco.mj_forward(sim.model, sim.data)
        if sim_time < self.prep_duration:
            _step(
                self,
                time_curr=sim_time,
                wrenches_by_site=self.measured_wrenches,
                sim_name=sim_name or "sim",
                is_real_world=is_real_world,
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
            _step(
                self,
                time_curr=sim_time,
                wrenches_by_site=self.measured_wrenches,
                sim_name=sim_name or "sim",
                is_real_world=is_real_world,
            )
            command_matrix = self.build_command_matrix(
                pose_command=np.asarray(self.pose_command, dtype=np.float32),
                wrench_command=np.asarray(self.wrench_command, dtype=np.float32),
                measured_wrenches=self.measured_wrenches,
            )
            next_motor_cmd, state_ref = self.apply_controller_step(
                command_matrix=command_matrix,
                target_motor_pos=self.motor_cmd,
                measured_wrenches=self.measured_wrenches,
                site_names=self.site_names,
                motor_torques=np.asarray(obs.motor_tor, dtype=np.float32),
                qpos=(
                    np.asarray(obs.qpos, dtype=np.float32)
                    if obs.qpos is not None
                    else np.asarray(
                        self.controller.wrench_sim.data.qpos, dtype=np.float32
                    )
                ),
                controlled_actuators_only=False,
            )
            self.motor_cmd = np.asarray(next_motor_cmd, dtype=np.float32)
            if state_ref is not None:
                self._last_state = state_ref
        return self.motor_cmd.copy()

    def close(self, exp_folder_path: str = "") -> None:
        if getattr(self, "control_receiver", None) is not None:
            self.control_receiver.close()
        super().close(exp_folder_path=exp_folder_path)
