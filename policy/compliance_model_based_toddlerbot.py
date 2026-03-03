from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import gin
import joblib
import mujoco
import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation as R

from hybrid_servo.algorithm.ochs import solve_ochs
from hybrid_servo.algorithm.solvehfvc import HFVC, transform_hfvc_to_global
from hybrid_servo.tasks.bimanual_ochs import (
    compute_center_quaternion_from_hands,
    compute_ochs_inputs,
)
from hybrid_servo.tasks.bimanual_ochs import (
    generate_constraint_jacobian as bimanual_generate_constraint_jacobian,
)
from hybrid_servo.utils import find_repo_root, sync_compliance_state_to_current_pose
from minimalist_compliance_control.compliance_ref import COMMAND_LAYOUT, ComplianceState
from minimalist_compliance_control.controller import (
    ComplianceController,
    ControllerConfig,
    RefConfig,
)
from minimalist_compliance_control.utils import (
    KeyboardControlReceiver,
    load_merged_motor_config,
    load_motor_params_from_config,
    make_clamped_torque_substep_control,
)
from minimalist_compliance_control.wrench_estimation import WrenchEstimateConfig
from policy.compliance import CompliancePolicy


@dataclass(frozen=True)
class PolicyConfig:
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
    distance_threshold_margin: float = 0.005
    approach_timeout: float = 5.0
    contact_wait_duration: float = 0.0
    pid_kp: float = 0.0
    kneel_motion_file: str = "descriptions/toddlerbot_2xm/kneel_2xm.lz4"
    initial_active_hands_mode: str = "left"
    threshold_angle: float = np.pi / 6.0
    threshold_angle_z: float = np.pi / 4.0
    print_ochs_world_velocity: bool = False
    ochs_print_interval: float = 0.2


@dataclass
class PolicyRuntime:
    """Mutable runtime state for model-based policy integration."""

    pose_command: ArrayF64
    wrench_command: ArrayF64
    pos_stiffness: ArrayF64
    pos_damping: ArrayF64
    rot_stiffness: ArrayF64
    rot_damping: ArrayF64
    delta_goal_angular_velocity: ArrayF64

    phase: str
    phase_start_time: float

    active_hands_mode: str
    active_hand_indices: Tuple[int, ...]

    goal_rotate_axis: ArrayF64
    goal_angular_velocity: float
    goal_speed: float
    goal_angle: float
    goal_time: Optional[float]

    kneel_action_arr: ArrayF64
    kneel_qpos: ArrayF64
    kneel_qpos_source_dim: int

    reach_init_state: bool = False
    contact_reach_time: Optional[float] = None
    model_based_start_time: Optional[float] = None

    approach_progress: dict[int, float] | None = None
    approach_start_pose: dict[int, Optional[ArrayF64]] | None = None

    default_left_hand_center_rotvec: Optional[ArrayF64] = None
    default_right_hand_center_rotvec: Optional[ArrayF64] = None

    rigid_body_center: Optional[ArrayF64] = None
    rigid_body_orientation: Optional[ArrayF64] = None  # wxyz
    hand_offsets_in_body_frame: Optional[ArrayF64] = None

    expected_ball_pos: Optional[ArrayF64] = None
    last_ochs_print_time: Optional[float] = None


# ---- Inlined Toddlerbot model-based helpers ----

ArrayF64 = npt.NDArray[np.float64]

HAND_POS_KEYS = (
    "left_hand_1_pos",
    "left_hand_2_pos",
    "left_hand_3_pos",
    "right_hand_1_pos",
    "right_hand_2_pos",
    "right_hand_3_pos",
)
HAND_QUAT_KEYS = (
    "left_hand_1_quat",
    "left_hand_2_quat",
    "left_hand_3_quat",
    "right_hand_1_quat",
    "right_hand_2_quat",
    "right_hand_3_quat",
)


def _normalize_mode(mode: str) -> str:
    mode_norm = str(mode).strip().lower()
    return mode_norm if mode_norm in ("left", "right", "both") else "left"


def _active_hand_indices_from_mode(mode: str) -> Tuple[int, ...]:
    m = _normalize_mode(mode)
    if m == "left":
        return (0, 1, 2)
    if m == "right":
        return (3, 4, 5)
    return (0, 1, 2, 3, 4, 5)


def _active_site_indices_from_mode(mode: str) -> Tuple[int, ...]:
    m = _normalize_mode(mode)
    if m == "left":
        return (0,)
    if m == "right":
        return (1,)
    return (0, 1)


def _goal_axis_from_mode(mode: str) -> ArrayF64:
    m = _normalize_mode(mode)
    if m == "both":
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if m == "left":
        return np.array([1.0, 0.0, 0.0], dtype=np.float64)
    return np.array([-1.0, 0.0, 0.0], dtype=np.float64)


def _set_active_hands_mode(
    runtime: PolicyRuntime,
    cfg: PolicyConfig,
    mode: str,
    *,
    keep_speed: bool = True,
) -> bool:
    mode_norm = _normalize_mode(mode)
    changed = mode_norm != runtime.active_hands_mode
    if not changed:
        return False

    runtime.active_hands_mode = mode_norm
    runtime.active_hand_indices = _active_hand_indices_from_mode(mode_norm)
    runtime.goal_rotate_axis = _goal_axis_from_mode(mode_norm)
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


def _interpolate_linear(
    p_start: ArrayF64, p_end: ArrayF64, duration: float, t: float
) -> ArrayF64:
    if t <= 0.0:
        return p_start
    if t >= duration:
        return p_end
    return p_start + (p_end - p_start) * (t / duration)


def _binary_search(arr: ArrayF64, t: float) -> int:
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


def _interpolate_action(t: float, time_arr: ArrayF64, action_arr: ArrayF64) -> ArrayF64:
    if t <= float(time_arr[0]):
        return np.asarray(action_arr[0], dtype=np.float64)
    if t >= float(time_arr[-1]):
        return np.asarray(action_arr[-1], dtype=np.float64)

    idx = _binary_search(time_arr, t)
    idx = max(0, min(idx, len(time_arr) - 2))
    p_start = np.asarray(action_arr[idx], dtype=np.float64)
    p_end = np.asarray(action_arr[idx + 1], dtype=np.float64)
    duration = float(time_arr[idx + 1] - time_arr[idx])
    return _interpolate_linear(p_start, p_end, duration, t - float(time_arr[idx]))


def _build_prep_traj(
    init_motor_pos: ArrayF64,
    target_motor_pos: ArrayF64,
    prep_duration: float,
    control_dt: float,
    prep_hold_duration: float,
) -> tuple[ArrayF64, ArrayF64]:
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
            prep_action[i] = _interpolate_linear(
                init_pos, target_pos, max(blend_duration, 1e-6), t
            )
        else:
            prep_action[i] = target_pos
    return prep_time, prep_action


def _poll_keyboard_command(control_receiver: object | None) -> str | None:
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


def _update_goal_from_keyboard_and_time(
    runtime: PolicyRuntime,
    cfg: PolicyConfig,
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
        mode_changed = _set_active_hands_mode(
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


def _sensor_data(model: mujoco.MjModel, data: mujoco.MjData, name: str) -> ArrayF64:
    sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
    if sensor_id < 0:
        raise KeyError(f"Sensor '{name}' not found.")
    start = int(model.sensor_adr[sensor_id])
    end = start + int(model.sensor_dim[sensor_id])
    return np.asarray(data.sensordata[start:end], dtype=np.float64).copy()


def _build_contact_state(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    left_site_id: int,
    right_site_id: int,
    ball_pos: Optional[ArrayF64] = None,
) -> Dict[str, ArrayF64]:
    if ball_pos is None:
        try:
            ball_body_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_BODY, "rolling_ball"
            )
            if ball_body_id < 0:
                raise KeyError("Body 'rolling_ball' not found.")
            ball_pos = np.asarray(data.body_xpos[ball_body_id], dtype=np.float64).copy()
        except Exception:
            ball_pos = _sensor_data(model, data, "rolling_ball_framepos")
    else:
        ball_pos = np.asarray(ball_pos, dtype=np.float64).reshape(3).copy()

    state: Dict[str, ArrayF64] = {
        "ball_pos": ball_pos,
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

        state[f"left_hand_{i}_pos"] = _sensor_data(model, data, f"{left_prefix}_pos")
        state[f"left_hand_{i}_quat"] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        state[f"left_hand_{i}_linvel"] = _sensor_data(
            model, data, f"{left_prefix}_linvel"
        )
        state[f"left_hand_{i}_angvel"] = _sensor_data(
            model, data, f"{left_prefix}_angvel"
        )

        state[f"right_hand_{i}_pos"] = _sensor_data(model, data, f"{right_prefix}_pos")
        state[f"right_hand_{i}_quat"] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        state[f"right_hand_{i}_linvel"] = _sensor_data(
            model, data, f"{right_prefix}_linvel"
        )
        state[f"right_hand_{i}_angvel"] = _sensor_data(
            model, data, f"{right_prefix}_angvel"
        )

    return state


def _load_robot_motor_config() -> dict:
    repo_root = find_repo_root(os.path.abspath(os.path.dirname(__file__)))
    default_path = os.path.join(repo_root, "descriptions", "default.yml")
    robot_path = os.path.join(
        repo_root,
        "descriptions",
        "toddlerbot_2xm",
        "robot.yml",
    )
    return load_merged_motor_config(default_path=default_path, robot_path=robot_path)


def _load_motor_params(model: mujoco.MjModel) -> tuple[ArrayF64, ...]:
    config = _load_robot_motor_config()
    return load_motor_params_from_config(
        model=model,
        config=config,
        allow_act_suffix=False,
        dtype=np.float64,
    )


def _load_motor_group_indices(
    model: mujoco.MjModel,
) -> dict[str, npt.NDArray[np.int32]]:
    config = _load_robot_motor_config()
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


def _load_kneel_trajectory(
    example_dir: str,
    cfg: PolicyConfig,
    default_motor_pos: ArrayF64,
    default_qpos: ArrayF64,
    motor_dim: int,
    qpos_dim: int,
) -> tuple[ArrayF64, ArrayF64, int]:
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


def _skew_matrix(vec: ArrayF64) -> ArrayF64:
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


def _interpolate_se3_pose(
    start_pose: ArrayF64, target_pose: ArrayF64, alpha: float
) -> ArrayF64:
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
    omega_hat = _skew_matrix(omega)
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
    omega_hat = _skew_matrix(omega)
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


def _ensure_default_hand_rotvec(
    runtime: PolicyRuntime,
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


def _reset_pose_command_to_current_sites(
    runtime: PolicyRuntime,
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


def _initialize_runtime_from_default_state(
    default_state: ComplianceState,
    cfg: PolicyConfig,
    kneel_action_arr: ArrayF64,
    kneel_qpos: ArrayF64,
    kneel_qpos_source_dim: int,
) -> PolicyRuntime:
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
    init_mode = _normalize_mode(cfg.initial_active_hands_mode)
    init_goal_vel = float(cfg.goal_angular_velocity)
    init_goal_speed = max(abs(init_goal_vel), 1e-6)

    return PolicyRuntime(
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
        active_hand_indices=_active_hand_indices_from_mode(init_mode),
        goal_rotate_axis=_goal_axis_from_mode(init_mode),
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


def _reset_approach_interp(runtime: PolicyRuntime) -> None:
    if runtime.approach_progress is None:
        runtime.approach_progress = {0: 0.0, 1: 0.0}
    if runtime.approach_start_pose is None:
        runtime.approach_start_pose = {0: None, 1: None}
    for hand_idx in (0, 1):
        runtime.approach_progress[hand_idx] = 0.0
        runtime.approach_start_pose[hand_idx] = None


def _compute_approach_target(
    cfg: PolicyConfig,
    ball_pos: ArrayF64,
    is_left_hand: bool,
    default_rotvec: Optional[ArrayF64],
) -> tuple[ArrayF64, R]:
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


def _interpolate_hand_pose_to_target(
    runtime: PolicyRuntime,
    cfg: PolicyConfig,
    hand_idx: int,
    target_pos: ArrayF64,
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

    interp_pose = _interpolate_se3_pose(start_pose, target_pose, alpha)
    runtime.pose_command[hand_idx] = interp_pose

    if alpha >= 1.0:
        runtime.approach_progress[hand_idx] = 0.0
        runtime.approach_start_pose[hand_idx] = None
    else:
        runtime.approach_progress[hand_idx] = min(progress + control_dt, duration)


def _run_approach_phase(
    runtime: PolicyRuntime,
    cfg: PolicyConfig,
    state: Dict[str, ArrayF64],
    control_dt: float,
) -> bool:
    ball_pos = np.asarray(state["ball_pos"], dtype=np.float64).reshape(3)
    active_sites = set(_active_site_indices_from_mode(runtime.active_hands_mode))
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
        _reset_approach_interp(runtime)
        return True

    left_target_pos, left_target_rot = _compute_approach_target(
        cfg,
        ball_pos,
        is_left_hand=True,
        default_rotvec=runtime.default_left_hand_center_rotvec,
    )
    right_target_pos, right_target_rot = _compute_approach_target(
        cfg,
        ball_pos,
        is_left_hand=False,
        default_rotvec=runtime.default_right_hand_center_rotvec,
    )

    if 0 in active_sites:
        _interpolate_hand_pose_to_target(
            runtime,
            cfg,
            hand_idx=0,
            target_pos=left_target_pos,
            target_rot=left_target_rot,
            control_dt=control_dt,
        )
    if 1 in active_sites:
        _interpolate_hand_pose_to_target(
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


def _initialize_rigid_body(runtime: PolicyRuntime, state: Dict[str, ArrayF64]) -> None:
    hand_positions = []
    hand_quats = []

    for idx in runtime.active_hand_indices:
        hand_key = HAND_POS_KEYS[idx]
        quat_key = HAND_QUAT_KEYS[idx]
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


def _distribute_rigid_body_motion(
    runtime: PolicyRuntime,
    hfvc_solution: HFVC,
    state: Dict[str, ArrayF64],
    dt: float,
) -> Dict[str, ArrayF64]:
    if runtime.rigid_body_center is None:
        _initialize_rigid_body(runtime, state)

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

    result: Dict[str, ArrayF64] = {
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
        hand_name = HAND_POS_KEYS[global_idx].replace("_pos", "")
        result[f"{hand_name}_vel"] = hand_velocities[local_idx]
        result[f"{hand_name}_pos"] = hand_positions_world[local_idx]

    return result


def _maybe_print_ochs_world_velocity() -> None:
    pass


def _assign_stiffness(
    runtime: PolicyRuntime,
    cfg: PolicyConfig,
    left_vel: ArrayF64,
    right_vel: ArrayF64,
) -> None:
    def build_diag(vel: ArrayF64) -> tuple[ArrayF64, ArrayF64]:
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


def _integrate_pose_command(
    runtime: PolicyRuntime,
    cfg: PolicyConfig,
    distributed_motion: Dict[str, ArrayF64],
    dt: float,
) -> None:
    active_sites = set(_active_site_indices_from_mode(runtime.active_hands_mode))
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


def _update_expected_ball_pos(
    runtime: PolicyRuntime,
    cfg: PolicyConfig,
    dt: float,
    state: Dict[str, ArrayF64],
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


def _update_delta_goal(
    runtime: PolicyRuntime, cfg: PolicyConfig, state: Dict[str, ArrayF64]
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


def _build_command_matrix(
    runtime: PolicyRuntime,
    measured_wrenches: Dict[str, ArrayF64],
    site_names: Tuple[str, str],
) -> ArrayF64:
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


def _resolve_mocap_target_ids(
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


def _update_mocap_targets_from_state_ref(
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


class ToddlerbotModelBasedPolicy(CompliancePolicy):
    def __init__(
        self,
        *,
        name: str,
        robot: str,
        init_motor_pos: npt.ArrayLike,
        vis: bool,
    ) -> None:
        self.cfg = PolicyConfig()
        self.vis = bool(vis)
        robot_name = str(robot).strip().lower()
        if robot_name != "toddlerbot":
            raise ValueError(
                f"ToddlerbotModelBasedPolicy requires robot='toddlerbot', got {robot}."
            )

        repo_root = find_repo_root(os.path.abspath(os.path.dirname(__file__)))
        os.chdir(repo_root)
        gin.clear_config()
        gin.parse_config_file(
            os.path.join(repo_root, "config", "toddlerbot.gin"), skip_unknown=True
        )
        gin.parse_config_file(
            os.path.join(repo_root, "config", "toddlerbot_model_based.gin"),
            skip_unknown=True,
        )

        self.controller = ComplianceController(
            config=ControllerConfig(),
            estimate_config=WrenchEstimateConfig(),
            ref_config=RefConfig(),
        )
        if self.controller.compliance_ref is None:
            raise RuntimeError("Compliance reference failed to initialize.")
        self.controller.wrench_sim.model.opt.timestep = float(self.cfg.sim_dt)
        mujoco.mj_resetData(
            self.controller.wrench_sim.model, self.controller.wrench_sim.data
        )
        mujoco.mj_forward(
            self.controller.wrench_sim.model, self.controller.wrench_sim.data
        )
        super().__init__(
            name=str(name),
            robot=robot_name,
            init_motor_pos=init_motor_pos,
            controller=self.controller,
            show_help=False,
            start_keyboard_listener=False,
            enable_plotter=False,
            enable_force_perturbation=False,
        )

        self.site_names = tuple(self.controller.config.site_names)
        if self.site_names != ("left_hand_center", "right_hand_center"):
            raise ValueError(
                "This example expects site_names == ('left_hand_center', 'right_hand_center')."
            )
        self.left_site_id = self.controller.wrench_sim.site_ids[self.site_names[0]]
        self.right_site_id = self.controller.wrench_sim.site_ids[self.site_names[1]]

        trnid = np.asarray(
            self.controller.wrench_sim.model.actuator_trnid[:, 0], dtype=np.int32
        )
        if not np.all(trnid >= 0):
            raise ValueError("Actuator without joint mapping is not supported.")
        self.qpos_adr = np.asarray(
            self.controller.wrench_sim.model.jnt_qposadr[trnid], dtype=np.int32
        )
        self.qvel_adr = np.asarray(
            self.controller.wrench_sim.model.jnt_dofadr[trnid], dtype=np.int32
        )

        default_state = self.default_state
        kneel_action_arr, kneel_qpos, kneel_qpos_source_dim = _load_kneel_trajectory(
            example_dir=repo_root,
            cfg=self.cfg,
            default_motor_pos=np.asarray(default_state.motor_pos, dtype=np.float64),
            default_qpos=np.asarray(default_state.qpos, dtype=np.float64),
            motor_dim=self.controller.wrench_sim.model.nu,
            qpos_dim=self.controller.wrench_sim.model.nq,
        )
        self.runtime = _initialize_runtime_from_default_state(
            default_state=default_state,
            cfg=self.cfg,
            kneel_action_arr=kneel_action_arr,
            kneel_qpos=kneel_qpos,
            kneel_qpos_source_dim=kneel_qpos_source_dim,
        )
        default_motor_pos = np.asarray(default_state.motor_pos, dtype=np.float64).copy()
        self.motor_cmd = np.asarray(self.init_motor_pos, dtype=np.float64).copy()
        self.latest_state_ref = None
        self.measured_wrenches: dict[str, np.ndarray] = {
            self.site_names[0]: np.zeros(6, dtype=np.float64),
            self.site_names[1]: np.zeros(6, dtype=np.float64),
        }

        self.jacobian_by_mode = {
            "left": bimanual_generate_constraint_jacobian(num_hands=3),
            "right": bimanual_generate_constraint_jacobian(num_hands=3),
            "both": bimanual_generate_constraint_jacobian(num_hands=6),
        }
        self.control_receiver = KeyboardControlReceiver(
            valid_commands={"c", "l", "r", "b"},
            name="model_based",
            help_labels={
                "c": "reverse",
                "l": "left",
                "r": "right",
                "b": "both",
            },
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
        ) = _load_motor_params(self.controller.wrench_sim.model)
        group_indices = _load_motor_group_indices(self.controller.wrench_sim.model)
        neck_indices = group_indices.get("neck", np.zeros(0, dtype=np.int32))
        leg_indices = group_indices.get("leg", np.zeros(0, dtype=np.int32))
        self.hold_indices = np.unique(
            np.concatenate([neck_indices, leg_indices])
        ).astype(np.int32)
        self.kneel_hold_motor_pos = self.runtime.kneel_action_arr[-1].copy()

        self.control_dt = float(self.controller.ref_config.dt)
        self.prep_time, self.prep_action = _build_prep_traj(
            self.init_motor_pos,
            default_motor_pos,
            self.cfg.prep_duration,
            self.control_dt,
            self.cfg.prep_hold_duration,
        )
        self.left_mocap_id, self.right_mocap_id = _resolve_mocap_target_ids(
            self.controller.wrench_sim.model
        )

        substep_control = make_clamped_torque_substep_control(
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
        )
        self.substep_control = substep_control
        try:
            ball_body_id = mujoco.mj_name2id(
                self.controller.wrench_sim.model,
                mujoco.mjtObj.mjOBJ_BODY,
                "rolling_ball",
            )
            if ball_body_id < 0:
                raise KeyError("Body 'rolling_ball' not found.")
            self.default_ball_pos = np.asarray(
                self.controller.wrench_sim.data.body_xpos[ball_body_id],
                dtype=np.float64,
            ).copy()
        except Exception:
            self.default_ball_pos = np.array([0.24, -0.0, 0.08], dtype=np.float64)
        self.ball_pos_estimate_log: list[np.ndarray] = []
        self.done = False

    def update_ball_pose_estimate(
        self, obs: Any, sim: Any, is_real: bool
    ) -> np.ndarray:
        if not is_real:
            try:
                ball_body_id = sim.model.body("rolling_ball").id
                return np.asarray(
                    sim.data.body_xpos[ball_body_id], dtype=np.float64
                ).copy()
            except Exception:
                pass

        if self.ball_pos_estimate_log:
            return self.ball_pos_estimate_log[-1].copy()
        return self.default_ball_pos.copy()

    def step(self, obs: Any, sim: Any) -> np.ndarray:
        t = float(obs.time)
        sim_name = str(getattr(sim, "name", ""))
        is_real_world = "real" in sim_name.lower()
        if (
            is_real_world
            and not self.prep_initialized_from_obs
            and obs.motor_pos is not None
            and t <= float(self.cfg.prep_duration)
        ):
            obs_motor_pos = np.asarray(obs.motor_pos, dtype=np.float64).reshape(-1)
            if obs_motor_pos.shape == self.prep_target_motor_pos.shape:
                self.prep_time, self.prep_action = _build_prep_traj(
                    obs_motor_pos,
                    self.prep_target_motor_pos,
                    self.cfg.prep_duration,
                    self.control_dt,
                    self.cfg.prep_hold_duration,
                )
                self.motor_cmd = obs_motor_pos.copy()
                self.prep_initialized_from_obs = True
        if obs.qpos is not None:
            qpos_obs = np.asarray(obs.qpos, dtype=np.float32).reshape(-1)
            if qpos_obs.shape[0] == int(self.controller.wrench_sim.model.nq):
                self.controller.sync_qpos(qpos_obs)
        if obs.qvel is not None:
            qvel_obs = np.asarray(obs.qvel, dtype=np.float32).reshape(-1)
            if qvel_obs.shape[0] == int(self.controller.wrench_sim.model.nv):
                self.controller.wrench_sim.data.qvel[:] = qvel_obs
                mujoco.mj_forward(
                    self.controller.wrench_sim.model, self.controller.wrench_sim.data
                )

        ball_pos = self.update_ball_pose_estimate(obs, sim, is_real_world)
        self.ball_pos_estimate_log.append(
            np.asarray(ball_pos, dtype=np.float64).reshape(3).copy()
        )
        state = _build_contact_state(
            self.controller.wrench_sim.model,
            self.controller.wrench_sim.data,
            self.left_site_id,
            self.right_site_id,
            ball_pos=ball_pos,
        )
        if self.runtime.phase == "prep":
            if t < float(self.cfg.prep_duration):
                self.motor_cmd = _interpolate_action(
                    t, self.prep_time, self.prep_action
                )
            else:
                self.runtime.phase = "kneel"
                self.runtime.phase_start_time = t
                _ensure_default_hand_rotvec(
                    self.runtime,
                    self.controller.wrench_sim.data,
                    self.left_site_id,
                    self.right_site_id,
                )
                self.motor_cmd = self.runtime.kneel_action_arr[0].copy()
                print("[model_based] Phase transition: prep -> kneel")
        elif self.runtime.phase == "kneel":
            _ensure_default_hand_rotvec(
                self.runtime,
                self.controller.wrench_sim.data,
                self.left_site_id,
                self.right_site_id,
            )
            elapsed = max(float(t - self.cfg.prep_duration), 0.0)
            idx = int(
                np.clip(
                    elapsed / max(self.control_dt, 1e-6),
                    0,
                    len(self.runtime.kneel_action_arr) - 1,
                )
            )
            self.motor_cmd = self.runtime.kneel_action_arr[idx].copy()
            if idx >= len(self.runtime.kneel_action_arr) - 1:
                self.runtime.phase = "approach"
                self.runtime.phase_start_time = t
                self.runtime.reach_init_state = False
                self.runtime.contact_reach_time = None
                self.runtime.rigid_body_center = None
                self.runtime.rigid_body_orientation = None
                self.runtime.hand_offsets_in_body_frame = None
                self.runtime.delta_goal_angular_velocity = np.zeros(3, dtype=np.float64)
                self.runtime.expected_ball_pos = None
                self.runtime.goal_time = None
                self.motor_cmd = self.runtime.kneel_action_arr[-1].copy()
                _reset_pose_command_to_current_sites(
                    self.runtime,
                    self.controller.wrench_sim.data,
                    self.left_site_id,
                    self.right_site_id,
                )
                _reset_approach_interp(self.runtime)
                self.runtime.wrench_command[:] = 0.0
                current_motor_pos = np.asarray(obs.motor_pos, dtype=np.float64).copy()
                sync_compliance_state_to_current_pose(
                    self.controller,
                    self.controller.wrench_sim.data,
                    current_motor_pos,
                )
                self.latest_state_ref = self.controller._last_state
                print("[model_based] Phase transition: kneel -> approach")
        elif self.runtime.phase == "approach":
            reached = _run_approach_phase(
                self.runtime, self.cfg, state, self.control_dt
            )
            approach_timed_out = float(self.cfg.approach_timeout) > 0.0 and (
                t - self.runtime.phase_start_time
            ) >= float(self.cfg.approach_timeout)
            if (not reached) and approach_timed_out:
                reached = True
            command_matrix = _build_command_matrix(
                self.runtime, self.measured_wrenches, self.site_names
            )
            next_motor_cmd, state_ref = self.apply_controller_step(
                command_matrix=command_matrix.astype(np.float32),
                target_motor_pos=np.asarray(self.motor_cmd, dtype=np.float32),
                measured_wrenches=self.measured_wrenches,
                site_names=self.site_names,
                motor_torques=obs.motor_tor,
                qpos=np.asarray(self.controller.wrench_sim.data.qpos, dtype=np.float32),
                controlled_actuators_only=True,
            )
            self.motor_cmd = np.asarray(next_motor_cmd, dtype=np.float64)
            if state_ref is not None:
                self.latest_state_ref = state_ref
            if reached:
                if self.runtime.contact_reach_time is None:
                    self.runtime.contact_reach_time = t
                if (
                    t - self.runtime.contact_reach_time
                    >= self.cfg.contact_wait_duration
                ):
                    self.runtime.phase = "model_based"
                    self.runtime.phase_start_time = t
                    self.runtime.model_based_start_time = t
                    self.runtime.last_ochs_print_time = None
                    self.runtime.rigid_body_center = None
                    self.runtime.rigid_body_orientation = None
                    self.runtime.hand_offsets_in_body_frame = None
                    self.runtime.expected_ball_pos = np.asarray(
                        state["ball_pos"], dtype=np.float64
                    ).copy()
                    print("[model_based] Phase transition: approach -> model_based")
        elif self.runtime.phase == "model_based":
            command = _poll_keyboard_command(self.control_receiver)
            mode_changed = _update_goal_from_keyboard_and_time(
                self.runtime, self.cfg, t, command
            )
            if mode_changed:
                self.runtime.phase = "approach"
                self.runtime.phase_start_time = t
                self.runtime.last_ochs_print_time = None
                self.runtime.reach_init_state = False
                self.runtime.contact_reach_time = None
                self.runtime.rigid_body_center = None
                self.runtime.rigid_body_orientation = None
                self.runtime.hand_offsets_in_body_frame = None
                _reset_approach_interp(self.runtime)
                self.runtime.wrench_command[:] = 0.0
            else:
                total_angular_velocity_vec = (
                    np.asarray(self.runtime.goal_rotate_axis, dtype=np.float64)
                    * float(self.runtime.goal_angular_velocity)
                    + self.runtime.delta_goal_angular_velocity
                )
                total_angular_velocity_mag = float(
                    np.linalg.norm(total_angular_velocity_vec)
                )
                if total_angular_velocity_mag < 1e-9:
                    total_angular_velocity_dir = np.asarray(
                        self.runtime.goal_rotate_axis, dtype=np.float64
                    )
                else:
                    total_angular_velocity_dir = (
                        total_angular_velocity_vec / total_angular_velocity_mag
                    )
                min_hand_force = (
                    self.cfg.min_hand_normal_force_both
                    if self.runtime.active_hands_mode == "both"
                    else self.cfg.min_hand_normal_force_single
                )
                jacobian_fn = self.jacobian_by_mode[self.runtime.active_hands_mode]
                ochs_inputs = compute_ochs_inputs(
                    state,
                    goal_angular_velocity=total_angular_velocity_mag,
                    goal_rotate_axis=total_angular_velocity_dir,
                    friction_coeff_ground=self.cfg.friction_coeff_ground,
                    friction_coeff_hand=self.cfg.friction_coeff_hand,
                    kMinHandNormalForce=min_hand_force,
                    active_hands=self.runtime.active_hands_mode,
                    jacobian=jacobian_fn,
                    kBallMass=self.cfg.rolling_ball_mass,
                    kBallRadius=self.cfg.ball_radius,
                )
                ochs_solution = solve_ochs(*ochs_inputs, kNumSeeds=1, kPrintLevel=0)
                distributed_motion = _distribute_rigid_body_motion(
                    self.runtime, ochs_solution, state, self.control_dt
                )
                _maybe_print_ochs_world_velocity()
                _assign_stiffness(
                    self.runtime,
                    self.cfg,
                    distributed_motion["left_linvel"],
                    distributed_motion["right_linvel"],
                )
                _integrate_pose_command(
                    self.runtime, self.cfg, distributed_motion, self.control_dt
                )
                _update_expected_ball_pos(
                    self.runtime, self.cfg, self.control_dt, state
                )
                _update_delta_goal(self.runtime, self.cfg, state)
                command_matrix = _build_command_matrix(
                    self.runtime, self.measured_wrenches, self.site_names
                )
                next_motor_cmd, state_ref = self.apply_controller_step(
                    command_matrix=command_matrix.astype(np.float32),
                    target_motor_pos=np.asarray(self.motor_cmd, dtype=np.float32),
                    measured_wrenches=self.measured_wrenches,
                    site_names=self.site_names,
                    motor_torques=obs.motor_tor,
                    qpos=np.asarray(
                        self.controller.wrench_sim.data.qpos, dtype=np.float32
                    ),
                    controlled_actuators_only=True,
                )
                self.motor_cmd = np.asarray(next_motor_cmd, dtype=np.float64)
                if state_ref is not None:
                    self.latest_state_ref = state_ref
        else:
            raise ValueError(f"Unknown phase: {self.runtime.phase}")

        _update_mocap_targets_from_state_ref(
            self.controller.wrench_sim.data,
            self.left_mocap_id,
            self.right_mocap_id,
            self.latest_state_ref,
        )
        if (
            self.runtime.phase in ("approach", "model_based")
            and self.hold_indices.size > 0
        ):
            self.motor_cmd[self.hold_indices] = self.kneel_hold_motor_pos[
                self.hold_indices
            ]

        return np.asarray(self.motor_cmd, dtype=np.float32).copy()

    def close(self, exp_folder_path: str = "") -> None:
        if self.control_receiver is not None and hasattr(
            self.control_receiver, "close"
        ):
            self.control_receiver.close()
        super().close(exp_folder_path=exp_folder_path)
