#!/usr/bin/env python3
"""Model-based compliance policy (minimal, but phase-complete).

This script keeps the runtime lightweight while directly using copied OCHS
components from toddlerbot_internal under `examples/model_based/hybrid_servo`.

Implemented control phases:
1) interpolate to default motor pose (prep)
2) kneel trajectory playback
3) approach-to-contact
4) model-based OCHS compliance control
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import gin
import mujoco
import numpy as np
import numpy.typing as npt
import yaml
from scipy.spatial.transform import Rotation as R

SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
EXAMPLE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if EXAMPLE_DIR not in sys.path:
    sys.path.insert(0, EXAMPLE_DIR)

try:
    from utils.zmq_control import KeyboardControlReceiver
except Exception:
    KeyboardControlReceiver = None

try:
    from hybrid_servo.algorithm.ochs import solve_ochs
    from hybrid_servo.algorithm.solvehfvc import HFVC, transform_hfvc_to_global
    from hybrid_servo.demo.two_hand_rotate_ball.ochs_helpers import (
        compute_center_quaternion_from_hands,
        compute_ochs_inputs,
        generate_constraint_jacobian,
    )
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "Missing model-based dependencies. Install with: pip install qpsolvers osqp sympy"
    ) from exc

from minimalist_compliance_control.controller import (
    ComplianceController,
    ComplianceInputs,
    ComplianceRefConfig,
    ControllerConfig,
)
from minimalist_compliance_control.reference.compliance_ref import COMMAND_LAYOUT
from minimalist_compliance_control.wrench_estimation import WrenchEstimateConfig

Array = npt.NDArray[np.float64]

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
    distance_threshold_margin: float = 0.015
    contact_wait_duration: float = 0.0
    pid_kp: float = 0.0
    kneel_motion_file: str = "utils/kneel_2xm.lz4"
    initial_active_hands_mode: str = "left"
    threshold_angle: float = np.pi / 5.0
    threshold_angle_z: float = np.pi / 4.0
    keyboard_control_port: int = 5592
    print_ochs_world_velocity: bool = True
    ochs_print_interval: float = 0.2


@dataclass
class PolicyRuntime:
    """Mutable runtime state for model-based policy integration."""

    pose_command: Array
    wrench_command: Array
    pos_stiffness: Array
    pos_damping: Array
    rot_stiffness: Array
    rot_damping: Array
    delta_goal_angular_velocity: Array

    phase: str
    phase_start_time: float

    active_hands_mode: str
    active_hand_indices: Tuple[int, ...]

    goal_rotate_axis: Array
    goal_angular_velocity: float
    goal_speed: float
    goal_angle: float
    goal_time: Optional[float]

    kneel_action_arr: Array
    kneel_qpos: Array
    kneel_qpos_source_dim: int

    reach_init_state: bool = False
    contact_reach_time: Optional[float] = None
    model_based_start_time: Optional[float] = None

    approach_progress: dict[int, float] | None = None
    approach_start_pose: dict[int, Optional[Array]] | None = None

    default_left_hand_center_rotvec: Optional[Array] = None
    default_right_hand_center_rotvec: Optional[Array] = None

    rigid_body_center: Optional[Array] = None
    rigid_body_orientation: Optional[Array] = None  # wxyz
    hand_offsets_in_body_frame: Optional[Array] = None

    expected_ball_pos: Optional[Array] = None
    last_ochs_print_time: Optional[float] = None


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


def _goal_axis_from_mode(mode: str) -> Array:
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
    p_start: Array, p_end: Array, duration: float, t: float
) -> Array:
    if t <= 0.0:
        return p_start
    if t >= duration:
        return p_end
    return p_start + (p_end - p_start) * (t / duration)


def _binary_search(arr: Array, t: float) -> int:
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


def _interpolate_action(t: float, time_arr: Array, action_arr: Array) -> Array:
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
    init_motor_pos: Array,
    target_motor_pos: Array,
    prep_duration: float,
    control_dt: float,
    prep_hold_duration: float,
) -> tuple[Array, Array]:
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


def _sensor_data(model: mujoco.MjModel, data: mujoco.MjData, name: str) -> Array:
    sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
    if sensor_id < 0:
        raise KeyError(f"Sensor '{name}' not found.")
    start = int(model.sensor_adr[sensor_id])
    end = start + int(model.sensor_dim[sensor_id])
    return np.asarray(data.sensordata[start:end], dtype=np.float64).copy()


def _deep_update(d: dict, u: dict) -> dict:
    for k, v in u.items():
        if isinstance(v, dict) and isinstance(d.get(k), dict):
            _deep_update(d[k], v)
        else:
            d[k] = v
    return d


def _find_repo_root(start_dir: str) -> str:
    curr = os.path.abspath(start_dir)
    while True:
        if os.path.isfile(os.path.join(curr, "pyproject.toml")):
            return curr
        parent = os.path.dirname(curr)
        if parent == curr:
            raise FileNotFoundError("Could not find repository root containing pyproject.toml.")
        curr = parent


def _build_contact_state(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    left_site_id: int,
    right_site_id: int,
) -> Dict[str, Array]:
    state: Dict[str, Array] = {
        "ball_pos": _sensor_data(model, data, "rolling_ball_framepos"),
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
    repo_root = _find_repo_root(SCRIPT_DIR)
    default_path = os.path.join(repo_root, "examples", "descriptions", "default.yml")
    robot_path = os.path.join(
        repo_root,
        "examples",
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


def _load_motor_params(model: mujoco.MjModel) -> tuple[Array, ...]:
    config = _load_robot_motor_config()

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
    default_motor_pos: Array,
    default_qpos: Array,
    motor_dim: int,
    qpos_dim: int,
) -> tuple[Array, Array, int]:
    try:
        import joblib
    except Exception:
        joblib = None

    candidates = [
        os.path.join(example_dir, cfg.kneel_motion_file),
        os.path.join(example_dir, "..", "..", "motion", cfg.kneel_motion_file),
        "/Users/hsb/code_space/toddlerbot_internal/motion/kneel_2xm.lz4",
    ]

    def _adapt_kneel_qpos(raw_qpos_last: Array, source_path: str) -> Array:
        raw = np.asarray(raw_qpos_last, dtype=np.float64).reshape(-1)
        default = np.asarray(default_qpos, dtype=np.float64).reshape(-1)
        if default.shape[0] != qpos_dim:
            raise ValueError(
                f"default_qpos dim {default.shape[0]} != qpos_dim {qpos_dim}"
            )

        if raw.shape[0] == qpos_dim:
            return raw.copy()

        if raw.shape[0] < qpos_dim:
            adapted = default.copy()
            adapted[: raw.shape[0]] = raw
            print(
                "[model_based] Adapted kneel qpos "
                f"{raw.shape[0]} -> {qpos_dim} using default tail from {source_path}"
            )
            return adapted

        adapted = raw[:qpos_dim].copy()
        print(
            "[model_based] Truncated kneel qpos "
            f"{raw.shape[0]} -> {qpos_dim} from {source_path}"
        )
        return adapted

    if joblib is not None:
        for path in candidates:
            path_abs = os.path.abspath(path)
            if not os.path.exists(path_abs):
                continue
            try:
                data = joblib.load(path_abs)
                action_arr = np.asarray(data["action"], dtype=np.float64)
                if action_arr.ndim == 1:
                    action_arr = action_arr.reshape(1, -1)
                if action_arr.shape[1] != motor_dim:
                    raise ValueError(
                        f"kneel action dim {action_arr.shape[1]} != motor_dim {motor_dim}"
                    )

                qpos_arr = np.asarray(data["qpos"], dtype=np.float64)
                if qpos_arr.ndim == 1:
                    qpos_last_raw = qpos_arr.copy()
                else:
                    qpos_last_raw = qpos_arr[-1].copy()
                source_qpos_dim = int(np.asarray(qpos_last_raw).reshape(-1).shape[0])
                qpos_last = _adapt_kneel_qpos(qpos_last_raw, path_abs)

                print(f"[model_based] Loaded kneel trajectory: {path_abs}")
                return action_arr, qpos_last, source_qpos_dim
            except Exception as exc:
                print(
                    f"[model_based] Failed to load kneel trajectory {path_abs}: {exc}"
                )

    print(
        "[model_based] Kneel trajectory unavailable, using single-step fallback."
    )
    fallback_action = np.asarray(default_motor_pos, dtype=np.float64).reshape(1, -1)
    fallback_qpos = np.asarray(default_qpos, dtype=np.float64).copy()
    return fallback_action, fallback_qpos, int(fallback_qpos.shape[0])


def _skew_matrix(vec: Array) -> Array:
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


def _interpolate_se3_pose(start_pose: Array, target_pose: Array, alpha: float) -> Array:
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
    default_state: dict,
    cfg: PolicyConfig,
    kneel_action_arr: Array,
    kneel_qpos: Array,
    kneel_qpos_source_dim: int,
) -> PolicyRuntime:
    pose_command = np.asarray(default_state["x_ref"], dtype=np.float64).copy()

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
    ball_pos: Array,
    is_left_hand: bool,
    default_rotvec: Optional[Array],
) -> tuple[Array, R]:
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
    target_pos: Array,
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
    state: Dict[str, Array],
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


def _initialize_rigid_body(runtime: PolicyRuntime, state: Dict[str, Array]) -> None:
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
    state: Dict[str, Array],
    dt: float,
) -> Dict[str, Array]:
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

    result: Dict[str, Array] = {
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


def _maybe_print_ochs_world_velocity(
    runtime: PolicyRuntime,
    cfg: PolicyConfig,
    t: float,
    distributed_motion: Dict[str, Array],
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


def _assign_stiffness(
    runtime: PolicyRuntime, cfg: PolicyConfig, left_vel: Array, right_vel: Array
) -> None:
    def build_diag(vel: Array) -> tuple[Array, Array]:
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
    distributed_motion: Dict[str, Array],
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
    runtime: PolicyRuntime, cfg: PolicyConfig, dt: float, state: Dict[str, Array]
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
    runtime: PolicyRuntime, cfg: PolicyConfig, state: Dict[str, Array]
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
    measured_wrenches: Dict[str, Array],
    site_names: Tuple[str, str],
) -> Array:
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


def _run_compliance_step(
    controller: ComplianceController,
    data: mujoco.MjData,
    t: float,
    command_matrix: Array,
    target_motor_pos: Array,
    measured_wrenches: Dict[str, Array],
    site_names: Tuple[str, str],
) -> tuple[Array, Optional[dict]]:
    inputs = ComplianceInputs(
        motor_torques=np.asarray(data.actuator_force, dtype=np.float32),
        qpos=np.asarray(data.qpos, dtype=np.float32),
        time=float(t),
        command_matrix=command_matrix.astype(np.float32),
    )
    out = controller.step(inputs, use_estimated_wrench=True)

    next_target = target_motor_pos
    state_ref = None
    if "state_ref" in out:
        if isinstance(out["state_ref"], dict):
            state_ref = out["state_ref"]
        state_ref_motor_pos = np.asarray(
            out["state_ref"]["motor_pos"], dtype=np.float64
        )
        next_target = np.asarray(target_motor_pos, dtype=np.float64).copy()
        controlled_actuators = np.asarray(
            controller.compliance_ref.actuator_indices, dtype=np.int32
        )
        next_target[controlled_actuators] = state_ref_motor_pos[controlled_actuators]

    if "wrenches" in out:
        for site in site_names:
            wrench = out["wrenches"].get(site)
            if wrench is not None:
                measured_wrenches[site] = np.asarray(wrench, dtype=np.float64)

    return next_target, state_ref


def _sync_compliance_state_to_current_pose(
    controller: ComplianceController,
    data: mujoco.MjData,
    motor_pos: Array,
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

    ref_state["x_ref"] = x_ref.copy()
    ref_state["x_ref_unprojected"] = x_ref.copy()
    ref_state["v_ref"] = np.zeros_like(x_ref)
    ref_state["a_ref"] = np.zeros_like(x_ref)
    ref_state["qpos"] = np.asarray(data.qpos, dtype=np.float32).copy()
    ref_state["motor_pos"] = np.asarray(motor_pos, dtype=np.float32).copy()
    controller._last_state = ref_state


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
    state_ref: Optional[dict],
) -> None:
    if state_ref is None:
        return
    x_ref = state_ref.get("x_ref")
    if x_ref is None:
        return

    x_ref_arr = np.asarray(x_ref, dtype=np.float64)
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


def main() -> None:
    cfg = PolicyConfig()

    config_path = os.path.join(SCRIPT_DIR, "config.gin")
    gin.clear_config()
    gin.parse_config_file(config_path)
    gin.bind_parameter("WrenchSimConfig.view", True)

    controller = ComplianceController(
        config=ControllerConfig(),
        estimate_config=WrenchEstimateConfig(),
        ref_config=ComplianceRefConfig(),
    )
    if controller.compliance_ref is None:
        raise RuntimeError("Compliance reference failed to initialize.")

    model = controller.wrench_sim.model
    data = controller.wrench_sim.data
    left_mocap_id, right_mocap_id = _resolve_mocap_target_ids(model)

    # Match toddlerbot_internal MuJoCoSim defaults: sim starts from model qpos0 and uses sim_dt=0.001.
    model.opt.timestep = float(cfg.sim_dt)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    site_names = tuple(controller.config.site_names)
    if site_names != ("left_hand_center", "right_hand_center"):
        raise ValueError(
            "This example expects site_names == ('left_hand_center', 'right_hand_center')."
        )

    left_site_id = controller.wrench_sim.site_ids[site_names[0]]
    right_site_id = controller.wrench_sim.site_ids[site_names[1]]

    trnid = np.asarray(model.actuator_trnid[:, 0], dtype=np.int32)
    valid = trnid >= 0
    if not np.all(valid):
        raise ValueError(
            "Actuator without joint mapping is not supported in this example."
        )
    qpos_adr = np.asarray(model.jnt_qposadr[trnid], dtype=np.int32)
    qvel_adr = np.asarray(model.jnt_dofadr[trnid], dtype=np.int32)

    default_state = controller.compliance_ref.get_default_state()

    kneel_action_arr, kneel_qpos, kneel_qpos_source_dim = _load_kneel_trajectory(
        example_dir=EXAMPLE_DIR,
        cfg=cfg,
        default_motor_pos=np.asarray(default_state["motor_pos"], dtype=np.float64),
        default_qpos=np.asarray(default_state["qpos"], dtype=np.float64),
        motor_dim=model.nu,
        qpos_dim=model.nq,
    )

    runtime = _initialize_runtime_from_default_state(
        default_state=default_state,
        cfg=cfg,
        kneel_action_arr=kneel_action_arr,
        kneel_qpos=kneel_qpos,
        kneel_qpos_source_dim=kneel_qpos_source_dim,
    )

    default_motor_pos = np.asarray(default_state["motor_pos"], dtype=np.float64).copy()
    prep_init_motor_pos = np.asarray(data.qpos[qpos_adr], dtype=np.float64).copy()
    target_motor_pos = prep_init_motor_pos.copy()
    latest_state_ref: Optional[dict] = None
    prep_delta_max = float(np.max(np.abs(prep_init_motor_pos - default_motor_pos)))
    prep_delta_mean = float(np.mean(np.abs(prep_init_motor_pos - default_motor_pos)))
    measured_wrenches: Dict[str, Array] = {
        site_names[0]: np.zeros(6, dtype=np.float64),
        site_names[1]: np.zeros(6, dtype=np.float64),
    }

    jacobian_by_mode = {
        "left": generate_constraint_jacobian(num_hands=3),
        "right": generate_constraint_jacobian(num_hands=3),
        "both": generate_constraint_jacobian(num_hands=6),
    }
    control_receiver = (
        KeyboardControlReceiver(port=cfg.keyboard_control_port)
        if KeyboardControlReceiver is not None
        else None
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
    ) = _load_motor_params(model)
    group_indices = _load_motor_group_indices(model)
    neck_indices = group_indices.get("neck", np.zeros(0, dtype=np.int32))
    leg_indices = group_indices.get("leg", np.zeros(0, dtype=np.int32))
    hold_indices = np.unique(np.concatenate([neck_indices, leg_indices])).astype(
        np.int32
    )
    kneel_hold_motor_pos = runtime.kneel_action_arr[-1].copy()
    if hold_indices.size > 0:
        print(
            "[model_based] Locking neck+leg motors to kneel posture "
            f"for indices={hold_indices.tolist()}"
        )
    else:
        print(
            "[model_based] Warning: neck/leg motor groups not found; no posture lock."
        )

    control_dt = float(controller.ref_config.dt)
    sim_dt = float(model.opt.timestep)
    n_frames = max(int(control_dt / max(sim_dt, 1e-9)), 1)
    prep_time, prep_action = _build_prep_traj(
        prep_init_motor_pos,
        default_motor_pos,
        cfg.prep_duration,
        control_dt,
        cfg.prep_hold_duration,
    )

    print(
        "[model_based] Phase = prep "
        f"(initial mode={runtime.active_hands_mode}, axis={runtime.goal_rotate_axis.tolist()})"
    )
    print(
        "[model_based] Prep start "
        f"(max|q_init-q_default|={prep_delta_max:.4f}, mean={prep_delta_mean:.4f})"
    )
    print(
        "[model_based] Timing "
        f"(sim_dt={sim_dt:.4f}, control_dt={control_dt:.4f}, n_frames={n_frames})"
    )

    try:
        while True:
            t = float(data.time)

            state = _build_contact_state(model, data, left_site_id, right_site_id)
            if runtime.phase == "prep":
                if t < float(cfg.prep_duration):
                    target_motor_pos = _interpolate_action(t, prep_time, prep_action)
                else:
                    runtime.phase = "kneel"
                    runtime.phase_start_time = t
                    _ensure_default_hand_rotvec(
                        runtime, data, left_site_id, right_site_id
                    )
                    target_motor_pos = runtime.kneel_action_arr[0].copy()
                    print("[model_based] Phase transition: prep -> kneel")

            elif runtime.phase == "kneel":
                _ensure_default_hand_rotvec(runtime, data, left_site_id, right_site_id)
                elapsed = max(float(t - cfg.prep_duration), 0.0)
                idx = int(
                    np.clip(
                        elapsed / max(control_dt, 1e-6),
                        0,
                        len(runtime.kneel_action_arr) - 1,
                    )
                )
                target_motor_pos = runtime.kneel_action_arr[idx].copy()

                if idx >= len(runtime.kneel_action_arr) - 1:
                    runtime.phase = "approach"
                    runtime.phase_start_time = t
                    runtime.reach_init_state = False
                    runtime.contact_reach_time = None
                    runtime.rigid_body_center = None
                    runtime.rigid_body_orientation = None
                    runtime.hand_offsets_in_body_frame = None
                    runtime.delta_goal_angular_velocity = np.zeros(3, dtype=np.float64)
                    runtime.expected_ball_pos = None
                    runtime.goal_time = None
                    target_motor_pos = runtime.kneel_action_arr[-1].copy()

                    if cfg.kneel_sync_qpos:
                        # toddlerbot_internal syncs kneel qpos here; with scene_ball preserve non-robot tail (ball state).
                        src_dim = int(
                            np.clip(runtime.kneel_qpos_source_dim, 0, model.nq)
                        )
                        if src_dim == model.nq:
                            data.qpos[:] = runtime.kneel_qpos
                        elif src_dim > 0:
                            print(
                                "[model_based] Kneel qpos sync "
                                f"(partial {src_dim}/{model.nq}; preserving tail state)."
                            )
                            data.qpos[:src_dim] = runtime.kneel_qpos[:src_dim]
                        mujoco.mj_forward(model, data)
                        state = _build_contact_state(
                            model, data, left_site_id, right_site_id
                        )

                    _reset_pose_command_to_current_sites(
                        runtime, data, left_site_id, right_site_id
                    )
                    _reset_approach_interp(runtime)
                    runtime.wrench_command[:] = 0.0
                    _sync_compliance_state_to_current_pose(
                        controller, data, target_motor_pos
                    )
                    latest_state_ref = (
                        controller._last_state
                        if isinstance(controller._last_state, dict)
                        else None
                    )
                    print("[model_based] Phase transition: kneel -> approach")

            elif runtime.phase == "approach":
                reached = _run_approach_phase(runtime, cfg, state, control_dt)

                command_matrix = _build_command_matrix(
                    runtime, measured_wrenches, site_names
                )
                target_motor_pos, state_ref = _run_compliance_step(
                    controller,
                    data,
                    t,
                    command_matrix,
                    target_motor_pos,
                    measured_wrenches,
                    site_names,
                )
                if state_ref is not None:
                    latest_state_ref = state_ref

                if reached:
                    if runtime.contact_reach_time is None:
                        runtime.contact_reach_time = t

                    if t - runtime.contact_reach_time >= cfg.contact_wait_duration:
                        runtime.phase = "model_based"
                        runtime.phase_start_time = t
                        runtime.model_based_start_time = t
                        runtime.last_ochs_print_time = None
                        runtime.rigid_body_center = None
                        runtime.rigid_body_orientation = None
                        runtime.hand_offsets_in_body_frame = None
                        runtime.expected_ball_pos = np.asarray(
                            state["ball_pos"], dtype=np.float64
                        ).copy()
                        print(
                            "[model_based] Phase transition: approach -> model_based"
                        )

            elif runtime.phase == "model_based":
                command = _poll_keyboard_command(control_receiver)
                mode_changed = _update_goal_from_keyboard_and_time(
                    runtime, cfg, t, command
                )
                if mode_changed:
                    runtime.phase = "approach"
                    runtime.phase_start_time = t
                    runtime.last_ochs_print_time = None
                    runtime.reach_init_state = False
                    runtime.contact_reach_time = None
                    runtime.rigid_body_center = None
                    runtime.rigid_body_orientation = None
                    runtime.hand_offsets_in_body_frame = None
                    _reset_approach_interp(runtime)
                    runtime.wrench_command[:] = 0.0
                    print(
                        "[model_based] Phase transition: mode switch -> approach"
                    )
                else:
                    total_angular_velocity_vec = (
                        np.asarray(runtime.goal_rotate_axis, dtype=np.float64)
                        * float(runtime.goal_angular_velocity)
                        + runtime.delta_goal_angular_velocity
                    )
                    total_angular_velocity_mag = float(
                        np.linalg.norm(total_angular_velocity_vec)
                    )
                    if total_angular_velocity_mag < 1e-9:
                        total_angular_velocity_dir = np.asarray(
                            runtime.goal_rotate_axis, dtype=np.float64
                        )
                    else:
                        total_angular_velocity_dir = (
                            total_angular_velocity_vec / total_angular_velocity_mag
                        )

                    min_hand_force = (
                        cfg.min_hand_normal_force_both
                        if runtime.active_hands_mode == "both"
                        else cfg.min_hand_normal_force_single
                    )
                    jacobian_fn = jacobian_by_mode[runtime.active_hands_mode]
                    ochs_inputs = compute_ochs_inputs(
                        state,
                        goal_angular_velocity=total_angular_velocity_mag,
                        goal_rotate_axis=total_angular_velocity_dir,
                        friction_coeff_ground=cfg.friction_coeff_ground,
                        friction_coeff_hand=cfg.friction_coeff_hand,
                        kMinHandNormalForce=min_hand_force,
                        active_hands=runtime.active_hands_mode,
                        jacobian=jacobian_fn,
                        kBallMass=cfg.rolling_ball_mass,
                        kBallRadius=cfg.ball_radius,
                    )
                    ochs_solution = solve_ochs(*ochs_inputs, kNumSeeds=1, kPrintLevel=0)

                    distributed_motion = _distribute_rigid_body_motion(
                        runtime,
                        ochs_solution,
                        state,
                        control_dt,
                    )
                    _maybe_print_ochs_world_velocity(
                        runtime, cfg, t, distributed_motion
                    )
                    _assign_stiffness(
                        runtime,
                        cfg,
                        distributed_motion["left_linvel"],
                        distributed_motion["right_linvel"],
                    )
                    _integrate_pose_command(
                        runtime, cfg, distributed_motion, control_dt
                    )
                    _update_expected_ball_pos(runtime, cfg, control_dt, state)
                    _update_delta_goal(runtime, cfg, state)

                    command_matrix = _build_command_matrix(
                        runtime, measured_wrenches, site_names
                    )
                    target_motor_pos, state_ref = _run_compliance_step(
                        controller,
                        data,
                        t,
                        command_matrix,
                        target_motor_pos,
                        measured_wrenches,
                        site_names,
                    )
                    if state_ref is not None:
                        latest_state_ref = state_ref

            else:
                raise ValueError(f"Unknown phase: {runtime.phase}")

            _update_mocap_targets_from_state_ref(
                data,
                left_mocap_id,
                right_mocap_id,
                latest_state_ref,
            )

            if runtime.phase in ("approach", "model_based") and hold_indices.size > 0:
                target_motor_pos[hold_indices] = kneel_hold_motor_pos[hold_indices]

            for _ in range(n_frames):
                q = np.asarray(data.qpos[qpos_adr], dtype=np.float64)
                q_dot = np.asarray(data.qvel[qvel_adr], dtype=np.float64)
                q_dot_dot = np.asarray(data.qacc[qvel_adr], dtype=np.float64)
                error = target_motor_pos - q

                real_kp = np.where(
                    q_dot_dot * error < 0.0, kp * passive_active_ratio, kp
                )
                tau_m = real_kp * error - (kd_min + kd) * q_dot

                abs_q_dot = np.abs(q_dot)
                slope = (tau_q_dot_max - tau_max) / (q_dot_max - q_dot_tau_max)
                taper_limit = tau_max + slope * (abs_q_dot - q_dot_tau_max)
                tau_acc_limit = np.where(
                    abs_q_dot <= q_dot_tau_max, tau_max, taper_limit
                )

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
                data.ctrl[:] = tau_m_clamped
                mujoco.mj_step(model, data)

            controller.wrench_sim.visualize()
            if controller.wrench_sim.viewer is None:
                break
    finally:
        if control_receiver is not None and hasattr(control_receiver, "close"):
            control_receiver.close()


if __name__ == "__main__":
    main()
