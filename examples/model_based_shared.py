from __future__ import annotations

from typing import Callable, Optional

import mujoco
import numpy as np
import yaml


def deep_update(dst: dict, src: dict) -> dict:
    for key, value in src.items():
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            deep_update(dst[key], value)
        else:
            dst[key] = value
    return dst


def load_merged_motor_config(
    default_path: str,
    robot_path: str,
    motors_path: Optional[str] = None,
) -> dict:
    with open(default_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    with open(robot_path, "r", encoding="utf-8") as f:
        robot_cfg = yaml.safe_load(f) or {}
    deep_update(config, robot_cfg)
    if motors_path:
        with open(motors_path, "r", encoding="utf-8") as f:
            motor_cfg = yaml.safe_load(f) or {}
        deep_update(config, motor_cfg)
    return config


def load_motor_params_from_config(
    model: mujoco.MjModel,
    config: dict,
    *,
    allow_act_suffix: bool = False,
    dtype=np.float64,
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
    kp_ratio = float(config["actuators"]["kp_ratio"])
    kd_ratio = float(config["actuators"]["kd_ratio"])
    passive_active_ratio = float(config["actuators"]["passive_active_ratio"])

    names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        for i in range(int(model.nu))
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
        if name is None:
            raise ValueError("Actuator without a name is not supported.")
        motor_key = str(name)
        if motor_key not in config["motors"] and allow_act_suffix and motor_key.endswith(
            "_act"
        ):
            base_key = motor_key[: -len("_act")]
            if base_key in config["motors"]:
                motor_key = base_key
        if motor_key not in config["motors"]:
            raise ValueError(f"Missing motor config for actuator '{motor_key}'")

        motor_cfg = config["motors"][motor_key]
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
        np.asarray(kp, dtype=dtype),
        np.asarray(kd, dtype=dtype),
        np.asarray(tau_max, dtype=dtype),
        np.asarray(q_dot_max, dtype=dtype),
        np.asarray(tau_q_dot_max, dtype=dtype),
        np.asarray(q_dot_tau_max, dtype=dtype),
        np.asarray(tau_brake_max, dtype=dtype),
        np.asarray(kd_min, dtype=dtype),
        passive_active_ratio,
    )


def make_clamped_torque_substep_control(
    *,
    qpos_adr: np.ndarray,
    qvel_adr: np.ndarray,
    target_motor_pos_getter: Callable[[], np.ndarray],
    kp: np.ndarray,
    kd: np.ndarray,
    tau_max: np.ndarray,
    q_dot_max: np.ndarray,
    tau_q_dot_max: np.ndarray,
    q_dot_tau_max: np.ndarray,
    tau_brake_max: np.ndarray,
    kd_min: np.ndarray,
    passive_active_ratio: float,
    extra_substep_fn: Optional[Callable[[mujoco.MjData], None]] = None,
) -> Callable[[mujoco.MjData], None]:
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


def get_ground_truth_wrenches(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    site_names: tuple[str, ...],
) -> dict[str, np.ndarray]:
    """Return per-site body cfrc_ext as [force(3), torque(3)]."""
    mujoco.mj_rnePostConstraint(model, data)
    wrenches: dict[str, np.ndarray] = {}
    for site_name in site_names:
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if site_id < 0:
            continue
        body_id = int(model.site_bodyid[site_id])
        raw_spatial = np.asarray(data.cfrc_ext[body_id], dtype=np.float32).reshape(-1)
        if raw_spatial.shape[0] >= 6:
            wrenches[site_name] = np.concatenate(
                [raw_spatial[3:6], raw_spatial[0:3]], axis=0
            ).astype(np.float32, copy=False)
        else:
            wrenches[site_name] = np.zeros(6, dtype=np.float32)
    return wrenches
