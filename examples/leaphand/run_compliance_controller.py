import os
import time
from dataclasses import dataclass
from typing import Optional, Sequence

import gin
import mujoco
import numpy as np
import yaml

from minimalist_compliance_control.controller import (
    ComplianceController,
    ComplianceInputs,
    ComplianceRefConfig,
    ControllerConfig,
)
from minimalist_compliance_control.reference.compliance_ref import COMMAND_LAYOUT
from minimalist_compliance_control.wrench_estimation import WrenchEstimateConfig


@gin.configurable
@dataclass
class RunnerConfig:
    kp_pos: Optional[float] = None
    kp_rot: Optional[float] = None
    wrench_debug_site: Optional[str] = None
    initial_pose: Sequence[Sequence[float]] = ()


def _deep_update(d, u):
    for k, v in u.items():
        if isinstance(v, dict) and isinstance(d.get(k), dict):
            _deep_update(d[k], v)
        else:
            d[k] = v
    return d


@gin.configurable
@dataclass
class MotorConfigPaths:
    default_config_path: Optional[str] = None
    robot_config_path: Optional[str] = None
    motors_config_path: Optional[str] = None


def _load_motor_params(
    model: mujoco.MjModel,
    motor_paths: MotorConfigPaths,
):
    if (
        not motor_paths.default_config_path
        or not motor_paths.robot_config_path
        or not motor_paths.motors_config_path
    ):
        raise ValueError(
            "MotorConfigPaths.default_config_path, robot_config_path, and motors_config_path must be configured."
        )
    with open(motor_paths.default_config_path, "r") as f:
        config = yaml.safe_load(f)
    with open(motor_paths.robot_config_path, "r") as f:
        robot_cfg = yaml.safe_load(f)
    if robot_cfg is not None:
        _deep_update(config, robot_cfg)
    with open(motor_paths.motors_config_path, "r") as f:
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
        motor_cfg = config["motors"][motor_key]
        motor_type = motor_cfg["motor"]
        act_cfg = config["actuators"][motor_type]
        kp.append(float(motor_cfg.get("kp", 0.0)) / kp_ratio)
        kd.append(float(motor_cfg.get("kd", 0.0)) / kd_ratio)
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


def main() -> None:
    config_path = os.path.join(os.path.dirname(__file__), "config.gin")
    gin.clear_config()
    gin.parse_config_file(config_path)
    runner_cfg = RunnerConfig()
    motor_cfg_paths = MotorConfigPaths()
    if runner_cfg.kp_pos is None or runner_cfg.kp_rot is None:
        raise ValueError("RunnerConfig.kp_pos and RunnerConfig.kp_rot must be configured.")
    controller = ComplianceController(
        config=ControllerConfig(),
        estimate_config=WrenchEstimateConfig(),
        ref_config=ComplianceRefConfig(),
    )

    model = controller.wrench_sim.model
    data = controller.wrench_sim.data

    if controller.compliance_ref is None:
        return

    data.qpos[:] = controller.compliance_ref.default_qpos.copy()
    mujoco.mj_forward(model, data)

    num_sites = len(controller.config.site_names)
    command_matrix = np.zeros((num_sites, COMMAND_LAYOUT.width), dtype=np.float32)
    default_state = controller.compliance_ref.get_default_state()
    init_pose_arr = np.asarray(runner_cfg.initial_pose, dtype=np.float32)
    if init_pose_arr.shape != (num_sites, 6):
        raise ValueError(
            f"RunnerConfig.initial_pose must be shape ({num_sites}, 6), got {init_pose_arr.shape}."
        )
    init_pos = init_pose_arr[:, :3]
    init_ori = init_pose_arr[:, 3:6]
    init_pose = np.concatenate([init_pos, init_ori], axis=1)
    command_matrix[:, COMMAND_LAYOUT.position] = init_pos
    command_matrix[:, COMMAND_LAYOUT.orientation] = init_ori
    default_state["x_ref"] = init_pose.copy()
    default_state["x_ref_unprojected"] = init_pose.copy()
    default_state["v_ref"] = np.zeros_like(default_state["v_ref"])
    default_state["a_ref"] = np.zeros_like(default_state["a_ref"])
    controller._last_state = default_state
    kp_pos = float(runner_cfg.kp_pos)
    kp_rot = float(runner_cfg.kp_rot)
    mass = float(controller.ref_config.mass)
    inertia_diag = np.asarray(controller.ref_config.inertia_diag, dtype=np.float32)
    kd_pos = 2.0 * np.sqrt(mass * kp_pos)
    kd_rot = 2.0 * np.sqrt(inertia_diag * kp_rot)
    command_matrix[:, COMMAND_LAYOUT.kp_pos] = (
        np.eye(3, dtype=np.float32) * kp_pos
    ).reshape(-1)
    command_matrix[:, COMMAND_LAYOUT.kd_pos] = (
        np.eye(3, dtype=np.float32) * kd_pos
    ).reshape(-1)
    command_matrix[:, COMMAND_LAYOUT.kp_rot] = (
        np.eye(3, dtype=np.float32) * kp_rot
    ).reshape(-1)
    command_matrix[:, COMMAND_LAYOUT.kd_rot] = (
        np.diag(kd_rot).astype(np.float32).reshape(-1)
    )

    trnid = np.asarray(model.actuator_trnid[:, 0], dtype=np.int32)
    qpos_adr = model.jnt_qposadr[trnid]
    qvel_adr = model.jnt_dofadr[trnid]

    sim_dt = float(model.opt.timestep)
    control_dt = float(controller.ref_config.dt)
    next_control_time = 0.0
    target_motor_pos = np.asarray(default_state["motor_pos"], dtype=np.float32)

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
    ) = _load_motor_params(model, motor_cfg_paths)

    t0 = time.time()
    debug_site: Optional[str] = (
        str(runner_cfg.wrench_debug_site).strip()
        if runner_cfg.wrench_debug_site is not None
        else None
    )
    if debug_site == "":
        debug_site = None
    while True:
        t = time.time() - t0
        if t >= next_control_time:
            command_matrix[:, COMMAND_LAYOUT.position] = init_pos
            command_matrix[:, COMMAND_LAYOUT.orientation] = init_ori
            inputs = ComplianceInputs(
                motor_torques=np.asarray(data.actuator_force, dtype=np.float32),
                qpos=np.asarray(data.qpos, dtype=np.float32),
                time=t,
                command_matrix=command_matrix,
            )
            out = controller.step(inputs, use_estimated_wrench=True)
            if "state_ref" in out:
                target_motor_pos = np.asarray(out["state_ref"]["motor_pos"])
            if "wrenches" in out and debug_site is not None:
                est = out["wrenches"].get(debug_site)
                if est is not None:
                    site_id = controller.wrench_sim.site_ids[debug_site]
                    body_id = int(model.site_bodyid[site_id])
                    applied_force = np.asarray(
                        data.xfrc_applied[body_id][:3], dtype=np.float32
                    )
                    tf = np.round(applied_force, 3)
                    ef = np.round(est[:3].astype(np.float32), 3)
                    print(f"{debug_site} applied_force {tf} est_force {ef}")
            next_control_time += control_dt

        q = data.qpos[qpos_adr]
        q_dot = data.qvel[qvel_adr]
        q_dot_dot = data.qacc[qvel_adr]
        error = target_motor_pos - q
        real_kp = np.where(q_dot_dot * error < 0, kp * passive_active_ratio, kp)
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
        data.ctrl[:] = tau_m_clamped

        mujoco.mj_step(model, data)
        controller.wrench_sim.visualize()
        if controller.wrench_sim.viewer is None:
            break
        time.sleep(sim_dt)


if __name__ == "__main__":
    main()
