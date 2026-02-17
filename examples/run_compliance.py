import argparse
import os
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Sequence

import gin
import numpy as np

from minimalist_compliance_control.compliance_ref import COMMAND_LAYOUT
from minimalist_compliance_control.controller import (
    ComplianceController,
    ComplianceRefConfig,
    ControllerConfig,
)
from minimalist_compliance_control.real_world import RealWorld
from minimalist_compliance_control.sim import (
    BaseSim,
    MuJoCoSim,
    build_clamped_torque_substep_control,
    build_site_force_applier,
)
from minimalist_compliance_control.utils import (
    KeyboardListener,
    KeyboardTeleop,
    load_motor_params,
)
from minimalist_compliance_control.visualization import CompliancePlotter
from minimalist_compliance_control.wrench_estimation import WrenchEstimateConfig


@gin.configurable
@dataclass
class RunnerConfig:
    kp_pos: Optional[float] = None
    kp_rot: Optional[float] = None
    initial_pose: Sequence[Sequence[float]] = ()


@gin.configurable
@dataclass
class MotorConfigPaths:
    default_config_path: Optional[str] = None
    robot_config_path: Optional[str] = None
    motors_config_path: Optional[str] = None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run compliance control for leap or toddlerbot."
    )
    parser.add_argument(
        "--sim",
        choices=["mujoco", "real"],
        default="mujoco",
        help="Backend to run.",
    )
    parser.add_argument(
        "--vis",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable MuJoCo interactive viewer (mujoco backend only).",
    )
    parser.add_argument(
        "--robot",
        choices=["toddlerbot", "leap"],
        required=True,
        help="Robot target.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        default=False,
        help="Enable live Matplotlib plots for compliance and estimated wrench.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    os.chdir(repo_root)

    config_name = "leap.gin" if args.robot == "leap" else "toddlerbot.gin"
    config_path = os.path.join(repo_root, "config", config_name)
    gin.clear_config()
    gin.parse_config_file(config_path)
    gin.bind_parameter("WrenchSimConfig.view", bool(args.vis and args.sim == "mujoco"))
    gin.bind_parameter("WrenchSimConfig.render", False)

    runner_cfg = RunnerConfig()
    motor_cfg_paths = MotorConfigPaths()
    if runner_cfg.kp_pos is None or runner_cfg.kp_rot is None:
        raise ValueError(
            "RunnerConfig.kp_pos and RunnerConfig.kp_rot must be configured."
        )

    controller = ComplianceController(
        config=ControllerConfig(),
        estimate_config=WrenchEstimateConfig(),
        ref_config=ComplianceRefConfig(),
    )

    if controller.compliance_ref is None:
        raise ValueError("Controller compliance_ref must be configured.")

    model = controller.wrench_sim.model
    data = controller.wrench_sim.data
    control_dt = float(controller.ref_config.dt)

    if not motor_cfg_paths.default_config_path or not motor_cfg_paths.robot_config_path:
        raise ValueError(
            "MotorConfigPaths.default_config_path and robot_config_path must be configured."
        )
    if args.robot == "leap" and not motor_cfg_paths.motors_config_path:
        raise ValueError(
            "MotorConfigPaths.motors_config_path must be configured for leap."
        )

    num_sites = len(controller.config.site_names)
    teleop = KeyboardTeleop(
        num_sites=num_sites, site_names=controller.config.site_names
    )
    print(
        "[teleop] keys: w/x:+/-x, a/d:+/-y, q/z:+/-z, "
        "p=toggle pos/rot, n=next site, r=reset site, f=toggle random force"
    )
    print("[teleop] focus the terminal (stdin) for keyboard controls.")
    key_listener = KeyboardListener(teleop)
    key_listener.start()

    command_matrix = np.zeros((num_sites, COMMAND_LAYOUT.width), dtype=np.float32)
    default_state = controller.compliance_ref.get_default_state()

    if args.robot == "leap":
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
        default_state.x_ref = init_pose.copy()
        default_state.v_ref = np.zeros_like(default_state.v_ref)
        default_state.a_ref = np.zeros_like(default_state.a_ref)
        controller._last_state = default_state
        pos_cmd = init_pos
        ori_cmd = init_ori
    else:
        home_pose = np.asarray(default_state.x_ref, dtype=np.float32)
        pos_cmd = home_pose[:, :3]
        ori_cmd = home_pose[:, 3:6]
        command_matrix[:, COMMAND_LAYOUT.position] = pos_cmd
        command_matrix[:, COMMAND_LAYOUT.orientation] = ori_cmd

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
    force_site_names = tuple(controller.config.site_names)
    force_site_ids = np.asarray(
        [controller.wrench_sim.site_ids[name] for name in force_site_names],
        dtype=np.int32,
    )
    force_rng = np.random.default_rng()
    force_max = 3.0
    force_vis_scale = 0.1
    perturb_site_forces = np.zeros((len(force_site_names), 3), dtype=np.float32)
    force_active = False
    force_phase_end_time = 0.0
    force_pause_duration = 1.0
    site_force_applier = None
    base_pos_cmd = pos_cmd.copy()
    base_ori_cmd = ori_cmd.copy()
    plotter: Optional[CompliancePlotter] = None
    if args.plot:
        run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = os.path.join(
            repo_root, "results", f"{args.robot}_compliance_{run_stamp}"
        )
        os.makedirs(results_dir, exist_ok=True)
        print(f"[plot] writing PNGs to: {results_dir}")
        plotter = CompliancePlotter(
            site_names=force_site_names,
            enabled=True,
            output_dir=results_dir,
        )
        if not plotter.enabled and plotter.error_message:
            print(f"[plot] disabled: {plotter.error_message}")

    target_motor_pos = np.asarray(default_state.motor_pos, dtype=np.float32)
    motor_params = None
    if args.sim == "mujoco":
        motor_params = load_motor_params(
            model=model,
            default_config_path=motor_cfg_paths.default_config_path,
            robot_config_path=motor_cfg_paths.robot_config_path,
            motors_config_path=motor_cfg_paths.motors_config_path,
        )
        torque_substep_control = build_clamped_torque_substep_control(
            qpos_adr=qpos_adr,
            qvel_adr=qvel_adr,
            motor_params=motor_params,
            target_motor_pos_getter=lambda: target_motor_pos,
        )
        site_force_applier = build_site_force_applier(
            model=model,
            site_ids=force_site_ids,
        )
        sim: BaseSim = MuJoCoSim(
            controller.wrench_sim,
            control_dt=control_dt,
            sim_dt=float(model.opt.timestep),
            vis=bool(args.vis),
            substep_control=torque_substep_control,
        )
        data.qpos[:] = controller.compliance_ref.default_qpos.copy()
        controller.wrench_sim.forward()
    else:
        sim = RealWorld(
            controller.wrench_sim,
            control_dt=control_dt,
            default_config_path=motor_cfg_paths.default_config_path,
            robot_config_path=motor_cfg_paths.robot_config_path,
            motors_config_path=motor_cfg_paths.motors_config_path,
            vis=bool(args.vis),
        )

    t0 = time.monotonic()
    next_tick = t0

    try:
        while True:
            pos_offsets, rot_offsets, force_enabled = teleop.snapshot()
            pos_cmd = base_pos_cmd + pos_offsets
            ori_cmd = base_ori_cmd + rot_offsets
            now = time.monotonic()
            if force_enabled:
                if now >= force_phase_end_time:
                    if force_active:
                        force_active = False
                        perturb_site_forces[:] = 0.0
                        force_phase_end_time = now + force_pause_duration
                    else:
                        force_active = True
                        num_sites = len(force_site_names)
                        if num_sites <= 0:
                            perturb_site_forces[:] = 0.0
                        else:
                            vec = force_rng.normal(size=(num_sites, 3)).astype(
                                np.float32
                            )
                            norms = np.linalg.norm(vec, axis=1, keepdims=True)
                            norms = np.maximum(norms, 1e-6)
                            direction = vec / norms
                            magnitudes = force_rng.uniform(
                                0.0, float(force_max), size=(num_sites, 1)
                            ).astype(np.float32)
                            perturb_site_forces[:] = direction * magnitudes
                        force_norms = np.linalg.norm(perturb_site_forces, axis=1)
                        non_zero = np.flatnonzero(force_norms > 1e-6)
                        if non_zero.size > 0:
                            desc = ", ".join(
                                f"{force_site_names[i]}:{force_norms[i]:.2f}N"
                                for i in non_zero
                            )
                            print(f"[force] sampled -> {desc}")
                        force_phase_end_time = now + float(force_rng.uniform(1.0, 5.0))
                elif not force_active:
                    perturb_site_forces[:] = 0.0
            else:
                force_active = False
                force_phase_end_time = now
                perturb_site_forces[:] = 0.0
            if args.sim == "mujoco":
                if site_force_applier is not None:
                    site_force_applier(data, perturb_site_forces)
                controller.wrench_sim.set_debug_site_targets(
                    {
                        site_name: np.concatenate(
                            [pos_cmd[idx], ori_cmd[idx]], axis=0
                        ).astype(np.float32)
                        for idx, site_name in enumerate(force_site_names)
                    }
                )
                controller.wrench_sim.set_debug_site_forces(
                    {
                        site_name: perturb_site_forces[idx]
                        for idx, site_name in enumerate(force_site_names)
                    },
                    vis_scale=force_vis_scale,
                )
            command_matrix[:, COMMAND_LAYOUT.position] = pos_cmd
            command_matrix[:, COMMAND_LAYOUT.orientation] = ori_cmd
            obs = sim.get_observation()
            qpos_obs = np.asarray(obs.get("qpos", data.qpos), dtype=np.float32)
            motor_tor_obs = np.asarray(obs["motor_tor"], dtype=np.float32)
            _wrenches, state_ref = controller.step(
                command_matrix=command_matrix,
                motor_torques=motor_tor_obs,
                qpos=qpos_obs,
            )
            if plotter is not None:
                plotter.update_from_wrench_sim(
                    time_s=float(obs.get("time", 0.0)),
                    command_pose=np.concatenate([pos_cmd, ori_cmd], axis=1).astype(
                        np.float32
                    ),
                    wrenches=_wrenches,
                    applied_site_forces=(
                        perturb_site_forces if args.sim == "mujoco" else None
                    ),
                    wrench_sim=controller.wrench_sim,
                    x_ref=(
                        np.asarray(state_ref.x_ref, dtype=np.float32)
                        if state_ref is not None
                        else None
                    ),
                    x_ik=(
                        np.asarray(state_ref.x_ik, dtype=np.float32)
                        if state_ref is not None
                        else None
                    ),
                )
            if state_ref is not None:
                target_motor_pos = np.asarray(state_ref.motor_pos)

            if args.sim == "real":
                data.ctrl[:] = target_motor_pos

            sim.step()
            if not sim.sync():
                break
            next_tick += control_dt
            sleep_s = next_tick - time.monotonic()
            if sleep_s > 0.0:
                time.sleep(sleep_s)
            else:
                next_tick = time.monotonic()
    except KeyboardInterrupt:
        pass
    finally:
        if plotter is not None:
            plotter.close()
        controller.wrench_sim.clear_debug_site_targets()
        controller.wrench_sim.clear_debug_site_forces()
        key_listener.stop()
        sim.close()


if __name__ == "__main__":
    main()
