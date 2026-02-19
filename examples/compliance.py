from __future__ import annotations

import os
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional, Sequence

import gin
import numpy as np

from minimalist_compliance_control.compliance_ref import COMMAND_LAYOUT
from minimalist_compliance_control.controller import (
    ComplianceController,
    ComplianceRefConfig,
    ControllerConfig,
)
from minimalist_compliance_control.utils import (
    KeyboardListener,
    KeyboardTeleop,
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


class CompliancePolicy:
    """Stateful compliance policy runner used by examples/run_policy.py."""

    def __init__(
        self,
        *,
        robot: str,
        sim: str,
        vis: bool,
        plot: bool,
    ) -> None:
        self.robot = str(robot)
        self.sim_backend = str(sim)
        self.vis = bool(vis)
        self.plot = bool(plot)
        self.repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        os.chdir(self.repo_root)

        config_name = "leap.gin" if self.robot == "leap" else "toddlerbot.gin"
        config_path = os.path.join(self.repo_root, "config", config_name)
        gin.clear_config()
        gin.parse_config_file(config_path)
        gin.bind_parameter(
            "WrenchSimConfig.view", bool(self.vis and self.sim_backend == "mujoco")
        )
        gin.bind_parameter("WrenchSimConfig.render", False)

        self.runner_cfg = RunnerConfig()
        self.motor_cfg_paths = MotorConfigPaths()
        if self.runner_cfg.kp_pos is None or self.runner_cfg.kp_rot is None:
            raise ValueError(
                "RunnerConfig.kp_pos and RunnerConfig.kp_rot must be configured."
            )

        self.controller = ComplianceController(
            config=ControllerConfig(),
            estimate_config=WrenchEstimateConfig(),
            ref_config=ComplianceRefConfig(),
        )
        if self.controller.compliance_ref is None:
            raise ValueError("Controller compliance_ref must be configured.")

        self.model = self.controller.wrench_sim.model
        self.data = self.controller.wrench_sim.data
        self.control_dt = float(self.controller.ref_config.dt)

        if (
            not self.motor_cfg_paths.default_config_path
            or not self.motor_cfg_paths.robot_config_path
        ):
            raise ValueError(
                "MotorConfigPaths.default_config_path and robot_config_path must be configured."
            )
        if self.robot == "leap" and not self.motor_cfg_paths.motors_config_path:
            raise ValueError(
                "MotorConfigPaths.motors_config_path must be configured for leap."
            )

        self.num_sites = len(self.controller.config.site_names)
        self.teleop = KeyboardTeleop(
            num_sites=self.num_sites, site_names=self.controller.config.site_names
        )
        print(
            "[teleop] keys: w/x:+/-x, a/d:+/-y, q/z:+/-z, "
            "p=toggle pos/rot, n=next site, r=reset site, f=toggle random force"
        )
        print("[teleop] focus the terminal (stdin) for keyboard controls.")
        self.key_listener = KeyboardListener(self.teleop)
        self.key_listener.start()

        self.command_matrix = np.zeros(
            (self.num_sites, COMMAND_LAYOUT.width), dtype=np.float32
        )
        self.default_state = self.controller.compliance_ref.get_default_state()

        if self.robot == "leap":
            init_pose_arr = np.asarray(self.runner_cfg.initial_pose, dtype=np.float32)
            if init_pose_arr.shape != (self.num_sites, 6):
                raise ValueError(
                    f"RunnerConfig.initial_pose must be shape ({self.num_sites}, 6), got {init_pose_arr.shape}."
                )
            init_pos = init_pose_arr[:, :3]
            init_ori = init_pose_arr[:, 3:6]
            init_pose = np.concatenate([init_pos, init_ori], axis=1)
            self.command_matrix[:, COMMAND_LAYOUT.position] = init_pos
            self.command_matrix[:, COMMAND_LAYOUT.orientation] = init_ori
            self.default_state.x_ref = init_pose.copy()
            self.default_state.v_ref = np.zeros_like(self.default_state.v_ref)
            self.default_state.a_ref = np.zeros_like(self.default_state.a_ref)
            self.controller._last_state = self.default_state
            self.pos_cmd = init_pos
            self.ori_cmd = init_ori
        else:
            home_pose = np.asarray(self.default_state.x_ref, dtype=np.float32)
            self.pos_cmd = home_pose[:, :3]
            self.ori_cmd = home_pose[:, 3:6]
            self.command_matrix[:, COMMAND_LAYOUT.position] = self.pos_cmd
            self.command_matrix[:, COMMAND_LAYOUT.orientation] = self.ori_cmd

        kp_pos = float(self.runner_cfg.kp_pos)
        kp_rot = float(self.runner_cfg.kp_rot)
        mass = float(self.controller.ref_config.mass)
        inertia_diag = np.asarray(
            self.controller.ref_config.inertia_diag, dtype=np.float32
        )
        kd_pos = 2.0 * np.sqrt(mass * kp_pos)
        kd_rot = 2.0 * np.sqrt(inertia_diag * kp_rot)
        self.command_matrix[:, COMMAND_LAYOUT.kp_pos] = (
            np.eye(3, dtype=np.float32) * kp_pos
        ).reshape(-1)
        self.command_matrix[:, COMMAND_LAYOUT.kd_pos] = (
            np.eye(3, dtype=np.float32) * kd_pos
        ).reshape(-1)
        self.command_matrix[:, COMMAND_LAYOUT.kp_rot] = (
            np.eye(3, dtype=np.float32) * kp_rot
        ).reshape(-1)
        self.command_matrix[:, COMMAND_LAYOUT.kd_rot] = (
            np.diag(kd_rot).astype(np.float32).reshape(-1)
        )

        trnid = np.asarray(self.model.actuator_trnid[:, 0], dtype=np.int32)
        self.qpos_adr = self.model.jnt_qposadr[trnid]
        self.qvel_adr = self.model.jnt_dofadr[trnid]
        self.force_site_names = tuple(self.controller.config.site_names)
        force_site_ids = np.asarray(
            [
                self.controller.wrench_sim.site_ids[name]
                for name in self.force_site_names
            ],
            dtype=np.int32,
        )
        self.force_rng = np.random.default_rng()
        self.force_max = 3.0
        self.force_vis_scale = 0.1
        self.perturb_site_forces = np.zeros(
            (len(self.force_site_names), 3), dtype=np.float32
        )
        self.force_active = False
        self.force_phase_end_time = 0.0
        self.force_pause_duration = 1.0
        self.site_force_applier = None
        self.base_pos_cmd = self.pos_cmd.copy()
        self.base_ori_cmd = self.ori_cmd.copy()

        self.plotter: Optional[CompliancePlotter] = None
        if self.plot:
            run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_dir = os.path.join(
                self.repo_root, "results", f"{self.robot}_compliance_{run_stamp}"
            )
            os.makedirs(results_dir, exist_ok=True)
            print(f"[plot] writing PNGs to: {results_dir}")
            self.plotter = CompliancePlotter(
                site_names=self.force_site_names,
                enabled=True,
                output_dir=results_dir,
            )
            if not self.plotter.enabled and self.plotter.error_message:
                print(f"[plot] disabled: {self.plotter.error_message}")

        self.target_motor_pos = np.asarray(
            self.default_state.motor_pos, dtype=np.float32
        )
        self.force_site_ids = force_site_ids
        self._closed = False

    def _update_force_perturbation(self) -> None:
        pos_offsets, rot_offsets, force_enabled = self.teleop.snapshot()
        self.pos_cmd = self.base_pos_cmd + pos_offsets
        self.ori_cmd = self.base_ori_cmd + rot_offsets

        now = time.monotonic()
        if force_enabled:
            if now >= self.force_phase_end_time:
                if self.force_active:
                    self.force_active = False
                    self.perturb_site_forces[:] = 0.0
                    self.force_phase_end_time = now + self.force_pause_duration
                else:
                    self.force_active = True
                    num_sites = len(self.force_site_names)
                    if num_sites <= 0:
                        self.perturb_site_forces[:] = 0.0
                    else:
                        vec = self.force_rng.normal(size=(num_sites, 3)).astype(
                            np.float32
                        )
                        norms = np.linalg.norm(vec, axis=1, keepdims=True)
                        norms = np.maximum(norms, 1e-6)
                        direction = vec / norms
                        magnitudes = self.force_rng.uniform(
                            0.0, float(self.force_max), size=(num_sites, 1)
                        ).astype(np.float32)
                        self.perturb_site_forces[:] = direction * magnitudes
                    force_norms = np.linalg.norm(self.perturb_site_forces, axis=1)
                    non_zero = np.flatnonzero(force_norms > 1e-6)
                    if non_zero.size > 0:
                        desc = ", ".join(
                            f"{self.force_site_names[i]}:{force_norms[i]:.2f}N"
                            for i in non_zero
                        )
                        print(f"[force] sampled -> {desc}")
                    self.force_phase_end_time = now + float(
                        self.force_rng.uniform(1.0, 5.0)
                    )
            elif not self.force_active:
                self.perturb_site_forces[:] = 0.0
        else:
            self.force_active = False
            self.force_phase_end_time = now
            self.perturb_site_forces[:] = 0.0

    def step(self, obs: Any, sim: Any) -> tuple[dict[str, float], np.ndarray]:
        self._update_force_perturbation()

        if self.sim_backend == "mujoco":
            if self.site_force_applier is not None:
                self.site_force_applier(self.data, self.perturb_site_forces)
            if hasattr(sim, "set_debug_site_targets"):
                sim.set_debug_site_targets(
                    {
                        site_name: np.concatenate(
                            [self.pos_cmd[idx], self.ori_cmd[idx]], axis=0
                        ).astype(np.float32)
                        for idx, site_name in enumerate(self.force_site_names)
                    }
                )
            if hasattr(sim, "set_debug_site_forces"):
                sim.set_debug_site_forces(
                    {
                        site_name: self.perturb_site_forces[idx]
                        for idx, site_name in enumerate(self.force_site_names)
                    },
                    vis_scale=self.force_vis_scale,
                )

        self.command_matrix[:, COMMAND_LAYOUT.position] = self.pos_cmd
        self.command_matrix[:, COMMAND_LAYOUT.orientation] = self.ori_cmd

        motor_tor_obs = np.asarray(obs["motor_tor"], dtype=np.float32)
        qpos_obs = np.asarray(obs["qpos"], dtype=np.float32)
        wrenches, state_ref = self.controller.step(
            command_matrix=self.command_matrix,
            motor_torques=motor_tor_obs,
            qpos=qpos_obs,
        )

        if self.plotter is not None:
            self.plotter.update_from_wrench_sim(
                time_s=float(obs.get("time", 0.0)),
                command_pose=np.concatenate(
                    [self.pos_cmd, self.ori_cmd], axis=1
                ).astype(np.float32),
                wrenches=wrenches,
                applied_site_forces=(
                    self.perturb_site_forces if self.sim_backend == "mujoco" else None
                ),
                wrench_sim=self.controller.wrench_sim,
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
            self.target_motor_pos = np.asarray(state_ref.motor_pos)

        return {}, np.asarray(self.target_motor_pos, dtype=np.float32).copy()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self.plotter is not None:
            self.plotter.close()
        self.key_listener.stop()
