"""Compliance control policy leveraging online wrench estimation."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

import gin
import joblib
import numpy as np
import numpy.typing as npt

from minimalist_compliance_control.compliance_ref import COMMAND_LAYOUT, ComplianceState
from minimalist_compliance_control.controller import (
    ComplianceController,
    ControllerConfig,
    RefConfig,
)
from minimalist_compliance_control.utils import (
    KeyboardListener,
    KeyboardTeleop,
    ensure_matrix,
    get_action_traj,
    get_damping_matrix,
    interpolate_action,
)
from minimalist_compliance_control.visualization import CompliancePlotter
from minimalist_compliance_control.wrench_estimation import WrenchEstimateConfig


@gin.configurable
@dataclass
class ComplianceConfig:
    kp_pos: Any = 100.0
    kp_rot: Any = 10.0
    kp_pos_normal: Optional[float] = None
    kp_pos_tangent: Optional[float] = None
    kp_rot_normal: Optional[float] = None
    kp_rot_tangent: Optional[float] = None
    fixed_contact_force: Optional[float] = None
    head_name: str = "head"
    initial_pose: Sequence[Sequence[float]] = ()
    ref_motor_pos: Sequence[float] = ()
    use_compliance: bool = True
    perturb_force_max: float = 3.0


class CompliancePolicy:
    """Old-style base compliance policy built on ComplianceController."""

    def __init__(
        self,
        name: str,
        robot: str,
        init_motor_pos: npt.ArrayLike,
        config_name: Optional[str] = None,
        controller_xml_path: Optional[str] = None,
        controller_dt: Optional[float] = None,
        show_help: bool = True,
        controller: Optional[ComplianceController] = None,
        start_keyboard_listener: bool = True,
        enable_plotter: bool = True,
        enable_force_perturbation: bool = True,
    ) -> None:
        self.name = name
        self.robot = robot

        self.repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

        if controller is None:
            selected_config_name: str
            if config_name is not None:
                selected_config_name = str(config_name)
            elif self.robot == "leap":
                selected_config_name = "leap.gin"
            elif self.robot == "g1":
                selected_config_name = "g1.gin"
            elif self.robot == "arx":
                selected_config_name = "arx.gin"
            else:
                selected_config_name = "toddlerbot.gin"

            gin.clear_config()
            gin.parse_config_file(
                os.path.join(self.repo_root, "config", selected_config_name)
            )
            gin.bind_parameter("WrenchSimConfig.view", False)
            gin.bind_parameter("WrenchSimConfig.render", False)
            if controller_xml_path is not None and str(controller_xml_path).strip():
                xml_path = str(controller_xml_path).strip()
                if not os.path.isabs(xml_path):
                    xml_path = os.path.abspath(os.path.join(self.repo_root, xml_path))
                gin.bind_parameter("WrenchSimConfig.xml_path", xml_path)
                gin.bind_parameter("ControllerConfig.xml_path", xml_path)
                fixed_model_xml = os.path.join(
                    os.path.dirname(xml_path), "left_hand_fixed.xml"
                )
                if os.path.exists(fixed_model_xml):
                    gin.bind_parameter("RefConfig.fixed_model_xml_path", fixed_model_xml)
            if controller_dt is not None:
                gin.bind_parameter("RefConfig.dt", float(controller_dt))

            controller_cfg = ControllerConfig()
            self.controller = ComplianceController(
                config=controller_cfg,
                estimate_config=WrenchEstimateConfig(),
                ref_config=RefConfig(),
            )
        else:
            self.controller = controller

        compliance_cfg = ComplianceConfig()
        self.compliance_cfg = compliance_cfg
        if self.controller.compliance_ref is None:
            raise ValueError("Controller compliance_ref must be configured.")

        self.control_dt = float(self.controller.ref_config.dt)
        self.wrench_site_names = tuple(self.controller.config.site_names)
        self.wrench_site_ids = self.controller.site_ids
        self.num_sites: int = len(self.wrench_site_names)

        self.default_state = self.controller.compliance_ref.get_default_state()
        self.default_motor_pos = np.asarray(
            self.default_state.motor_pos, dtype=np.float32
        )
        self.default_qpos = np.asarray(self.default_state.qpos, dtype=np.float32)

        init_motor_pos_arr = np.asarray(init_motor_pos, dtype=np.float32).reshape(-1)
        if init_motor_pos_arr.shape[0] == self.default_motor_pos.shape[0]:
            self.init_motor_pos = init_motor_pos_arr.copy()
        else:
            self.init_motor_pos = self.default_motor_pos.copy()
        self.ref_motor_pos = self.default_motor_pos.copy()

        self.pose_command = np.asarray(
            self.default_state.x_ref, dtype=np.float32
        ).copy()
        init_pose_arr = np.asarray(compliance_cfg.initial_pose, dtype=np.float32)
        self._has_initial_pose_override = bool(init_pose_arr.size > 0)
        if init_pose_arr.size > 0:
            if init_pose_arr.shape != (self.num_sites, 6):
                raise ValueError(
                    "ComplianceConfig.initial_pose must be "
                    f"({self.num_sites}, 6), got {init_pose_arr.shape}."
                )
            self.pose_command = init_pose_arr.copy()
        self.base_pose_command = self.pose_command.copy()

        self.teleop = KeyboardTeleop(
            num_sites=self.num_sites,
            site_names=self.wrench_site_names,
            show_help=show_help,
        )
        self.key_listener: Optional[KeyboardListener] = None
        if bool(start_keyboard_listener):
            self.key_listener = KeyboardListener(self.teleop)
            self.key_listener.start()

        self.mass = float(self.controller.ref_config.mass)
        self.inertia_diag = np.asarray(
            self.controller.ref_config.inertia_diag, dtype=np.float32
        )

        self.pos_stiffness = np.zeros((self.num_sites, 9), dtype=np.float32)
        self.rot_stiffness = np.zeros((self.num_sites, 9), dtype=np.float32)
        self.pos_damping = np.zeros((self.num_sites, 9), dtype=np.float32)
        self.rot_damping = np.zeros((self.num_sites, 9), dtype=np.float32)

        self.wrench_command = np.zeros((self.num_sites, 6), dtype=np.float32)
        self.command_matrix = np.zeros(
            (self.num_sites, COMMAND_LAYOUT.width), dtype=np.float32
        )
        self.enable_force_perturbation = bool(enable_force_perturbation)
        self.force_site_names = tuple(self.wrench_site_names)
        self.force_site_ids = np.asarray(
            [self.wrench_site_ids[name] for name in self.force_site_names],
            dtype=np.int32,
        )
        self.force_rng = np.random.default_rng()
        self.force_max = (
            float(compliance_cfg.perturb_force_max)
            if self.enable_force_perturbation
            else 0.0
        )
        self.force_vis_scale = 0.1
        self.perturb_site_forces: Optional[npt.NDArray[np.float32]] = None
        self.force_active = False
        self.force_phase_end_time = 0.0
        self.force_pause_duration = 1.0
        self.site_force_applier = None

        self.use_compliance = bool(compliance_cfg.use_compliance)
        self._alignment_applied = False
        self.use_balance = False
        self.balance_yaw = False

        self.wrenches_by_site: Dict[str, npt.NDArray[np.float32]] = {}

        self.compliance_time_log: list[float] = []
        self.pose_command_log: list[np.ndarray] = []
        self.x_ref_log: list[np.ndarray] = []
        self.x_ik_log: list[np.ndarray] = []
        self.x_obs_log: list[np.ndarray] = []
        self.obs_motor_pos_log: list[np.ndarray] = []

        self.plotter = (
            CompliancePlotter(site_names=self.wrench_site_names, enabled=True)
            if bool(enable_plotter)
            else None
        )

        self.is_prepared = False

        self.set_stiffness(compliance_cfg.kp_pos, compliance_cfg.kp_rot)

    def set_stiffness(
        self,
        pos_stiffness: Sequence[float] | npt.NDArray[np.float32] | float,
        rot_stiffness: Sequence[float] | npt.NDArray[np.float32] | float,
        pos_damp_ratio: float = 1.0,
        rot_damp_ratio: float = 1.0,
        pos_damping: Optional[Sequence[float] | npt.NDArray[np.float32]] = None,
        rot_damping: Optional[Sequence[float] | npt.NDArray[np.float32]] = None,
    ) -> None:
        kp_pos = ensure_matrix(pos_stiffness)
        kp_rot = ensure_matrix(rot_stiffness)

        if pos_damping is not None:
            kd_pos = ensure_matrix(pos_damping)
        else:
            kd_pos = get_damping_matrix(kp_pos, ensure_matrix(self.mass)) * float(
                pos_damp_ratio
            )

        if rot_damping is not None:
            kd_rot = ensure_matrix(rot_damping)
        else:
            kd_rot = get_damping_matrix(
                kp_rot, ensure_matrix(self.inertia_diag)
            ) * float(rot_damp_ratio)

        self.pos_stiffness = np.tile(kp_pos.reshape(1, 9), (self.num_sites, 1)).astype(
            np.float32
        )
        self.rot_stiffness = np.tile(kp_rot.reshape(1, 9), (self.num_sites, 1)).astype(
            np.float32
        )
        self.pos_damping = np.tile(kd_pos.reshape(1, 9), (self.num_sites, 1)).astype(
            np.float32
        )
        self.rot_damping = np.tile(kd_rot.reshape(1, 9), (self.num_sites, 1)).astype(
            np.float32
        )

    def update_pose_command_from_obs(
        self, obs: Any, x_obs: npt.NDArray[np.float32]
    ) -> None:
        del obs
        if self.pose_command is None:
            self.pose_command = x_obs.copy()

    def build_command_matrix(
        self,
        pose_command: npt.NDArray[np.float32],
        wrench_command: Optional[npt.NDArray[np.float32]] = None,
        measured_wrenches: Optional[Dict[str, npt.NDArray[np.float32]]] = None,
    ) -> npt.NDArray[np.float32]:
        cmd = np.zeros((self.num_sites, COMMAND_LAYOUT.width), dtype=np.float32)
        cmd[:, COMMAND_LAYOUT.position] = pose_command[:, :3]
        cmd[:, COMMAND_LAYOUT.orientation] = pose_command[:, 3:6]

        cmd[:, COMMAND_LAYOUT.kp_pos] = self.pos_stiffness
        cmd[:, COMMAND_LAYOUT.kp_rot] = self.rot_stiffness
        cmd[:, COMMAND_LAYOUT.kd_pos] = self.pos_damping
        cmd[:, COMMAND_LAYOUT.kd_rot] = self.rot_damping

        wrench_cmd = self.wrench_command if wrench_command is None else wrench_command
        cmd[:, COMMAND_LAYOUT.force] = wrench_cmd[:, :3]
        cmd[:, COMMAND_LAYOUT.torque] = wrench_cmd[:, 3:6]
        if measured_wrenches is not None:
            for idx, site_name in enumerate(self.wrench_site_names):
                wrench = measured_wrenches.get(site_name)
                if wrench is None:
                    continue
                wrench_arr = np.asarray(wrench, dtype=np.float32).reshape(-1)
                if wrench_arr.shape[0] < 6:
                    continue
                cmd[idx, COMMAND_LAYOUT.measured_force] = wrench_arr[:3]
                cmd[idx, COMMAND_LAYOUT.measured_torque] = wrench_arr[3:6]
        return cmd

    def compute_direct_action(self) -> npt.NDArray[np.float32]:
        return np.asarray(self.ref_motor_pos, dtype=np.float32).copy()

    def apply_controller_step(
        self,
        *,
        command_matrix: npt.NDArray[np.float32],
        target_motor_pos: npt.ArrayLike,
        measured_wrenches: Optional[Dict[str, npt.NDArray[np.float32]]] = None,
        site_names: Optional[Sequence[str]] = None,
        motor_torques: Optional[npt.ArrayLike] = None,
        qpos: Optional[npt.ArrayLike] = None,
        controlled_actuators_only: bool = False,
    ) -> tuple[npt.NDArray[np.float32], Optional[ComplianceState]]:
        if motor_torques is None:
            motor_torque_arr = np.asarray(
                self.controller.wrench_sim.data.actuator_force, dtype=np.float32
            )
        else:
            motor_torque_arr = np.asarray(motor_torques, dtype=np.float32)

        if qpos is None:
            qpos_arr = np.asarray(self.controller.wrench_sim.data.qpos, dtype=np.float32)
        else:
            qpos_arr = np.asarray(qpos, dtype=np.float32)

        wrenches_out, state_ref = self.controller.step(
            command_matrix=np.asarray(command_matrix, dtype=np.float32),
            motor_torques=motor_torque_arr,
            qpos=qpos_arr,
        )

        next_target = np.asarray(target_motor_pos, dtype=np.float32).copy()
        if state_ref is not None:
            state_ref_motor = np.asarray(state_ref.motor_pos, dtype=np.float32)
            if (
                controlled_actuators_only
                and self.controller.compliance_ref is not None
                and self.controller.compliance_ref.actuator_indices is not None
            ):
                actuator_indices = np.asarray(
                    self.controller.compliance_ref.actuator_indices, dtype=np.int32
                )
                valid_idx = actuator_indices[
                    np.logical_and(
                        actuator_indices >= 0,
                        actuator_indices < min(next_target.shape[0], state_ref_motor.shape[0]),
                    )
                ]
                next_target[valid_idx] = state_ref_motor[valid_idx]
            elif state_ref_motor.shape == next_target.shape:
                next_target = state_ref_motor.copy()
            else:
                copied = min(next_target.shape[0], state_ref_motor.shape[0])
                next_target[:copied] = state_ref_motor[:copied]

        if measured_wrenches is not None:
            sites = self.wrench_site_names if site_names is None else tuple(site_names)
            for site in sites:
                wrench = wrenches_out.get(site)
                if wrench is not None:
                    measured_wrenches[str(site)] = np.asarray(wrench, dtype=np.float32)

        return next_target, state_ref

    def _compute_reference_and_log(self, obs: Any) -> ComplianceState:
        qpos = np.asarray(obs.qpos, dtype=np.float32)
        motor_tor = np.asarray(obs.motor_tor, dtype=np.float32)

        command_matrix = self.build_command_matrix(
            np.asarray(self.pose_command, dtype=np.float32)
        )
        wrenches_by_site, state_ref = self.controller.step(
            command_matrix=command_matrix,
            motor_torques=motor_tor,
            qpos=qpos,
        )
        self.wrenches_by_site = {
            site: np.asarray(w, dtype=np.float32).copy()
            for site, w in wrenches_by_site.items()
        }
        if state_ref is None:
            raise RuntimeError("Controller returned no compliance reference state.")

        x_obs = self.controller.get_x_obs()
        self.compliance_time_log.append(float(obs.time))
        self.pose_command_log.append(
            np.asarray(self.pose_command, dtype=np.float32).copy()
        )
        self.x_obs_log.append(x_obs.copy())
        self.obs_motor_pos_log.append(
            np.asarray(obs.motor_pos, dtype=np.float32).copy()
        )

        self.x_ref_log.append(np.asarray(state_ref.x_ref, dtype=np.float32).copy())
        self.x_ik_log.append(np.asarray(state_ref.x_ik, dtype=np.float32).copy())

        if self.plotter is not None:
            self.plotter.update_from_wrench_sim(
                time_s=float(obs.time),
                command_pose=np.asarray(self.pose_command, dtype=np.float32),
                x_ref=np.asarray(state_ref.x_ref, dtype=np.float32),
                x_ik=np.asarray(state_ref.x_ik, dtype=np.float32),
                wrenches=self.wrenches_by_site,
                applied_site_forces=self.perturb_site_forces,
                x_obs=x_obs,
            )

        return state_ref

    def compute_compliant_action(self, obs: Any) -> npt.NDArray[np.float32]:
        state_ref = self._compute_reference_and_log(obs)
        return np.asarray(state_ref.motor_pos, dtype=np.float32)

    def _align_command_to_observation(self, obs: Any) -> None:
        if self._alignment_applied:
            return
        if self._has_initial_pose_override:
            self._alignment_applied = True
            return

        qpos = np.asarray(obs.qpos, dtype=np.float32)
        self.controller.sync_qpos(qpos)
        x_obs = self.controller.get_x_obs()
        self.pose_command = x_obs.copy()
        self.base_pose_command = x_obs.copy()

        motor_pos = np.asarray(obs.motor_pos, dtype=np.float32).reshape(-1)
        if motor_pos.shape[0] != self.default_motor_pos.shape[0]:
            motor_pos = self.default_motor_pos.copy()
        zeros = np.zeros_like(x_obs, dtype=np.float32)
        self.controller._last_state = ComplianceState(
            x_ref=x_obs.copy(),
            x_ik=x_obs.copy(),
            v_ref=zeros.copy(),
            a_ref=zeros.copy(),
            motor_pos=motor_pos.copy(),
            qpos=qpos.copy(),
        )
        self._alignment_applied = True

    def _update_force_perturbation(self) -> None:
        if not bool(self.enable_force_perturbation):
            if self.perturb_site_forces is not None:
                self.perturb_site_forces[:] = 0.0
            self.force_active = False
            return

        pos_offsets, rot_offsets, force_enabled = self.teleop.snapshot()
        self.pose_command[:, :3] = self.base_pose_command[:, :3] + pos_offsets
        self.pose_command[:, 3:6] = self.base_pose_command[:, 3:6] + rot_offsets

        if self.perturb_site_forces is None:
            self.force_active = False
            self.force_phase_end_time = time.monotonic()
            return

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

    def step(self, obs: Any, sim: Any) -> npt.NDArray[np.float32]:
        has_mujoco_state = str(getattr(sim, "name", "")).lower() == "mujoco"
        if has_mujoco_state and bool(self.enable_force_perturbation):
            if self.perturb_site_forces is None:
                self.perturb_site_forces = np.zeros(
                    (len(self.force_site_names), 3), dtype=np.float32
                )
        else:
            self.perturb_site_forces = None

        if not self.is_prepared:
            self.is_prepared = True
            self.prep_duration = 2.0 if has_mujoco_state else 7.0
            self.prep_time, self.prep_action = get_action_traj(
                0.0,
                self.init_motor_pos,
                self.ref_motor_pos,
                self.prep_duration,
                self.control_dt,
                end_time=0.0 if has_mujoco_state else 5.0,
            )
        if float(obs.time) < float(self.prep_duration):
            return np.asarray(
                interpolate_action(float(obs.time), self.prep_time, self.prep_action),
                dtype=np.float32,
            )

        if not self._alignment_applied:
            self._align_command_to_observation(obs)

        self._update_force_perturbation()
        if (
            has_mujoco_state
            and bool(self.enable_force_perturbation)
            and self.perturb_site_forces is not None
        ):
            if self.site_force_applier is not None:
                self.site_force_applier(sim.data, self.perturb_site_forces)
            sim.set_debug_site_targets(
                {
                    site_name: np.concatenate(
                        [self.pose_command[idx, :3], self.pose_command[idx, 3:6]],
                        axis=0,
                    ).astype(np.float32)
                    for idx, site_name in enumerate(self.force_site_names)
                }
            )
            sim.set_debug_site_forces(
                {
                    site_name: self.perturb_site_forces[idx]
                    for idx, site_name in enumerate(self.force_site_names)
                },
                vis_scale=self.force_vis_scale,
            )

        state_ref = self._compute_reference_and_log(obs)
        if self.use_compliance:
            action = np.asarray(state_ref.motor_pos, dtype=np.float32)
        else:
            action = self.compute_direct_action()

        return np.asarray(action, dtype=np.float32)

    def save_compliance_ref_log(self, exp_folder_path: str) -> None:
        if not exp_folder_path:
            return
        os.makedirs(exp_folder_path, exist_ok=True)
        payload = {
            "time": np.asarray(self.compliance_time_log, dtype=np.float32),
            "pose_command": np.asarray(self.pose_command_log, dtype=np.float32),
            "x_ref": np.asarray(self.x_ref_log, dtype=np.float32),
            "x_ik": np.asarray(self.x_ik_log, dtype=np.float32),
            "x_obs": np.asarray(self.x_obs_log, dtype=np.float32),
            "obs_motor_pos": np.asarray(self.obs_motor_pos_log, dtype=np.float32),
        }
        joblib.dump(payload, os.path.join(exp_folder_path, "compliance_log.lz4"))

    def close(self, exp_folder_path: str = "") -> None:
        if self.plotter is not None:
            self.plotter.close(exp_folder_path=exp_folder_path)

        self.save_compliance_ref_log(exp_folder_path)
        if self.key_listener is not None:
            self.key_listener.stop()
        self.controller.close()
