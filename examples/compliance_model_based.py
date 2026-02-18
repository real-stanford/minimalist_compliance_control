from __future__ import annotations

import argparse
import os
from typing import Any, Sequence

import gin
import mujoco
import numpy as np

from hybrid_servo.algorithm.ochs import solve_ochs
from hybrid_servo.tasks.bimanual_ochs import compute_ochs_inputs
from hybrid_servo.tasks.bimanual_ochs import (
    generate_constraint_jacobian as bimanual_generate_constraint_jacobian,
)
from hybrid_servo.utils import (
    PREPARE_POS,
    _build_command_matrix,
    _build_controller,
    _find_repo_root,
    _get_ground_truth_wrenches,
    _load_motor_params,
    _tb__assign_stiffness,
    _tb__build_command_matrix,
    _tb__build_contact_state,
    _tb__build_prep_traj,
    _tb__distribute_rigid_body_motion,
    _tb__ensure_default_hand_rotvec,
    _tb__initialize_runtime_from_default_state,
    _tb__integrate_pose_command,
    _tb__interpolate_action,
    _tb__load_kneel_trajectory,
    _tb__load_motor_group_indices,
    _tb__load_motor_params,
    _tb__maybe_print_ochs_world_velocity,
    _tb__poll_keyboard_command,
    _tb__reset_approach_interp,
    _tb__reset_pose_command_to_current_sites,
    _tb__resolve_mocap_target_ids,
    _tb__run_approach_phase,
    _tb__run_compliance_step,
    _tb__sync_compliance_state_to_current_pose,
    _tb__update_delta_goal,
    _tb__update_expected_ball_pos,
    _tb__update_goal_from_keyboard_and_time,
    _tb__update_mocap_targets_from_state_ref,
    _tb_PolicyConfig,
    create_leap_rotate_policy,
    leap_rotate_policy_capture_object_init,
    leap_rotate_policy_fix_object,
    leap_rotate_policy_forward_object_to_init,
    leap_rotate_policy_step,
)
from minimalist_compliance_control.controller import (
    ComplianceController,
    ComplianceRefConfig,
    ControllerConfig,
)
from minimalist_compliance_control.utils import KeyboardControlReceiver
from minimalist_compliance_control.wrench_estimation import WrenchEstimateConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Model-based compliance per-tick policy"
    )
    parser.add_argument(
        "--scene-xml",
        type=str,
        default="",
        help="LEAP only: scene XML path (default descriptions/leap_hand/scene_object_fixed.xml).",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=120.0,
        help="LEAP only: stop after this many simulation seconds; <=0 disables.",
    )
    parser.add_argument(
        "--control-dt",
        type=float,
        default=0.02,
        help="LEAP only: control dt.",
    )
    parser.add_argument(
        "--prep-duration",
        type=float,
        default=7.0,
        help="LEAP only: prep phase duration.",
    )
    parser.add_argument(
        "--status-interval",
        type=float,
        default=1.0,
        help="LEAP only: print status interval (s).",
    )
    return parser


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


class LeapModelBasedPolicy:
    def __init__(self, args: argparse.Namespace, *, vis: bool) -> None:
        self.args = args
        self.vis = bool(vis)
        self.repo_root = _find_repo_root(os.path.abspath(os.path.dirname(__file__)))

        if args.scene_xml:
            scene_xml_path = os.path.abspath(args.scene_xml)
        else:
            scene_xml_path = os.path.join(
                self.repo_root, "descriptions", "leap_hand", "scene_object_fixed.xml"
            )
        self.scene_xml_path = scene_xml_path

        self.controller = _build_controller(scene_xml_path, float(args.control_dt))
        if self.controller.compliance_ref is None:
            raise RuntimeError("Controller compliance_ref is not initialized.")
        self.model = self.controller.wrench_sim.model
        self.data = self.controller.wrench_sim.data
        self.site_names = tuple(self.controller.config.site_names)

        self.data.qpos[:] = self.controller.compliance_ref.default_qpos.copy()
        mujoco.mj_forward(self.model, self.data)

        self.policy = create_leap_rotate_policy(
            wrench_sim=self.controller.wrench_sim,
            wrench_site_names=self.site_names,
            control_dt=float(self.controller.ref_config.dt),
            prep_duration=max(float(args.prep_duration), 0.0),
            auto_switch_target_enabled=True,
        )
        leap_rotate_policy_forward_object_to_init(self.policy, sim_name="sim")

        self.target_motor_pos = np.asarray(
            self.controller.compliance_ref.default_motor_pos, dtype=np.float32
        )
        self.measured_wrenches: dict[str, np.ndarray] = {}
        self.control_dt = float(self.controller.ref_config.dt)

        trnid = np.asarray(self.model.actuator_trnid[:, 0], dtype=np.int32)
        self.qpos_adr = self.model.jnt_qposadr[trnid]
        self.qvel_adr = self.model.jnt_dofadr[trnid]

        self.prep_start_motor_pos = np.asarray(
            self.data.qpos[self.qpos_adr], dtype=np.float32
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
        ) = _load_motor_params(self.repo_root, robot_desc_dir, self.model)

        def _extra_substep(_data: mujoco.MjData) -> None:
            sim_time_local = float(_data.time)
            if sim_time_local < self.prep_duration or self.policy.phase == "close":
                leap_rotate_policy_capture_object_init(self.policy)
                leap_rotate_policy_fix_object(
                    self.policy, self.controller.wrench_sim, sim_name="sim"
                )
                mujoco.mj_forward(self.model, self.data)

        substep_control = _make_clamped_torque_substep_control(
            qpos_adr=self.qpos_adr,
            qvel_adr=self.qvel_adr,
            target_motor_pos_getter=lambda: self.target_motor_pos,
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
        self.substep_control = substep_control

        self.status_interval = max(float(args.status_interval), 1e-3)
        self.next_status_time = 0.0
        self.done = False

    def step(self, obs: Any, sim: Any) -> tuple[dict[str, float], np.ndarray]:
        del sim
        sim_time = float(obs.get("time", self.data.time))
        if float(self.args.duration) > 0.0 and sim_time >= float(self.args.duration):
            print("[leaphand] Reached duration limit, exiting.")
            self.done = True
            return {}, self.target_motor_pos.copy()

        self.measured_wrenches = _get_ground_truth_wrenches(
            self.model, self.data, self.site_names
        )
        if sim_time < self.prep_duration:
            leap_rotate_policy_step(
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
                self.target_motor_pos = (
                    self.prep_start_motor_pos
                    + (self.prep_target_motor_pos - self.prep_start_motor_pos) * alpha
                ).astype(np.float32)
            else:
                self.target_motor_pos = self.prep_target_motor_pos.copy()
        else:
            policy_out = leap_rotate_policy_step(
                self.policy,
                time_curr=sim_time,
                wrenches_by_site=self.measured_wrenches,
                sim_name="sim",
                is_real_world=False,
            )
            command_matrix = _build_command_matrix(
                site_names=self.site_names,
                policy_out=policy_out,
                measured_wrenches=self.measured_wrenches,
            )
            qpos_obs = np.asarray(obs.get("qpos", self.data.qpos), dtype=np.float32)
            motor_tor_obs = np.asarray(obs["motor_tor"], dtype=np.float32)
            _, state_ref = self.controller.step(
                command_matrix=command_matrix,
                motor_torques=motor_tor_obs,
                qpos=qpos_obs,
            )
            if state_ref is not None:
                self.target_motor_pos = np.asarray(
                    state_ref.motor_pos, dtype=np.float32
                )

        if sim_time >= self.next_status_time:
            print(
                f"[leaphand] t={sim_time:.2f}s phase={self.policy.phase} mode={self.policy.control_mode}"
            )
            self.next_status_time = sim_time + self.status_interval

        return {}, self.target_motor_pos.copy()

    def close(self) -> None:
        if getattr(self.policy, "control_receiver", None) is not None:
            self.policy.control_receiver.close()


class ToddlerBotModelBasedPolicy:
    def __init__(self, *, vis: bool) -> None:
        self.cfg = _tb_PolicyConfig()
        self.vis = bool(vis)

        repo_root = _find_repo_root(os.path.abspath(os.path.dirname(__file__)))
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
            ref_config=ComplianceRefConfig(),
        )
        if self.controller.compliance_ref is None:
            raise RuntimeError("Compliance reference failed to initialize.")

        self.model = self.controller.wrench_sim.model
        self.data = self.controller.wrench_sim.data
        self.model.opt.timestep = float(self.cfg.sim_dt)
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

        self.site_names = tuple(self.controller.config.site_names)
        if self.site_names != ("left_hand_center", "right_hand_center"):
            raise ValueError(
                "This example expects site_names == ('left_hand_center', 'right_hand_center')."
            )
        self.left_site_id = self.controller.wrench_sim.site_ids[self.site_names[0]]
        self.right_site_id = self.controller.wrench_sim.site_ids[self.site_names[1]]

        trnid = np.asarray(self.model.actuator_trnid[:, 0], dtype=np.int32)
        if not np.all(trnid >= 0):
            raise ValueError("Actuator without joint mapping is not supported.")
        self.qpos_adr = np.asarray(self.model.jnt_qposadr[trnid], dtype=np.int32)
        self.qvel_adr = np.asarray(self.model.jnt_dofadr[trnid], dtype=np.int32)

        default_state = self.controller.compliance_ref.get_default_state()
        kneel_action_arr, kneel_qpos, kneel_qpos_source_dim = (
            _tb__load_kneel_trajectory(
                example_dir=repo_root,
                cfg=self.cfg,
                default_motor_pos=np.asarray(default_state.motor_pos, dtype=np.float64),
                default_qpos=np.asarray(default_state.qpos, dtype=np.float64),
                motor_dim=self.model.nu,
                qpos_dim=self.model.nq,
            )
        )
        self.runtime = _tb__initialize_runtime_from_default_state(
            default_state=default_state,
            cfg=self.cfg,
            kneel_action_arr=kneel_action_arr,
            kneel_qpos=kneel_qpos,
            kneel_qpos_source_dim=kneel_qpos_source_dim,
        )
        default_motor_pos = np.asarray(default_state.motor_pos, dtype=np.float64).copy()
        prep_init_motor_pos = np.asarray(
            self.data.qpos[self.qpos_adr], dtype=np.float64
        ).copy()
        self.target_motor_pos = prep_init_motor_pos.copy()
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
        self.control_receiver = KeyboardControlReceiver()
        if self.control_receiver.enabled:
            print("[model_based] Commands: c=reverse, l=left, r=right, b=both")

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
        ) = _tb__load_motor_params(self.model)
        group_indices = _tb__load_motor_group_indices(self.model)
        neck_indices = group_indices.get("neck", np.zeros(0, dtype=np.int32))
        leg_indices = group_indices.get("leg", np.zeros(0, dtype=np.int32))
        self.hold_indices = np.unique(
            np.concatenate([neck_indices, leg_indices])
        ).astype(np.int32)
        self.kneel_hold_motor_pos = self.runtime.kneel_action_arr[-1].copy()

        self.control_dt = float(self.controller.ref_config.dt)
        self.prep_time, self.prep_action = _tb__build_prep_traj(
            prep_init_motor_pos,
            default_motor_pos,
            self.cfg.prep_duration,
            self.control_dt,
            self.cfg.prep_hold_duration,
        )
        self.left_mocap_id, self.right_mocap_id = _tb__resolve_mocap_target_ids(
            self.model
        )

        substep_control = _make_clamped_torque_substep_control(
            qpos_adr=self.qpos_adr,
            qvel_adr=self.qvel_adr,
            target_motor_pos_getter=lambda: self.target_motor_pos,
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
                self.model, mujoco.mjtObj.mjOBJ_BODY, "rolling_ball"
            )
            if ball_body_id < 0:
                raise KeyError("Body 'rolling_ball' not found.")
            self.default_ball_pos = np.asarray(
                self.data.body_xpos[ball_body_id], dtype=np.float64
            ).copy()
        except Exception:
            self.default_ball_pos = np.array([0.24, -0.015, 0.08], dtype=np.float64)
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

        ball_pos_obs = None
        if obs is not None and hasattr(obs, "get"):
            ball_pos_obs = obs.get("ball_pos", None)
        if ball_pos_obs is not None:
            ball_pos_arr = np.asarray(ball_pos_obs, dtype=np.float64).reshape(-1)
            if ball_pos_arr.shape[0] >= 3:
                return ball_pos_arr[:3].copy()

        if self.ball_pos_estimate_log:
            return self.ball_pos_estimate_log[-1].copy()
        return self.default_ball_pos.copy()

    def step(self, obs: Any, sim: Any) -> tuple[dict[str, float], np.ndarray]:
        t = float(obs.get("time", self.data.time))
        ball_pos = self.update_ball_pose_estimate(obs, sim, "real" in sim.name)
        self.ball_pos_estimate_log.append(
            np.asarray(ball_pos, dtype=np.float64).reshape(3).copy()
        )
        state = _tb__build_contact_state(
            self.model,
            self.data,
            self.left_site_id,
            self.right_site_id,
            ball_pos=ball_pos,
        )
        if self.runtime.phase == "prep":
            if t < float(self.cfg.prep_duration):
                self.target_motor_pos = _tb__interpolate_action(
                    t, self.prep_time, self.prep_action
                )
            else:
                self.runtime.phase = "kneel"
                self.runtime.phase_start_time = t
                _tb__ensure_default_hand_rotvec(
                    self.runtime, self.data, self.left_site_id, self.right_site_id
                )
                self.target_motor_pos = self.runtime.kneel_action_arr[0].copy()
                print("[model_based] Phase transition: prep -> kneel")
        elif self.runtime.phase == "kneel":
            _tb__ensure_default_hand_rotvec(
                self.runtime, self.data, self.left_site_id, self.right_site_id
            )
            elapsed = max(float(t - self.cfg.prep_duration), 0.0)
            idx = int(
                np.clip(
                    elapsed / max(self.control_dt, 1e-6),
                    0,
                    len(self.runtime.kneel_action_arr) - 1,
                )
            )
            self.target_motor_pos = self.runtime.kneel_action_arr[idx].copy()
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
                self.target_motor_pos = self.runtime.kneel_action_arr[-1].copy()
                _tb__reset_pose_command_to_current_sites(
                    self.runtime, self.data, self.left_site_id, self.right_site_id
                )
                _tb__reset_approach_interp(self.runtime)
                self.runtime.wrench_command[:] = 0.0
                _tb__sync_compliance_state_to_current_pose(
                    self.controller, self.data, self.target_motor_pos
                )
                self.latest_state_ref = self.controller._last_state
                print("[model_based] Phase transition: kneel -> approach")
        elif self.runtime.phase == "approach":
            reached = _tb__run_approach_phase(
                self.runtime, self.cfg, state, self.control_dt
            )
            approach_timed_out = float(self.cfg.approach_timeout) > 0.0 and (
                t - self.runtime.phase_start_time
            ) >= float(self.cfg.approach_timeout)
            if (not reached) and approach_timed_out:
                reached = True
            command_matrix = _tb__build_command_matrix(
                self.runtime, self.measured_wrenches, self.site_names
            )
            self.target_motor_pos, state_ref = _tb__run_compliance_step(
                self.controller,
                self.data,
                t,
                command_matrix,
                self.target_motor_pos,
                self.measured_wrenches,
                self.site_names,
            )
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
            command = _tb__poll_keyboard_command(self.control_receiver)
            mode_changed = _tb__update_goal_from_keyboard_and_time(
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
                _tb__reset_approach_interp(self.runtime)
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
                print(
                    "[model_based][goal_velocity] "
                    f"t={float(t):.3f} "
                    f"base={float(self.runtime.goal_angular_velocity):.6f} "
                    f"delta={np.asarray(self.runtime.delta_goal_angular_velocity, dtype=np.float64).tolist()} "
                    f"total_vec={total_angular_velocity_vec.tolist()} "
                    f"total_mag={total_angular_velocity_mag:.6f} "
                    f"goal_angle={float(self.runtime.goal_angle):.6f}rad "
                    f"({np.degrees(float(self.runtime.goal_angle)):.2f}deg)"
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
                distributed_motion = _tb__distribute_rigid_body_motion(
                    self.runtime, ochs_solution, state, self.control_dt
                )
                _tb__maybe_print_ochs_world_velocity(
                    self.runtime, self.cfg, t, distributed_motion
                )
                _tb__assign_stiffness(
                    self.runtime,
                    self.cfg,
                    distributed_motion["left_linvel"],
                    distributed_motion["right_linvel"],
                )
                _tb__integrate_pose_command(
                    self.runtime, self.cfg, distributed_motion, self.control_dt
                )
                _tb__update_expected_ball_pos(
                    self.runtime, self.cfg, self.control_dt, state
                )
                _tb__update_delta_goal(self.runtime, self.cfg, state)
                command_matrix = _tb__build_command_matrix(
                    self.runtime, self.measured_wrenches, self.site_names
                )
                self.target_motor_pos, state_ref = _tb__run_compliance_step(
                    self.controller,
                    self.data,
                    t,
                    command_matrix,
                    self.target_motor_pos,
                    self.measured_wrenches,
                    self.site_names,
                )
                if state_ref is not None:
                    self.latest_state_ref = state_ref
        else:
            raise ValueError(f"Unknown phase: {self.runtime.phase}")

        _tb__update_mocap_targets_from_state_ref(
            self.data,
            self.left_mocap_id,
            self.right_mocap_id,
            self.latest_state_ref,
        )
        if (
            self.runtime.phase in ("approach", "model_based")
            and self.hold_indices.size > 0
        ):
            self.target_motor_pos[self.hold_indices] = self.kneel_hold_motor_pos[
                self.hold_indices
            ]

        return {}, np.asarray(self.target_motor_pos, dtype=np.float32).copy()

    def close(self) -> None:
        if self.control_receiver is not None and hasattr(
            self.control_receiver, "close"
        ):
            self.control_receiver.close()


class ModelBasedPolicy:
    """Per-tick model-based compliance policy wrapper."""

    def __init__(
        self,
        impl: Any,
    ) -> None:
        self.impl = impl
        self.control_dt = float(impl.control_dt)
        self.done = False

    @classmethod
    def from_argv(
        cls,
        argv: Sequence[str],
        *,
        robot: str,
        sim: str,
        vis: bool,
        plot: bool,
    ) -> "ModelBasedPolicy":
        del plot
        if str(sim) != "mujoco":
            raise ValueError(
                "compliance_model_based currently supports only --sim mujoco"
            )
        if str(robot) == "toddlerbot":
            if len(argv) > 0:
                raise SystemExit(
                    "toddlerbot model-based policy does not accept extra CLI args."
                )
            return cls(ToddlerBotModelBasedPolicy(vis=bool(vis)))
        args = build_parser().parse_args(list(argv))
        return cls(LeapModelBasedPolicy(args, vis=bool(vis)))

    def step(self, obs: Any, sim: Any) -> tuple[dict[str, float], np.ndarray]:
        out = self.impl.step(obs, sim)
        self.done = bool(getattr(self.impl, "done", False))
        return out

    def close(self) -> None:
        self.impl.close()


ComplianceModelBasedPolicy = ModelBasedPolicy
