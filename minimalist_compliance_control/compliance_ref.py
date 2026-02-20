"""Compliance reference (site-based, no MotionReference dependency)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import gin
import mujoco
import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation as R

from minimalist_compliance_control.ik_solvers import MinkIK


@dataclass(frozen=True)
class CommandLayout:
    width: int = 54
    position: slice = slice(0, 3)
    orientation: slice = slice(3, 6)
    measured_force: slice = slice(6, 9)
    measured_torque: slice = slice(9, 12)
    kp_pos: slice = slice(12, 21)
    kp_rot: slice = slice(21, 30)
    kd_pos: slice = slice(30, 39)
    kd_rot: slice = slice(39, 48)
    force: slice = slice(48, 51)
    torque: slice = slice(51, 54)


COMMAND_LAYOUT = CommandLayout()


@dataclass
class ComplianceState:
    x_ref: npt.NDArray[np.float32]
    x_ik: npt.NDArray[np.float32]
    v_ref: npt.NDArray[np.float32]
    a_ref: npt.NDArray[np.float32]
    motor_pos: npt.NDArray[np.float32]
    qpos: npt.NDArray[np.float32]


@gin.configurable
class ComplianceReference:
    """Site-based compliance reference without robot-specific assumptions."""

    def __init__(
        self,
        dt: float,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        site_names: Sequence[str],
        actuator_indices: npt.NDArray[np.int32],
        joint_indices: npt.NDArray[np.int32],
        joint_to_actuator_fn: Callable,
        actuator_to_joint_fn: Callable,
        default_motor_pos: npt.NDArray[np.float32],
        default_qpos: npt.NDArray[np.float32],
        fixed_model_xml_path: Optional[str],
        q_start_idx: int,
        qd_start_idx: int,
        ik_position_only: bool,
        mass: float,
        inertia_diag: npt.NDArray[np.float32],
        mink_num_iter: int,
        mink_damping: float,
        avoid_self_collision: bool = False,
    ) -> None:
        del data

        self.dt = float(dt)
        self.control_dt = float(dt)
        self.model = model
        self.mass = float(mass)
        self.inertia_diag = np.asarray(inertia_diag, dtype=np.float32)
        self.q_start_idx = int(q_start_idx)
        self.qd_start_idx = int(qd_start_idx)

        self.site_names = list(site_names)
        if not self.site_names:
            raise ValueError("site_names must be provided.")
        site_set = set(self.site_names)
        model_hint = (fixed_model_xml_path or "").lower()
        self.is_toddlerbot = ("toddlerbot" in model_hint) or {
            "left_hand_center",
            "right_hand_center",
        }.issubset(site_set)

        self.actuator_indices = np.asarray(actuator_indices, dtype=np.int32)
        self.joint_indices = np.asarray(joint_indices, dtype=np.int32)
        self.joint_to_actuator_fn = joint_to_actuator_fn
        self.actuator_to_joint_fn = actuator_to_joint_fn

        self.default_motor_pos = np.asarray(default_motor_pos, dtype=np.float32)
        self.default_qpos = np.asarray(default_qpos, dtype=np.float32)

        self.site_ids: list[int] = []
        for site_name in self.site_names:
            site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
            if site_id < 0:
                raise ValueError(f"Site '{site_name}' not found in model.")
            self.site_ids.append(int(site_id))

        self.ik_position_only = bool(ik_position_only)
        self.mink_num_iter = int(mink_num_iter)
        self.mink_damping = float(mink_damping)

        self.fixed_model = None
        self.fixed_data = None
        mink_model = model
        if fixed_model_xml_path:
            self.fixed_model = mujoco.MjModel.from_xml_path(fixed_model_xml_path)
            self.fixed_data = mujoco.MjData(self.fixed_model)
            mink_model = self.fixed_model
        self.ik_site_ids: list[int] = []
        for site_name in self.site_names:
            site_id = mujoco.mj_name2id(mink_model, mujoco.mjtObj.mjOBJ_SITE, site_name)
            if site_id < 0:
                raise ValueError(f"Site '{site_name}' not found in IK model.")
            self.ik_site_ids.append(int(site_id))

        self.mink_ik = MinkIK(
            model=mink_model,
            site_names=self.site_names,
            joint_indices=self.joint_indices,
            joint_to_actuator_fn=self.joint_to_actuator_fn,
            ik_position_only=self.ik_position_only,
            source_q_start_idx=self.q_start_idx,
            enable_self_collision_avoidance=bool(avoid_self_collision),
            is_toddlerbot=bool(self.is_toddlerbot),
        )

        default_state = self.get_default_state()
        self.site_home_pose = np.asarray(default_state.x_ref, dtype=np.float32).copy()

    def get_x_ref_from_motor_pos(
        self, motor_pos: npt.NDArray[np.float32]
    ) -> npt.NDArray[np.float32]:
        """Compute world-frame site pose references for a given motor position."""
        motor_arr = np.asarray(motor_pos, dtype=np.float32).reshape(-1)
        if int(np.max(self.actuator_indices)) >= int(motor_arr.shape[0]):
            raise ValueError(
                f"motor_pos length {motor_arr.shape[0]} does not cover actuator indices."
            )

        qpos = np.asarray(self.default_qpos, dtype=np.float32).copy()
        qpos_indices = int(self.q_start_idx) + self.joint_indices
        if int(np.min(qpos_indices)) < 0 or int(np.max(qpos_indices)) >= int(
            qpos.shape[0]
        ):
            raise ValueError("Computed qpos indices are out of bounds.")

        actuator_pos = np.asarray(motor_arr[self.actuator_indices], dtype=np.float32)
        joint_pos = np.asarray(
            self.actuator_to_joint_fn(actuator_pos), dtype=np.float32
        )
        if joint_pos.shape != self.joint_indices.shape:
            raise ValueError(
                f"actuator_to_joint_fn returned shape {joint_pos.shape}, "
                f"expected {self.joint_indices.shape}."
            )
        qpos[qpos_indices] = joint_pos

        data = mujoco.MjData(self.model)
        data.qpos[:] = qpos
        mujoco.mj_forward(self.model, data)

        x_ref = np.zeros((len(self.site_names), 6), dtype=np.float32)
        for idx, site_id in enumerate(self.site_ids):
            x_ref[idx, 0:3] = np.asarray(data.site(site_id).xpos, dtype=np.float32)
            rotmat = np.asarray(data.site(site_id).xmat, dtype=np.float32).reshape(3, 3)
            x_ref[idx, 3:6] = R.from_matrix(rotmat).as_rotvec().astype(np.float32)
        return x_ref

    def get_default_state(self) -> ComplianceState:
        num_sites = len(self.site_names)
        zeros = np.zeros((num_sites, 6), dtype=np.float32)

        data = mujoco.MjData(self.model)
        data.qpos[:] = self.default_qpos.copy()
        mujoco.mj_forward(self.model, data)

        home_pose = np.zeros((num_sites, 6), dtype=np.float32)
        for idx, site_id in enumerate(self.site_ids):
            home_pose[idx, 0:3] = np.asarray(data.site(site_id).xpos, dtype=np.float32)
            rotmat = np.asarray(data.site(site_id).xmat, dtype=np.float32).reshape(3, 3)
            home_pose[idx, 3:6] = R.from_matrix(rotmat).as_rotvec().astype(np.float32)

        return ComplianceState(
            x_ref=home_pose.copy(),
            x_ik=home_pose.copy(),
            v_ref=zeros.copy(),
            a_ref=zeros.copy(),
            motor_pos=self.default_motor_pos.copy(),
            qpos=self.default_qpos.copy(),
        )

    def integrate_commands(
        self,
        x_prev: npt.NDArray[np.float32],
        v_prev: npt.NDArray[np.float32],
        command_matrix: npt.NDArray[np.float32],
    ) -> tuple[
        npt.NDArray[np.float32],
        npt.NDArray[np.float32],
        npt.NDArray[np.float32],
    ]:
        positions = command_matrix[:, COMMAND_LAYOUT.position]
        orientations = command_matrix[:, COMMAND_LAYOUT.orientation]
        measured_force = command_matrix[:, COMMAND_LAYOUT.measured_force]
        measured_torque = command_matrix[:, COMMAND_LAYOUT.measured_torque]
        cmd_force = command_matrix[:, COMMAND_LAYOUT.force]
        cmd_torque = command_matrix[:, COMMAND_LAYOUT.torque]
        net_force = measured_force + cmd_force
        net_torque = measured_torque + cmd_torque
        kp_pos = command_matrix[:, COMMAND_LAYOUT.kp_pos].reshape(-1, 3, 3)
        kp_rot = command_matrix[:, COMMAND_LAYOUT.kp_rot].reshape(-1, 3, 3)
        kd_pos = command_matrix[:, COMMAND_LAYOUT.kd_pos].reshape(-1, 3, 3)
        kd_rot = command_matrix[:, COMMAND_LAYOUT.kd_rot].reshape(-1, 3, 3)

        x_next = x_prev.copy()
        v_next = v_prev.copy()
        a_next = np.zeros_like(v_prev)

        idx = np.arange(len(self.site_names), dtype=np.int32)
        pos_prev = x_prev[idx, :3]
        vel_prev = v_prev[idx, :3]
        pos_des = positions[idx]
        pos_error = pos_des - pos_prev

        kp_term = np.matmul(kp_pos[idx], pos_error[..., None]).reshape(-1, 3)
        kd_term = np.matmul(kd_pos[idx], vel_prev[..., None]).reshape(-1, 3)
        lin_acc = (kp_term - kd_term + net_force[idx]) / self.mass
        vel_next = vel_prev + lin_acc * self.dt
        pos_next = pos_prev + vel_next * self.dt

        ori_prev = R.from_rotvec(x_prev[idx, 3:6])
        omega_prev = v_prev[idx, 3:6]
        ori_des = R.from_rotvec(orientations[idx])
        ori_error = (ori_des * ori_prev.inv()).as_rotvec()

        kp_rot_term = np.matmul(kp_rot[idx], ori_error[..., None]).reshape(-1, 3)
        kd_rot_term = np.matmul(kd_rot[idx], omega_prev[..., None]).reshape(-1, 3)
        ang_acc = (kp_rot_term - kd_rot_term + net_torque[idx]) / self.inertia_diag
        omega_next = omega_prev + ang_acc * self.dt
        ori_next = (R.from_rotvec(omega_next * self.dt) * ori_prev).as_rotvec()

        x_next[idx, 0:3] = pos_next
        x_next[idx, 3:6] = ori_next
        v_next[idx, 0:3] = vel_next
        v_next[idx, 3:6] = omega_next
        a_next[idx, 0:3] = lin_acc
        a_next[idx, 3:6] = ang_acc

        return x_next, v_next, a_next

    def get_actuator_ref(
        self,
        data: mujoco.MjData,
        x_ref: npt.NDArray[np.float32],
        v_ref: npt.NDArray[np.float32],
        a_ref: npt.NDArray[np.float32],
    ) -> npt.NDArray[np.float32]:
        del a_ref
        return self.mink_ik.solve(
            data,
            x_ref,
            v_ref,
            self.dt,
            num_iter=self.mink_num_iter,
            damping=self.mink_damping,
        )

    def get_state_ref(
        self,
        command_matrix: npt.NDArray[np.float32],
        last_state: ComplianceState,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        site_names: Optional[list[str]] = None,
        base_pos: Optional[npt.NDArray[np.float32]] = None,
        base_quat: Optional[npt.NDArray[np.float32]] = None,
    ) -> ComplianceState:
        del model, site_names

        x_ref, v_ref, a_ref = self.integrate_commands(
            np.asarray(last_state.x_ref, dtype=np.float32),
            np.asarray(last_state.v_ref, dtype=np.float32),
            command_matrix,
        )

        base_pos_arr = (
            np.asarray(base_pos, dtype=np.float32)
            if base_pos is not None
            else np.zeros(3, dtype=np.float32)
        )
        base_rot_arr = (
            np.asarray(base_quat, dtype=np.float32)
            if base_quat is not None
            else np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        )
        x_ref_for_ik = self.transform_x_ref_to_base_frame(
            x_ref,
            base_pos_arr,
            base_rot_arr,
        )

        actuator_pos = self.get_actuator_ref(data, x_ref_for_ik, v_ref, a_ref)
        x_ik = self.get_x_ik_world(base_pos_arr, base_rot_arr)
        motor_pos = self.default_motor_pos.copy()
        motor_pos[self.actuator_indices] = actuator_pos

        return ComplianceState(
            x_ref=x_ref,
            x_ik=x_ik,
            v_ref=v_ref,
            a_ref=a_ref,
            motor_pos=motor_pos,
            qpos=np.asarray(data.qpos, dtype=np.float32).copy(),
        )

    def get_x_ik_world(
        self,
        base_pos: npt.NDArray[np.float32],
        base_quat_wxyz: npt.NDArray[np.float32],
    ) -> npt.NDArray[np.float32]:
        x_ik_local = np.zeros((len(self.site_names), 6), dtype=np.float32)
        cfg_data = self.mink_ik.config.data
        for idx, site_id in enumerate(self.ik_site_ids):
            x_ik_local[idx, 0:3] = np.asarray(
                cfg_data.site_xpos[site_id], dtype=np.float32
            )
            rotmat = np.asarray(cfg_data.site_xmat[site_id], dtype=np.float32).reshape(
                3, 3
            )
            x_ik_local[idx, 3:6] = R.from_matrix(rotmat).as_rotvec().astype(np.float32)

        base_pos_arr = np.asarray(base_pos, dtype=np.float32).reshape(3)
        base_quat = np.asarray(base_quat_wxyz, dtype=np.float32).reshape(4)
        base_quat = base_quat / (np.linalg.norm(base_quat) + 1e-9)
        base_quat_xyzw = base_quat[[1, 2, 3, 0]]
        base_rot = R.from_quat(base_quat_xyzw)

        pos_world = base_rot.apply(x_ik_local[:, :3]) + base_pos_arr
        rot_world = (
            base_rot * R.from_rotvec(np.asarray(x_ik_local[:, 3:6], dtype=np.float32))
        ).as_rotvec()
        return np.concatenate(
            [
                np.asarray(pos_world, dtype=np.float32),
                np.asarray(rot_world, dtype=np.float32),
            ],
            axis=1,
        ).astype(np.float32)

    def transform_x_ref_to_base_frame(
        self,
        x_ref: npt.NDArray[np.float32],
        base_pos: npt.NDArray[np.float32],
        base_quat_wxyz: npt.NDArray[np.float32],
    ) -> npt.NDArray[np.float32]:
        base_pos = np.asarray(base_pos, dtype=np.float32).reshape(3)
        base_quat = np.asarray(base_quat_wxyz, dtype=np.float32).reshape(4)
        base_quat = base_quat / (np.linalg.norm(base_quat) + 1e-9)
        base_quat_xyzw = base_quat[[1, 2, 3, 0]]
        base_rot = R.from_quat(base_quat_xyzw)

        pos_world = np.asarray(x_ref[:, :3], dtype=np.float32)
        rotvec_world = np.asarray(x_ref[:, 3:6], dtype=np.float32)
        pos_local = base_rot.inv().apply(pos_world - base_pos)
        rot_world = R.from_rotvec(rotvec_world).as_matrix()
        rot_local = np.einsum("ij,njk->nik", base_rot.as_matrix().T, rot_world)
        rotvec_local = R.from_matrix(rot_local).as_rotvec().astype(np.float32)
        return np.concatenate([pos_local, rotvec_local], axis=-1).astype(np.float32)
