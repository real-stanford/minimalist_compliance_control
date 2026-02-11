"""IK solver utilities for compliance reference (decoupled from robot config)."""

from __future__ import annotations

from dataclasses import dataclass

# from typing import List, Optional
from typing import Callable, Sequence

import mink
import mujoco

from minimum_compliance.utils.array_utils import ArrayType, R
from minimum_compliance.utils.array_utils import array_lib as np
import gin

JointToActuatorFn = Callable[[ArrayType], ArrayType]


def _sensor_slice(model: mujoco.MjModel, sensor_id: int) -> slice:
    start_adr = int(model.sensor_adr[sensor_id])
    end_adr = (
        int(model.sensor_adr[sensor_id + 1])
        if sensor_id + 1 < model.nsensor
        else int(model.nsensordata)
    )
    return slice(start_adr, end_adr)


@gin.configurable
@dataclass
class IKGains:
    pos_kp: float = 100.0
    pos_kd: float = 20.0
    rot_kp: float = 5.0
    rot_kd: float = 0.1
    default_kp: float = 5.0


@gin.configurable
class JacobianIK:
    def __init__(
        self,
        model: mujoco.MjModel,
        site_ids: Sequence[int],
        linvel_sensor_ids: Sequence[int],
        angvel_sensor_ids: Sequence[int],
        q_start_idx: int,
        qd_start_idx: int,
        joint_indices: ArrayType,
        joint_to_actuator_fn: JointToActuatorFn,
        mass: float,
        inertia_diag: ArrayType,
        gains: IKGains,
    ) -> None:
        self.model = model
        self.site_ids = list(site_ids)
        self.linvel_sensor_ids = list(linvel_sensor_ids)
        self.angvel_sensor_ids = list(angvel_sensor_ids)
        self.q_start_idx = int(q_start_idx)
        self.qd_start_idx = int(qd_start_idx)
        self.joint_indices = np.asarray(joint_indices, dtype=np.int32)
        self.joint_to_actuator_fn = joint_to_actuator_fn
        self.mass = float(mass)
        self.inertia_diag = np.asarray(inertia_diag, dtype=np.float32)
        self.gains = gains

    def solve(
        self,
        data: mujoco.MjData,
        x_ref: ArrayType,
        v_ref: ArrayType,
        a_ref: ArrayType,
    ) -> ArrayType:
        joint_rows = self.qd_start_idx + self.joint_indices
        joint_torque = np.zeros(len(self.joint_indices), dtype=np.float32)

        for idx, site_id in enumerate(self.site_ids):
            jacp = np.zeros((3, self.model.nv))
            jacr = np.zeros((3, self.model.nv))
            mujoco.mj_jacSite(self.model, data, jacp, jacr, site_id)
            J_full = np.concatenate([jacp, jacr]).T

            J = J_full[joint_rows, :]

            site_pos = data.site_xpos[site_id]
            site_rotmat = data.site_xmat[site_id].reshape(3, 3)

            lin_id = self.linvel_sensor_ids[idx]
            if lin_id >= 0:
                site_linvel = data.sensordata[_sensor_slice(self.model, lin_id)]
            else:
                site_linvel = np.zeros(3, dtype=np.float32)

            ang_id = self.angvel_sensor_ids[idx]
            if ang_id >= 0:
                site_angvel = data.sensordata[_sensor_slice(self.model, ang_id)]
            else:
                site_angvel = np.zeros(3, dtype=np.float32)

            x_obs = np.concatenate([site_pos, R.from_matrix(site_rotmat).as_rotvec()])
            v_obs = np.concatenate([site_linvel, site_angvel])

            pos_error = x_ref[idx, :3] - x_obs[:3]
            vel_error = v_ref[idx, :3] - v_obs[:3]
            rot_error = (
                (R.from_rotvec(x_ref[idx, 3:6]) * R.from_rotvec(x_obs[3:6]).inv())
                .as_rotvec()
                .astype(np.float32)
            )
            ang_error = v_ref[idx, 3:6] - v_obs[3:6]

            lin_task = self.gains.pos_kp * pos_error + self.gains.pos_kd * vel_error
            ang_task = self.gains.rot_kp * rot_error + self.gains.rot_kd * ang_error

            joint_torque = (
                joint_torque
                + J[:, :3] @ (lin_task * self.mass)
                + J[:, 3:6] @ (ang_task * self.inertia_diag)
            )

        joint_delta_pos = joint_torque / self.gains.default_kp
        joint_pos = data.qpos[self.q_start_idx + self.joint_indices] + joint_delta_pos
        return self.joint_to_actuator_fn(joint_pos)


@gin.configurable
class MinkIK:
    def __init__(
        self,
        model: mujoco.MjModel,
        site_names: Sequence[str],
        joint_indices: ArrayType,
        joint_to_actuator_fn: JointToActuatorFn,
        ik_position_only: bool = False,
        source_q_start_idx: int = 0,
    ) -> None:
        self.model = model
        self.site_names = list(site_names)
        self.joint_indices = np.asarray(joint_indices, dtype=np.int32)
        self.joint_to_actuator_fn = joint_to_actuator_fn
        self.ik_position_only = ik_position_only
        self.source_q_start_idx = int(source_q_start_idx)

        self.config = mink.Configuration(model)
        try:
            self.config.update_from_keyframe("home")
        except Exception:
            pass
        self.tasks = None
        self.limits = [mink.ConfigurationLimit(model)]

    def solve(
        self,
        data: mujoco.MjData,
        x_ref: ArrayType,
        v_ref: ArrayType,
        dt: float,
        num_iter: int = 5,
        damping: float = 1e-2,
    ) -> ArrayType:
        if self.config.data.qpos.shape[0] == data.qpos.shape[0]:
            self.config.data.qpos[:] = data.qpos.copy()
        else:
            start = self.source_q_start_idx
            end = start + self.config.data.qpos.shape[0]
            self.config.data.qpos[:] = 0.0
            if end <= data.qpos.shape[0]:
                self.config.data.qpos[:] = data.qpos[start:end]
            elif start < data.qpos.shape[0]:
                available = data.qpos[start:]
                self.config.data.qpos[: available.shape[0]] = available
        mujoco.mj_forward(self.config.model, self.config.data)

        if self.tasks is None:
            self.tasks = {}
            posture_task = mink.PostureTask(self.config.model, cost=0.1)
            posture_task.set_target_from_configuration(self.config)
            self.tasks["posture"] = posture_task

            for site_name in self.site_names:
                frame_task = mink.FrameTask(
                    frame_name=site_name,
                    frame_type="site",
                    position_cost=10.0,
                    orientation_cost=0.0 if self.ik_position_only else 0.5,
                    lm_damping=1.0,
                )
                self.tasks[site_name] = frame_task

        for site_name in self.site_names:
            idx = self.site_names.index(site_name)
            target_pos = x_ref[idx, :3]
            target_rotvec = x_ref[idx, 3:6]
            target_rotmat = R.from_rotvec(target_rotvec).as_matrix()
            target_rot = mink.SO3.from_matrix(target_rotmat)
            target = mink.SE3.from_rotation_and_translation(target_rot, target_pos)
            self.tasks[site_name].set_target(target)

        for _ in range(num_iter):
            vel = mink.solve_ik(
                self.config,
                list(self.tasks.values()),
                dt,
                solver="quadprog",
                damping=damping,
                limits=self.limits,
            )
            self.config.integrate_inplace(vel, dt)

        joint_pos = self.config.data.qpos[self.joint_indices].copy()
        return self.joint_to_actuator_fn(joint_pos)
