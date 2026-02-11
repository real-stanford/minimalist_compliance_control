"""IK solver utilities for compliance reference."""

from __future__ import annotations

from typing import Callable, Sequence

import gin
import mink
import mujoco
import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation as R

JointToActuatorFn = Callable[[npt.NDArray[np.float32]], npt.NDArray[np.float32]]


@gin.configurable
class MinkIK:
    def __init__(
        self,
        model: mujoco.MjModel,
        site_names: Sequence[str],
        joint_indices: npt.NDArray[np.int32],
        joint_to_actuator_fn: JointToActuatorFn,
        ik_position_only: bool,
        source_q_start_idx: int,
    ) -> None:
        self.model = model
        self.site_names = list(site_names)
        self.site_name_to_idx = {name: idx for idx, name in enumerate(self.site_names)}
        self.joint_indices = np.asarray(joint_indices, dtype=np.int32)
        self.joint_to_actuator_fn = joint_to_actuator_fn
        self.ik_position_only = bool(ik_position_only)
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
        x_ref: npt.NDArray[np.float32],
        v_ref: npt.NDArray[np.float32],
        dt: float,
        num_iter: int,
        damping: float,
    ) -> npt.NDArray[np.float32]:
        del v_ref

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
            idx = self.site_name_to_idx[site_name]
            target_pos = x_ref[idx, :3]
            target_rotvec = x_ref[idx, 3:6]
            target_rotmat = R.from_rotvec(target_rotvec).as_matrix()
            target_rot = mink.SO3.from_matrix(target_rotmat)
            target = mink.SE3.from_rotation_and_translation(target_rot, target_pos)
            self.tasks[site_name].set_target(target)

        for _ in range(int(num_iter)):
            vel = mink.solve_ik(
                self.config,
                list(self.tasks.values()),
                dt,
                solver="quadprog",
                damping=float(damping),
                limits=self.limits,
            )
            self.config.integrate_inplace(vel, dt)

        joint_pos = self.config.data.qpos[self.joint_indices].copy()
        return self.joint_to_actuator_fn(joint_pos)
