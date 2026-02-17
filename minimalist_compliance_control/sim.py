"""Minimal simulation backends used by the compliance runners."""

from __future__ import annotations

from typing import Any, Callable, Protocol, runtime_checkable

import mujoco
import numpy as np
import numpy.typing as npt

from minimalist_compliance_control.utils import (
    MotorParams,
    compute_clamped_motor_torque,
)
from minimalist_compliance_control.wrench_sim import WrenchSim


@runtime_checkable
class BaseSim(Protocol):
    """Minimal interface required by compliance examples."""

    model: mujoco.MjModel
    data: mujoco.MjData

    def step(self) -> None:
        """Advance one backend step."""

    def get_observation(self) -> dict[str, Any]:
        """Return latest observation used by controllers."""

    def sync(self) -> bool:
        """Synchronize visualization/output and report whether to keep running."""

    def close(self) -> None:
        """Release backend resources."""


def build_clamped_torque_substep_control(
    qpos_adr: npt.NDArray[np.int32],
    qvel_adr: npt.NDArray[np.int32],
    motor_params: MotorParams,
    target_motor_pos_getter: Callable[[], npt.NDArray[np.float32]],
) -> Callable[[mujoco.MjData], None]:
    qpos_adr_arr = np.asarray(qpos_adr, dtype=np.int32)
    qvel_adr_arr = np.asarray(qvel_adr, dtype=np.int32)

    def _substep_control(data_step: mujoco.MjData) -> None:
        target_motor_pos = np.asarray(target_motor_pos_getter(), dtype=np.float32)
        q = data_step.qpos[qpos_adr_arr]
        q_dot = data_step.qvel[qvel_adr_arr]
        q_dot_dot = data_step.qacc[qvel_adr_arr]
        data_step.ctrl[:] = compute_clamped_motor_torque(
            target_motor_pos=target_motor_pos,
            q=q,
            q_dot=q_dot,
            q_dot_dot=q_dot_dot,
            motor_params=motor_params,
        )

    return _substep_control


def build_site_force_applier(
    model: mujoco.MjModel,
    site_ids: npt.NDArray[np.int32],
) -> Callable[[mujoco.MjData, npt.NDArray[np.float32]], None]:
    site_ids_arr = np.asarray(site_ids, dtype=np.int32).reshape(-1)
    torque_zero = np.zeros(3, dtype=np.float64)
    qfrc_tmp = np.zeros(model.nv, dtype=np.float64)

    def _apply_site_forces(
        data_step: mujoco.MjData,
        site_forces: npt.NDArray[np.float32],
    ) -> None:
        data_step.qfrc_applied[:] = 0.0
        forces = np.asarray(site_forces, dtype=np.float64)
        if forces.shape != (site_ids_arr.shape[0], 3):
            raise ValueError(
                f"site_forces must have shape ({site_ids_arr.shape[0]}, 3), got {forces.shape}"
            )
        for idx, site_id in enumerate(site_ids_arr):
            force = forces[idx]
            if not np.any(force):
                continue
            body_id = int(model.site_bodyid[site_id])
            point = np.asarray(data_step.site_xpos[site_id], dtype=np.float64)
            qfrc_tmp[:] = 0.0
            mujoco.mj_applyFT(
                model,
                data_step,
                force,
                torque_zero,
                point,
                body_id,
                qfrc_tmp,
            )
            data_step.qfrc_applied[:] += qfrc_tmp

    return _apply_site_forces


class MuJoCoSim:
    """Thin wrapper around the local WrenchSim MuJoCo state."""

    def __init__(
        self,
        wrench_sim: WrenchSim,
        control_dt: float,
        sim_dt: float | None = None,
        vis: bool = False,
        substep_control: Callable[[mujoco.MjData], None] | None = None,
    ) -> None:
        self.wrench_sim = wrench_sim
        self.model = wrench_sim.model
        self.data = wrench_sim.data
        self.vis = bool(vis)
        self.substep_control = substep_control
        self.control_dt = float(control_dt)
        if self.control_dt <= 0.0:
            raise ValueError("control_dt must be > 0.")
        if sim_dt is not None:
            self.model.opt.timestep = float(sim_dt)
        self.sim_dt = float(self.model.opt.timestep)
        if self.sim_dt <= 0.0:
            raise ValueError("sim_dt must be > 0.")
        self.n_substeps = max(1, int(round(self.control_dt / self.sim_dt)))
        trnid = np.asarray(self.model.actuator_trnid[:, 0], dtype=np.int32)
        if np.any(trnid < 0):
            raise ValueError("All actuators must map to valid joints in MuJoCoSim.")
        self._qpos_adr = np.asarray(self.model.jnt_qposadr[trnid], dtype=np.int32)
        self._qvel_adr = np.asarray(self.model.jnt_dofadr[trnid], dtype=np.int32)

    def step(self) -> None:
        for _ in range(self.n_substeps):
            if self.substep_control is not None:
                self.substep_control(self.data)
            mujoco.mj_step(self.model, self.data)

    def get_observation(self) -> dict[str, Any]:
        return {
            "time": float(self.data.time),
            "motor_pos": np.asarray(
                self.data.qpos[self._qpos_adr], dtype=np.float32
            ).copy(),
            "motor_vel": np.asarray(
                self.data.qvel[self._qvel_adr], dtype=np.float32
            ).copy(),
            "motor_acc": np.asarray(
                self.data.qacc[self._qvel_adr], dtype=np.float32
            ).copy(),
            "motor_tor": np.asarray(self.data.actuator_force, dtype=np.float32).copy(),
            "qpos": np.asarray(self.data.qpos, dtype=np.float32).copy(),
            "qvel": np.asarray(self.data.qvel, dtype=np.float32).copy(),
        }

    def sync(self) -> bool:
        if not self.vis:
            return True
        self.wrench_sim.visualize()
        return self.wrench_sim.viewer is not None

    def close(self) -> None:
        self.wrench_sim.close()
