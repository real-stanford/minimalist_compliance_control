"""Minimal simulation backends used by the compliance runners."""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Protocol, runtime_checkable

import mujoco
import mujoco.viewer
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

    def set_motor_target(self, motor_target: npt.NDArray[np.float32]) -> None:
        """Set motor target for the next backend step."""

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
        self.name = "mujoco"
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
        self.viewer = None
        self._debug_site_forces: Dict[str, npt.NDArray[np.float32]] = {}
        self._debug_force_vis_scale = 0.02
        self._debug_site_targets: Dict[str, npt.NDArray[np.float32]] = {}
        self._debug_target_axis_length = 0.04

    def step(self) -> None:
        for _ in range(self.n_substeps):
            if self.substep_control is not None:
                self.substep_control(self.data)
            mujoco.mj_step(self.model, self.data)

    def set_motor_target(self, motor_target: npt.NDArray[np.float32]) -> None:
        target = np.asarray(motor_target, dtype=np.float32).reshape(-1)
        if target.shape[0] != self.model.nu:
            raise ValueError(
                f"motor_target shape {target.shape[0]} must equal model.nu {self.model.nu}"
            )
        self.data.ctrl[:] = target

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

    def set_debug_site_targets(
        self,
        site_targets: Dict[str, npt.NDArray[np.float32]],
        axis_length: float = 0.04,
    ) -> None:
        self._debug_site_targets = {
            name: np.asarray(target, dtype=np.float32).reshape(6)
            for name, target in site_targets.items()
        }
        self._debug_target_axis_length = float(axis_length)

    def clear_debug_site_targets(self) -> None:
        self._debug_site_targets = {}

    def set_debug_site_forces(
        self,
        site_forces: Dict[str, npt.NDArray[np.float32]],
        vis_scale: float = 0.02,
    ) -> None:
        self._debug_site_forces = {
            name: np.asarray(force, dtype=np.float32).reshape(3)
            for name, force in site_forces.items()
        }
        self._debug_force_vis_scale = float(vis_scale)

    def clear_debug_site_forces(self) -> None:
        self._debug_site_forces = {}

    def _ensure_viewer(self) -> None:
        if self.viewer is not None:
            return
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

    def _draw_debug_overlays(self) -> None:
        if self.viewer is None:
            return
        if not hasattr(self.viewer, "user_scn"):
            return
        if not hasattr(self.viewer, "lock"):
            return

        with self.viewer.lock():
            scene = self.viewer.user_scn
            scene.ngeom = 0
            self._draw_debug_site_targets(scene)
            self._draw_debug_site_forces(scene)

    def _draw_connector(
        self,
        scene: mujoco.MjvScene,
        start: npt.NDArray[np.float64],
        end: npt.NDArray[np.float64],
        radius: float,
        color_rgba: npt.NDArray[np.float32],
        geom_type: int,
    ) -> None:
        if scene.ngeom >= scene.maxgeom:
            return
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(
            geom,
            geom_type,
            np.array([0.01, 0.01, 0.01], dtype=np.float64),
            np.zeros(3, dtype=np.float64),
            np.eye(3, dtype=np.float64).reshape(-1),
            np.asarray(color_rgba, dtype=np.float32),
        )
        mujoco.mjv_connector(
            geom,
            geom_type,
            float(radius),
            start,
            end,
        )
        scene.ngeom += 1

    def _rotvec_to_mat(
        self, rotvec: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        theta = float(np.linalg.norm(rotvec))
        if theta < 1e-12:
            return np.eye(3, dtype=np.float64)
        axis = rotvec / theta
        x, y, z = float(axis[0]), float(axis[1]), float(axis[2])
        k = np.array(
            [
                [0.0, -z, y],
                [z, 0.0, -x],
                [-y, x, 0.0],
            ],
            dtype=np.float64,
        )
        eye = np.eye(3, dtype=np.float64)
        return eye + np.sin(theta) * k + (1.0 - np.cos(theta)) * (k @ k)

    def _draw_debug_site_targets(self, scene: mujoco.MjvScene) -> None:
        if not self._debug_site_targets:
            return
        axis_colors = (
            np.array([1.0, 0.1, 0.1, 0.9], dtype=np.float32),
            np.array([0.1, 1.0, 0.1, 0.9], dtype=np.float32),
            np.array([0.1, 0.1, 1.0, 0.9], dtype=np.float32),
        )
        axis_length = float(self._debug_target_axis_length)
        for target in self._debug_site_targets.values():
            start = np.asarray(target[:3], dtype=np.float64)
            rotmat = self._rotvec_to_mat(np.asarray(target[3:6], dtype=np.float64))
            for axis_idx in range(3):
                end = start + axis_length * rotmat[:, axis_idx]
                self._draw_connector(
                    scene=scene,
                    start=start,
                    end=end,
                    radius=0.004,
                    color_rgba=axis_colors[axis_idx],
                    geom_type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                )
                if scene.ngeom >= scene.maxgeom:
                    return

    def _draw_debug_site_forces(self, scene: mujoco.MjvScene) -> None:
        if not self._debug_site_forces:
            return
        for site_name, force in self._debug_site_forces.items():
            site_id = self.wrench_sim.site_ids.get(site_name, -1)
            if site_id < 0:
                continue
            if scene.ngeom >= scene.maxgeom:
                break
            start = np.asarray(self.data.site_xpos[site_id], dtype=np.float64)
            end = start + np.asarray(force, dtype=np.float64) * float(
                self._debug_force_vis_scale
            )
            self._draw_connector(
                scene=scene,
                start=start,
                end=end,
                radius=0.005,
                color_rgba=np.array([1.0, 0.2, 0.2, 0.9], dtype=np.float32),
                geom_type=mujoco.mjtGeom.mjGEOM_ARROW,
            )

    def sync(self) -> bool:
        if not self.vis:
            return True
        if self.viewer is None:
            self._ensure_viewer()
        if self.viewer is None:
            return False
        if not self.viewer.is_running():
            self.viewer.close()
            self.viewer = None
            return False
        self._draw_debug_overlays()
        self.viewer.sync()
        if getattr(self.wrench_sim.config, "render", False):
            self.wrench_sim.record_frame()
        return True

    def close(self) -> None:
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        self.wrench_sim.close()
