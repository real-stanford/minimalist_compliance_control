"""Lightweight MuJoCo-based wrench simulation backend.

This module intentionally avoids any dependency on the larger project sim stack.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import gin
import mujoco
import mujoco.viewer
import numpy as np
import numpy.typing as npt


@gin.configurable
@dataclass
class WrenchSimConfig:
    """Configuration for the local MuJoCo wrench sim."""

    xml_path: str
    site_names: Sequence[str]
    fixed_base: bool = True
    view: bool = False
    render: bool = False
    render_width: int = 640
    render_height: int = 480
    render_camera: Optional[str] = None


class WrenchSim:
    """Standalone MuJoCo wrapper to compute site Jacobians and bias torques."""

    def __init__(self, config: WrenchSimConfig):
        self.config = config
        self.model = mujoco.MjModel.from_xml_path(config.xml_path)
        self.data = mujoco.MjData(self.model)
        self.site_ids: Dict[str, int] = {}
        for site in config.site_names:
            site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site)
            if site_id < 0:
                raise ValueError(f"Site {site!r} not found in XML: {config.xml_path}")
            self.site_ids[site] = int(site_id)

        self.jacp = np.zeros((3, self.model.nv), dtype=np.float64)
        self.jacr = np.zeros((3, self.model.nv), dtype=np.float64)
        self.renderer: Optional[mujoco.Renderer] = None
        self._frames: List[np.ndarray] = []
        self.viewer = None
        self._debug_site_forces: Dict[str, npt.NDArray[np.float32]] = {}
        self._debug_force_vis_scale = 0.02
        self._debug_site_targets: Dict[str, npt.NDArray[np.float32]] = {}
        self._debug_target_axis_length = 0.04
        if self.config.render:
            self._ensure_renderer()

    def _ensure_renderer(self) -> None:
        if self.renderer is not None:
            return
        self.renderer = mujoco.Renderer(
            self.model,
            width=int(self.config.render_width),
            height=int(self.config.render_height),
        )

    def _ensure_viewer(self) -> None:
        if self.viewer is not None:
            return
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

    def set_qpos(self, qpos: npt.NDArray[np.float32]) -> None:
        qpos = np.asarray(qpos, dtype=np.float32)
        if qpos.shape[0] != self.model.nq:
            raise ValueError(f"qpos size {qpos.shape[0]} != model.nq {self.model.nq}")
        self.data.qpos[:] = qpos

    def set_joint_positions(
        self, joint_pos: Dict[str, float] | npt.NDArray[np.float32]
    ) -> None:
        """Set joint positions directly in qpos."""
        if isinstance(joint_pos, dict):
            for name, value in joint_pos.items():
                joint_id = mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_JOINT, name
                )
                if joint_id < 0:
                    raise ValueError(f"Joint {name!r} not found in XML.")
                qpos_adr = int(self.model.jnt_qposadr[joint_id])
                jnt_type = int(self.model.jnt_type[joint_id])
                if jnt_type == mujoco.mjtJoint.mjJNT_FREE:
                    qpos_size = 7
                else:
                    qpos_size = 1
                if np.ndim(value) == 0:
                    if qpos_size != 1:
                        raise ValueError(
                            f"Joint {name!r} expects qpos size {qpos_size}, got scalar."
                        )
                    self.data.qpos[qpos_adr] = float(value)
                else:
                    arr = np.asarray(value, dtype=np.float32).reshape(-1)
                    if arr.size != qpos_size:
                        raise ValueError(
                            f"Joint {name!r} expects qpos size {qpos_size}, got {arr.size}"
                        )
                    self.data.qpos[qpos_adr : qpos_adr + qpos_size] = arr
            return

        values = np.asarray(joint_pos, dtype=np.float32).reshape(-1)
        if values.shape[0] == self.model.nq:
            self.data.qpos[:] = values
            return
        if values.shape[0] != self.model.njnt:
            raise ValueError(
                f"joint_pos size {values.shape[0]} != model.njnt {self.model.njnt} or model.nq {self.model.nq}"
            )
        for joint_id in range(self.model.njnt):
            qpos_adr = int(self.model.jnt_qposadr[joint_id])
            jnt_type = int(self.model.jnt_type[joint_id])
            if jnt_type == mujoco.mjtJoint.mjJNT_FREE:
                qpos_size = 7
            else:
                qpos_size = 1
            if qpos_size != 1:
                raise ValueError(
                    "Joint position vector only supports 1-DoF joints; "
                    "provide full qpos for free joints."
                )
            self.data.qpos[qpos_adr] = float(values[joint_id])

    def set_dof_positions(
        self, dof_indices: npt.NDArray[np.int32], values: npt.NDArray[np.float32]
    ) -> None:
        """Set qpos entries using DOF (nv) indices for 1-DoF joints."""
        dof_indices = np.asarray(dof_indices, dtype=np.int32).reshape(-1)
        values = np.asarray(values, dtype=np.float32).reshape(-1)
        if dof_indices.shape[0] != values.shape[0]:
            raise ValueError(
                f"dof_indices size {dof_indices.shape[0]} != values size {values.shape[0]}"
            )
        for dof_id, value in zip(dof_indices, values, strict=False):
            joint_id = int(self.model.dof_jntid[dof_id])
            jnt_type = int(self.model.jnt_type[joint_id])
            if jnt_type == mujoco.mjtJoint.mjJNT_FREE:
                raise ValueError(
                    "set_dof_positions does not support free joints; provide full qpos."
                )
            qpos_adr = int(self.model.jnt_qposadr[joint_id])
            self.data.qpos[qpos_adr] = float(value)

    def set_root_quat(self, quat_wxyz: npt.NDArray[np.float32]) -> None:
        quat = np.asarray(quat_wxyz, dtype=np.float32)
        if quat.shape[0] != 4:
            raise ValueError("Root quat must be length 4 (wxyz).")
        if self.model.nq < 7:
            return
        self.data.qpos[3:7] = quat

    def set_root_pos(self, pos_xyz: npt.NDArray[np.float32]) -> None:
        pos = np.asarray(pos_xyz, dtype=np.float32)
        if pos.shape[0] != 3:
            raise ValueError("Root pos must be length 3 (xyz).")
        if self.model.nq < 3:
            return
        self.data.qpos[0:3] = pos

    def forward(self) -> None:
        mujoco.mj_forward(self.model, self.data)

    def visualize(self) -> None:
        if self.viewer is None:
            self._ensure_viewer()
        if self.viewer is None:
            return
        if not self.viewer.is_running():
            self.viewer.close()
            self.viewer = None
            return
        self._draw_debug_overlays()
        self.viewer.sync()

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
            site_id = self.site_ids.get(site_name, -1)
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

    def render(self, camera: Optional[str] = None) -> np.ndarray:
        self._ensure_renderer()
        assert self.renderer is not None
        cam = camera if camera is not None else self.config.render_camera
        if cam is None or cam == "":
            self.renderer.update_scene(self.data)
        else:
            self.renderer.update_scene(self.data, camera=cam)
        return self.renderer.render().copy()

    def record_frame(self, camera: Optional[str] = None) -> None:
        frame = self.render(camera=camera)
        self._frames.append(frame)

    def reset_recording(self) -> None:
        self._frames.clear()

    def save_recording(
        self, exp_folder_path: str, fps: float = 30.0, name: str = "wrench_sim.mp4"
    ) -> None:
        if not self._frames:
            return
        from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

        os.makedirs(exp_folder_path, exist_ok=True)
        out_path = os.path.join(exp_folder_path, name)
        clip = ImageSequenceClip(self._frames, fps=float(fps))
        clip.write_videofile(
            out_path,
            fps=float(fps),
            codec="libx264",
            audio=False,
            logger=None,
            verbose=False,
        )
        if hasattr(clip, "close"):
            clip.close()

    def close(self) -> None:
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def site_jacobian(
        self, site_name: str
    ) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        site_id = self.site_ids[site_name]
        self.jacp.fill(0.0)
        self.jacr.fill(0.0)
        mujoco.mj_jacSite(self.model, self.data, self.jacp, self.jacr, site_id)
        return self.jacp.copy(), self.jacr.copy()

    def bias_torque(self) -> npt.NDArray[np.float32]:
        return np.asarray(self.data.qfrc_bias, dtype=np.float32).copy()

    def joint_dof_indices(self, joint_names: Iterable[str]) -> npt.NDArray[np.int32]:
        idx = []
        for name in joint_names:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                raise ValueError(f"Joint {name!r} not found in XML.")
            dof_adr = int(self.model.jnt_dofadr[jid])
            dof_num = int(self.model.jnt_dofnum[jid])
            idx.extend(range(dof_adr, dof_adr + dof_num))
        return np.asarray(idx, dtype=np.int32)
