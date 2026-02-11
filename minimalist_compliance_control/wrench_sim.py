"""Lightweight MuJoCo-based wrench simulation backend.

This module intentionally avoids any dependency on the larger project sim stack.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import mujoco
import mujoco.viewer
import numpy as np
import numpy.typing as npt
import gin


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
        if self.config.render:
            self._ensure_renderer()
        if self.config.view:
            self._ensure_viewer()

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
        self.viewer.sync()

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
        import imageio.v2 as imageio

        os.makedirs(exp_folder_path, exist_ok=True)
        out_path = os.path.join(exp_folder_path, name)
        imageio.mimsave(out_path, self._frames, fps=float(fps))

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
