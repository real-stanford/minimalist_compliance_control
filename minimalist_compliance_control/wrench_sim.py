"""Lightweight MuJoCo-based wrench simulation backend.

This module intentionally avoids any dependency on the larger project sim stack.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import gin
import mujoco
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

    def set_qpos(self, qpos: npt.NDArray[np.float32]) -> None:
        qpos = np.asarray(qpos, dtype=np.float32)
        if qpos.shape[0] != self.model.nq:
            raise ValueError(f"qpos size {qpos.shape[0]} != model.nq {self.model.nq}")
        self.data.qpos[:] = qpos

    def forward(self) -> None:
        mujoco.mj_forward(self.model, self.data)

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
