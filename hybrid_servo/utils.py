from __future__ import annotations

import os

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

from minimalist_compliance_control.controller import ComplianceController


def find_repo_root(start_dir: str) -> str:
    cur = os.path.abspath(start_dir)
    while True:
        if os.path.isfile(os.path.join(cur, "pyproject.toml")):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            raise FileNotFoundError(
                "Could not locate repository root (pyproject.toml)."
            )
        cur = parent


def sync_compliance_state_to_current_pose(
    controller: ComplianceController,
    data: mujoco.MjData,
    motor_pos: np.ndarray,
) -> None:
    compliance_ref = controller.compliance_ref
    if compliance_ref is None:
        return

    ref_state = compliance_ref.get_default_state()
    site_ids = compliance_ref.site_ids
    x_ref = np.zeros((len(site_ids), 6), dtype=np.float32)
    for idx, site_id in enumerate(site_ids):
        pos = np.asarray(data.site_xpos[site_id], dtype=np.float32).copy()
        rotmat = (
            np.asarray(data.site_xmat[site_id], dtype=np.float32).reshape(3, 3).copy()
        )
        rotvec = R.from_matrix(rotmat).as_rotvec().astype(np.float32)
        x_ref[idx, :3] = pos
        x_ref[idx, 3:] = rotvec

    ref_state.x_ref = x_ref.copy()
    ref_state.v_ref = np.zeros_like(x_ref)
    ref_state.a_ref = np.zeros_like(x_ref)
    ref_state.qpos = np.asarray(data.qpos, dtype=np.float32).copy()
    ref_state.motor_pos = np.asarray(motor_pos, dtype=np.float32).copy()
    controller._last_state = ref_state


def get_ground_truth_wrenches(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    site_names: tuple[str, ...],
) -> dict[str, np.ndarray]:
    """Return per-site body cfrc_ext as [force(3), torque(3)]."""
    mujoco.mj_rnePostConstraint(model, data)
    wrenches: dict[str, np.ndarray] = {}
    for site_name in site_names:
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if site_id < 0:
            continue
        body_id = int(model.site_bodyid[site_id])
        raw_spatial = np.asarray(data.cfrc_ext[body_id], dtype=np.float32).reshape(-1)
        if raw_spatial.shape[0] >= 6:
            wrenches[site_name] = np.concatenate(
                [raw_spatial[3:6], raw_spatial[0:3]], axis=0
            ).astype(np.float32, copy=False)
        else:
            wrenches[site_name] = np.zeros(6, dtype=np.float32)
    return wrenches
