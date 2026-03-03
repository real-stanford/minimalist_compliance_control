"""Wrench estimation utilities independent of any sim backend."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import gin
import numpy as np
import numpy.typing as npt

AXIS_MAP: Dict[str, Tuple[int, float]] = {
    "+x": (0, 1.0),
    "-x": (0, -1.0),
    "+y": (1, 1.0),
    "-y": (1, -1.0),
    "+z": (2, 1.0),
    "-z": (2, -1.0),
}


@gin.configurable
@dataclass
class WrenchEstimateConfig:
    force_reg: float = 1e-3
    torque_reg: float = 1e-2
    force_only: bool = False
    axis_aligned: bool = False
    normal_axis: str | int = "+z"


def solve_axis_component(
    jacobian: npt.NDArray[np.float32],
    axis: npt.NDArray[np.float32],
    reg: float,
    tau_ext: npt.NDArray[np.float32],
) -> float:
    projected = (axis.reshape(1, 3) @ jacobian).reshape(-1)
    denom = float(projected @ projected + reg)
    if denom <= 1e-12:
        return 0.0
    return float(projected @ tau_ext) / denom


def solve_dense_component(
    jacobian: npt.NDArray[np.float32],
    tau_ext: npt.NDArray[np.float32],
    reg: float,
) -> npt.NDArray[np.float32]:
    jac = np.asarray(jacobian, dtype=np.float32)
    tau = np.asarray(tau_ext, dtype=np.float32)
    rows = jac.shape[0]
    identity = np.eye(rows, dtype=np.float32)
    mat = jac @ jac.T + reg * identity
    rhs = jac @ tau
    return np.linalg.solve(mat.astype(np.float32), rhs.astype(np.float32)).astype(
        np.float32
    )


def estimate_wrench(
    jacp: npt.NDArray[np.float32],
    jacr: npt.NDArray[np.float32],
    tau_ext: npt.NDArray[np.float32],
    site_rotmat: npt.NDArray[np.float32],
    config: WrenchEstimateConfig,
) -> npt.NDArray[np.float32]:
    if not config.axis_aligned:
        force_vec = solve_dense_component(jacp, tau_ext, config.force_reg)
        torque_vec = (
            np.zeros(3, dtype=np.float32)
            if config.force_only
            else solve_dense_component(jacr, tau_ext, config.torque_reg)
        )
    else:
        axis_spec = config.normal_axis
        if isinstance(axis_spec, int):
            axis_idx, sign = axis_spec, 1.0
        else:
            axis_idx, sign = AXIS_MAP.get(axis_spec, (2, 1.0))
        normal_vec = sign * site_rotmat[:, axis_idx]
        tangent_axes = [axis for axis in (0, 1, 2) if axis != axis_idx]
        tangent_basis = site_rotmat[:, tangent_axes]
        force_vec = solve_axis_component(jacp, normal_vec, config.force_reg, tau_ext)
        force_vec = force_vec * normal_vec
        torque_vec = np.zeros(3, dtype=np.float32)
        if not config.force_only:
            torque_vec = sum(
                solve_axis_component(
                    jacr, tangent_basis[:, idx], config.torque_reg, tau_ext
                )
                * tangent_basis[:, idx]
                for idx in range(tangent_basis.shape[1])
            )
    return np.concatenate([force_vec, torque_vec], axis=0).astype(np.float32)
