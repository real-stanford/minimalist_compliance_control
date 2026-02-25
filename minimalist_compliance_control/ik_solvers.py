"""IK solver utilities for compliance reference."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Callable, Sequence

import gin
import mink
import mujoco
import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation as R


def _filter_existing_geom_names(
    model: mujoco.MjModel, geom_names: Sequence[str]
) -> list[str]:
    out: list[str] = []
    for name in geom_names:
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if geom_id >= 0:
            out.append(name)
    return out


def _resolve_collision_pairs(
    model: mujoco.MjModel,
    raw_pairs: Sequence[tuple[Sequence[str], Sequence[str]]],
) -> list[tuple[list[str], list[str]]]:
    pairs: list[tuple[list[str], list[str]]] = []
    for lhs_raw, rhs_raw in raw_pairs:
        lhs = _filter_existing_geom_names(model, lhs_raw)
        rhs = _filter_existing_geom_names(model, rhs_raw)
        if lhs and rhs:
            pairs.append((lhs, rhs))
    return pairs


@gin.configurable
@dataclass
class IKConfig:
    ik_position_only: bool = False
    mink_num_iter: int = 1
    mink_damping: float = 0.1
    avoid_self_collision: bool = False
    collision_pairs: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = ()


class CollisionAvoidanceLimit(mink.CollisionAvoidanceLimit):
    """Compatibility collision limit matching old behavior.

    This keeps all geom pairs produced by Mink's name expansion and skips
    parent/weld/affinity filtering, matching the old implementation.
    """

    def _construct_geom_id_pairs(self, geom_pairs):
        geom_id_pairs = []
        for id_pair in self._collision_pairs_to_geom_id_pairs(geom_pairs):
            for geom_a, geom_b in itertools.product(*id_pair):
                geom_id_pairs.append((min(geom_a, geom_b), max(geom_a, geom_b)))
        return geom_id_pairs


@gin.configurable
class MinkIK:
    def __init__(
        self,
        model: mujoco.MjModel,
        site_names: Sequence[str],
        joint_indices: npt.NDArray[np.int32],
        joint_to_actuator_fn: Callable,
        ik_position_only: bool,
        source_q_start_idx: int,
        hand_orientation_cost_default: float = 10.0,
        hand_orientation_cost_position_only: float = 0.0,
        site_orientation_cost_overrides: dict[str, float] | None = None,
        enable_self_collision_avoidance: bool = False,
        ik_config: IKConfig | None = None,
    ) -> None:
        self.model = model
        self.site_names = list(site_names)
        self.site_name_to_idx = {name: idx for idx, name in enumerate(self.site_names)}
        self.joint_indices = np.asarray(joint_indices, dtype=np.int32)
        self.joint_to_actuator_fn = joint_to_actuator_fn
        self.ik_position_only = bool(ik_position_only)
        self.source_q_start_idx = int(source_q_start_idx)
        self.hand_orientation_cost_default = float(hand_orientation_cost_default)
        self.hand_orientation_cost_position_only = float(
            hand_orientation_cost_position_only
        )
        self.site_orientation_cost_overrides = {
            str(name): float(cost)
            for name, cost in (site_orientation_cost_overrides or {}).items()
        }
        self.ik_config = IKConfig() if ik_config is None else ik_config

        self.config = mink.Configuration(model)
        try:
            self.config.update_from_keyframe("home")
        except Exception:
            pass

        self.tasks = None
        self.limits = [mink.ConfigurationLimit(model)]
        if bool(enable_self_collision_avoidance):
            collision_pairs = _resolve_collision_pairs(
                model,
                self.ik_config.collision_pairs,
            )
            if collision_pairs:
                self.limits.append(
                    CollisionAvoidanceLimit(
                        model=model,
                        geom_pairs=collision_pairs,
                    )
                )

    def solve(
        self,
        data: mujoco.MjData,
        x_ref: npt.NDArray[np.float32],
        dt: float,
        num_iter: int,
        damping: float,
    ) -> npt.NDArray[np.float32]:
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
                orientation_cost = (
                    self.hand_orientation_cost_position_only
                    if self.ik_position_only
                    else self.hand_orientation_cost_default
                )
                orientation_cost = self.site_orientation_cost_overrides.get(
                    site_name, orientation_cost
                )
                frame_task = mink.FrameTask(
                    frame_name=site_name,
                    frame_type="site",
                    position_cost=10.0,
                    orientation_cost=orientation_cost,
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

        ik_dt = float(self.model.opt.timestep)
        if not np.isfinite(ik_dt) or ik_dt <= 0.0:
            ik_dt = float(dt)

        for _ in range(int(num_iter)):
            with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
                vel = mink.solve_ik(
                    self.config,
                    list(self.tasks.values()),
                    ik_dt,
                    solver="daqp",
                    damping=float(damping),
                    limits=self.limits,
                )
            self.config.integrate_inplace(vel, ik_dt)

        joint_pos = self.config.data.qpos[self.joint_indices].copy()
        return self.joint_to_actuator_fn(joint_pos)
