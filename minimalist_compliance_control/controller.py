"""Unified minimal compliance pipeline controller."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import numpy as np
import numpy.typing as npt
import gin
import mujoco

from minimalist_compliance_control.wrench_estimation import (
    WrenchEstimateConfig,
    estimate_wrench,
)
from minimalist_compliance_control.wrench_sim import WrenchSim, WrenchSimConfig
from minimalist_compliance_control.reference.compliance_ref import (
    COMMAND_LAYOUT,
    ComplianceReference,
)


@dataclass
class ComplianceInputs:
    """Inputs at each control step (pre-compliance state)."""

    motor_torques: npt.NDArray[np.float32]
    joint_pos: Optional[npt.NDArray[np.float32]] = None
    qpos: Optional[npt.NDArray[np.float32]] = None
    base_pos: Optional[npt.NDArray[np.float32]] = None
    base_quat: Optional[npt.NDArray[np.float32]] = None
    time: float = 0.0
    command_matrix: Optional[npt.NDArray[np.float32]] = None


@dataclass
class ComplianceRefOutput:
    """Output from a compliance reference (e.g., command matrix)."""

    command_matrix: npt.NDArray[np.float32]
    last_state: Optional[dict] = None


@gin.configurable
@dataclass
class ControllerConfig:
    """Configuration for the controller pipeline."""

    xml_path: Optional[str] = None
    site_names: Optional[Sequence[str]] = None
    fixed_base: Optional[bool] = None
    base_body_name: Optional[str] = None
    joint_indices_by_site: Optional[Dict[str, npt.NDArray[np.int32]]] = None
    motor_indices_by_site: Optional[Dict[str, npt.NDArray[np.int32]]] = None
    gear_ratios_by_site: Optional[Dict[str, npt.NDArray[np.float32]]] = None


@gin.configurable
@dataclass
class ComplianceRefConfig:
    dt: float = 0.02
    ik_position_only: bool = False
    mass: float = 1.0
    inertia_diag: Sequence[float] = (1.0, 1.0, 1.0)
    mink_num_iter: int = 5
    mink_damping: float = 1e-2
    q_start_idx: int = 0
    qd_start_idx: int = 0
    actuator_indices: Optional[Sequence[int]] = None
    joint_indices: Optional[Sequence[int]] = None
    default_motor_pos: Optional[Sequence[float]] = None
    default_qpos: Optional[Sequence[float]] = None
    joint_to_actuator_scale: Optional[Sequence[float]] = None
    joint_to_actuator_bias: Optional[Sequence[float]] = None
    fixed_model_xml_path: Optional[str] = None


@gin.configurable
class ComplianceController:
    """Orchestrates wrench sim + wrench estimation + optional compliance reference."""

    def __init__(
        self,
        gin_path: Optional[str] = None,
        config: Optional[ControllerConfig] = None,
        estimate_config: Optional[WrenchEstimateConfig] = None,
        ref_config: Optional[ComplianceRefConfig] = None,
    ) -> None:
        if gin_path is not None:
            if any(arg is not None for arg in (config, estimate_config, ref_config)):
                raise ValueError(
                    "Provide only gin_path or explicit config objects, not both."
                )
            gin.clear_config()
            gin.parse_config_file(gin_path)
            config = ControllerConfig()
            estimate_config = WrenchEstimateConfig()
            ref_config = ComplianceRefConfig()

        if config is None:
            raise ValueError("Either gin_path or config must be provided.")

        if (
            config.xml_path is None
            or config.site_names is None
            or config.fixed_base is None
        ):
            sim_cfg = WrenchSimConfig()
            if config.xml_path is None:
                config.xml_path = sim_cfg.xml_path
            if config.site_names is None:
                config.site_names = sim_cfg.site_names
            if config.fixed_base is None:
                config.fixed_base = sim_cfg.fixed_base

        self.config = config
        self.estimate_config = estimate_config or WrenchEstimateConfig()
        if self.config.fixed_base and self.config.base_body_name:
            raise ValueError("base_body_name must be empty when fixed_base is True.")
        self.wrench_sim = WrenchSim(
            WrenchSimConfig(
                xml_path=config.xml_path,
                site_names=config.site_names,
                fixed_base=config.fixed_base,
            )
        )
        self.ref_config = ref_config or ComplianceRefConfig()
        self.compliance_ref: Optional[ComplianceReference] = None
        self._last_state: Optional[dict] = None
        self._build_compliance_ref()

    @classmethod
    def from_gin(cls, gin_path: str) -> "ComplianceController":
        """Build controller from a single gin config file path."""
        return cls(gin_path=gin_path)

    def _build_compliance_ref(self) -> None:
        cfg = self.ref_config
        model = self.wrench_sim.model
        data = self.wrench_sim.data
        site_names = self.config.site_names

        trnid = np.asarray(model.actuator_trnid, dtype=np.int32)
        valid = trnid[:, 0] >= 0

        if cfg.actuator_indices is None:
            actuator_indices = np.flatnonzero(valid).astype(np.int32)
        else:
            actuator_indices = np.asarray(cfg.actuator_indices, dtype=np.int32)

        if cfg.joint_indices is None:
            if self.config.joint_indices_by_site:
                seen = set()
                ordered = []
                for site in site_names:
                    for idx in self.config.joint_indices_by_site[site]:
                        if idx in seen:
                            continue
                        seen.add(int(idx))
                        ordered.append(int(idx))
                qd_offset = int(cfg.qd_start_idx)
                if qd_offset:
                    ordered = [idx - qd_offset for idx in ordered]
                joint_indices = np.asarray(ordered, dtype=np.int32)
            else:
                joint_indices = trnid[valid, 0].astype(np.int32)
        else:
            joint_indices = np.asarray(cfg.joint_indices, dtype=np.int32)

        scale = (
            np.asarray(cfg.joint_to_actuator_scale, dtype=np.float32)
            if cfg.joint_to_actuator_scale is not None
            else None
        )
        bias = (
            np.asarray(cfg.joint_to_actuator_bias, dtype=np.float32)
            if cfg.joint_to_actuator_bias is not None
            else None
        )

        def joint_to_actuator_fn(
            joint_pos: npt.NDArray[np.float32],
        ) -> npt.NDArray[np.float32]:
            out = np.asarray(joint_pos, dtype=np.float32)
            if scale is not None:
                out = out * scale
            if bias is not None:
                out = out + bias
            return out

        default_qpos = (
            np.asarray(cfg.default_qpos, dtype=np.float32)
            if cfg.default_qpos is not None
            else np.asarray(data.qpos, dtype=np.float32).copy()
        )
        default_motor_pos = (
            np.asarray(cfg.default_motor_pos, dtype=np.float32)
            if cfg.default_motor_pos is not None
            else np.zeros(model.nu, dtype=np.float32)
        )

        self.compliance_ref = ComplianceReference(
            dt=cfg.dt,
            model=model,
            data=data,
            site_names=site_names,
            actuator_indices=actuator_indices,
            joint_indices=joint_indices,
            joint_to_actuator_fn=joint_to_actuator_fn,
            default_motor_pos=default_motor_pos,
            default_qpos=default_qpos,
            fixed_model_xml_path=cfg.fixed_model_xml_path,
            q_start_idx=cfg.q_start_idx,
            qd_start_idx=cfg.qd_start_idx,
            ik_position_only=cfg.ik_position_only,
            mass=cfg.mass,
            inertia_diag=cfg.inertia_diag,
            mink_num_iter=cfg.mink_num_iter,
            mink_damping=cfg.mink_damping,
        )
        self._last_state = self.compliance_ref.get_default_state()
        self._default_motor_pos = default_motor_pos
        if default_qpos is not None and default_qpos.size == model.nq:
            self.wrench_sim.set_qpos(default_qpos)
            self.wrench_sim.forward()

        self._joint_dof_union: Optional[npt.NDArray[np.int32]] = None
        if self.config.joint_indices_by_site:
            union = np.unique(
                np.concatenate(
                    [
                        np.asarray(v, dtype=np.int32).reshape(-1)
                        for v in self.config.joint_indices_by_site.values()
                    ]
                )
            )
            self._joint_dof_union = union

    def step(
        self,
        inputs: ComplianceInputs,
        use_estimated_wrench: bool = False,
    ) -> Dict[str, npt.NDArray[np.float32] | dict]:
        """Run one loop and return estimated wrenches (and optionally state_ref)."""
        if inputs.qpos is not None:
            self.wrench_sim.set_qpos(inputs.qpos)
        elif inputs.joint_pos is not None:
            joint_pos = np.asarray(inputs.joint_pos, dtype=np.float32).reshape(-1)
            if joint_pos.size in (self.wrench_sim.model.nq, self.wrench_sim.model.njnt):
                self.wrench_sim.set_joint_positions(joint_pos)
            elif (
                self._joint_dof_union is not None
                and joint_pos.size == self._joint_dof_union.size
            ):
                self.wrench_sim.set_dof_positions(self._joint_dof_union, joint_pos)
            else:
                raise ValueError(
                    "joint_pos length does not match model.nq/model.njnt or joint_indices_by_site union."
                )
        else:
            motor_pos = (
                np.asarray(inputs.motor_pos, dtype=np.float32)
                if inputs.motor_pos is not None
                else np.asarray(self._default_motor_pos, dtype=np.float32)
            )
            self.wrench_sim.set_motor_angles(motor_pos)
        self.wrench_sim.forward()
        if getattr(self.wrench_sim.config, "view", False):
            self.wrench_sim.visualize()
        if getattr(self.wrench_sim.config, "render", False):
            self.wrench_sim.record_frame()

        base_pos = None
        base_quat = None
        if self.config.base_body_name:
            base_body_id = mujoco.mj_name2id(
                self.wrench_sim.model,
                mujoco.mjtObj.mjOBJ_BODY,
                self.config.base_body_name,
            )
            if base_body_id >= 0:
                base_pos = np.asarray(
                    self.wrench_sim.data.xpos[base_body_id], dtype=np.float32
                )
                base_quat = np.asarray(
                    self.wrench_sim.data.xquat[base_body_id], dtype=np.float32
                )
                # print(
                #     f"[MinimumCompliance] {self.config.base_body_name} pos: {base_pos}"
                # )

        wrenches: Dict[str, npt.NDArray[np.float32]] = {}
        bias = self.wrench_sim.bias_torque()
        for site in self.config.site_names:
            jacp, jacr = self.wrench_sim.site_jacobian(site)
            if self.config.joint_indices_by_site is None:
                joint_idx = np.arange(self.wrench_sim.model.nv, dtype=np.int32)
            else:
                joint_idx = np.asarray(
                    self.config.joint_indices_by_site[site], dtype=np.int32
                )

            if self.config.motor_indices_by_site is None:
                tau_raw = np.asarray(inputs.motor_torques, dtype=np.float32)
            else:
                motor_idx = np.asarray(
                    self.config.motor_indices_by_site[site], dtype=np.int32
                )
                gear = (
                    self.config.gear_ratios_by_site.get(site)
                    if self.config.gear_ratios_by_site is not None
                    else None
                )
                if gear is None:
                    tau_raw = np.asarray(inputs.motor_torques, dtype=np.float32)[
                        motor_idx
                    ]
                else:
                    gear = np.asarray(gear, dtype=np.float32)
                    tau_raw = (
                        np.asarray(inputs.motor_torques, dtype=np.float32)[motor_idx]
                        * gear
                    )

            tau_bias = bias[joint_idx]
            tau_ext = -(tau_raw - tau_bias)

            site_rot = self.wrench_sim.data.site_xmat[
                self.wrench_sim.site_ids[site]
            ].reshape(3, 3)
            wrench = estimate_wrench(
                jacp[:, joint_idx],
                jacr[:, joint_idx],
                tau_ext,
                site_rot,
                self.estimate_config,
            )
            wrenches[site] = wrench

        result: Dict[str, npt.NDArray[np.float32] | dict] = {"wrenches": wrenches}
        command_matrix = inputs.command_matrix
        if use_estimated_wrench and command_matrix is not None:
            command_matrix = np.asarray(command_matrix, dtype=np.float32).copy()
            for idx, site in enumerate(self.config.site_names):
                wrench = wrenches.get(site)
                if wrench is None:
                    continue
                command_matrix[idx, COMMAND_LAYOUT.measured_force] = wrench[:3]
                command_matrix[idx, COMMAND_LAYOUT.measured_torque] = wrench[3:6]

        if self.compliance_ref is not None and command_matrix is not None:
            if self._last_state is None:
                self._last_state = self.compliance_ref.get_default_state()
            state_ref = self.compliance_ref.get_state_ref(
                time_curr=float(inputs.time),
                command_matrix=command_matrix,
                last_state=self._last_state,
                model=self.wrench_sim.model,
                data=self.wrench_sim.data,
                base_pos=base_pos,
                base_quat=base_quat,
            )
            self._last_state = state_ref
            result["state_ref"] = state_ref
        return result
