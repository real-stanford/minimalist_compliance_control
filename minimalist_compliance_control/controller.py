"""Unified minimal compliance pipeline controller."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import gin
import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation as R

from minimalist_compliance_control.compliance_ref import (
    COMMAND_LAYOUT,
    ComplianceReference,
    ComplianceState,
)
from minimalist_compliance_control.wrench_estimation import (
    WrenchEstimateConfig,
    estimate_wrench,
)
from minimalist_compliance_control.wrench_sim import WrenchSim, WrenchSimConfig


@gin.configurable
@dataclass
class ControllerConfig:
    """Configuration for the controller pipeline."""

    xml_path: Optional[str] = None
    site_names: Optional[Sequence[str]] = None
    fixed_base: Optional[bool] = None
    prep_duration: float = 2.0
    base_body_name: Optional[str] = None
    joint_indices_by_site: Optional[Dict[str, npt.NDArray[np.int32]]] = None
    motor_indices_by_site: Optional[Dict[str, npt.NDArray[np.int32]]] = None
    gear_ratios_by_site: Optional[Dict[str, npt.NDArray[np.float32]]] = None
    motor_torque_ema_alpha: float = 0.1


@gin.configurable
@dataclass
class RefConfig:
    dt: Optional[float] = None
    ik_position_only: Optional[bool] = None
    mass: Optional[float] = None
    inertia_diag: Optional[Sequence[float]] = None
    mink_num_iter: Optional[int] = None
    mink_damping: Optional[float] = None
    q_start_idx: Optional[int] = None
    qd_start_idx: Optional[int] = None
    actuator_indices: Optional[Sequence[int]] = None
    joint_indices: Optional[Sequence[int]] = None
    default_motor_pos: Optional[Sequence[float]] = None
    default_qpos: Optional[Sequence[float]] = None
    joint_to_actuator_scale: Optional[Sequence[float]] = None
    joint_to_actuator_bias: Optional[Sequence[float]] = None
    fixed_model_xml_path: Optional[str] = None
    avoid_self_collision: Optional[bool] = None


@gin.configurable
class ComplianceController:
    """Orchestrates wrench sim + wrench estimation + optional compliance reference."""

    def __init__(
        self,
        gin_path: Optional[str] = None,
        config: Optional[ControllerConfig] = None,
        estimate_config: Optional[WrenchEstimateConfig] = None,
        ref_config: Optional[RefConfig] = None,
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
            ref_config = RefConfig()

        if config is None:
            raise ValueError("Either gin_path or config must be provided.")

        missing_cfg = []
        if config.xml_path is None:
            missing_cfg.append("ControllerConfig.xml_path")
        if config.site_names is None:
            missing_cfg.append("ControllerConfig.site_names")
        if config.fixed_base is None:
            missing_cfg.append("ControllerConfig.fixed_base")
        if missing_cfg:
            raise ValueError(
                "Missing required controller configuration: " + ", ".join(missing_cfg)
            )

        self.config = config
        self.estimate_config = estimate_config or WrenchEstimateConfig()
        self._motor_torque_ema_alpha = float(self.config.motor_torque_ema_alpha)
        if not (0.0 < self._motor_torque_ema_alpha <= 1.0):
            raise ValueError(
                "ControllerConfig.motor_torque_ema_alpha must be in (0, 1]."
            )
        self._motor_torque_ema: Optional[npt.NDArray[np.float32]] = None
        if self.config.fixed_base and self.config.base_body_name:
            raise ValueError("base_body_name must be empty when fixed_base is True.")
        self.wrench_sim = WrenchSim(
            WrenchSimConfig(
                xml_path=str(config.xml_path),
                site_names=tuple(config.site_names),
                fixed_base=bool(config.fixed_base),
            )
        )
        self.ref_config = ref_config or RefConfig()
        self.compliance_ref: Optional[ComplianceReference] = None
        self._last_state: Optional[ComplianceState] = None
        self._build_compliance_ref()

    @classmethod
    def from_gin(cls, gin_path: str) -> "ComplianceController":
        """Build controller from a single gin config file path."""
        return cls(gin_path=gin_path)

    @property
    def site_ids(self) -> Dict[str, int]:
        return {
            str(name): int(self.wrench_sim.site_ids[name])
            for name in self.config.site_names
        }

    def get_x_obs(self) -> npt.NDArray[np.float32]:
        num_sites = len(self.config.site_names)
        x_obs = np.zeros((num_sites, 6), dtype=np.float32)
        for idx, site_name in enumerate(self.config.site_names):
            site_id = int(self.wrench_sim.site_ids[site_name])
            x_obs[idx, :3] = np.asarray(
                self.wrench_sim.data.site_xpos[site_id], dtype=np.float32
            )
            rotmat = np.asarray(
                self.wrench_sim.data.site_xmat[site_id], dtype=np.float32
            ).reshape(3, 3)
            x_obs[idx, 3:6] = R.from_matrix(rotmat).as_rotvec().astype(np.float32)
        return x_obs

    def sync_qpos(self, qpos: npt.NDArray[np.float32]) -> None:
        self.wrench_sim.set_qpos(np.asarray(qpos, dtype=np.float32))
        self.wrench_sim.forward()

    def _build_compliance_ref(self) -> None:
        cfg = self.ref_config
        missing_ref_cfg = []
        if cfg.dt is None:
            missing_ref_cfg.append("RefConfig.dt")
        if cfg.ik_position_only is None:
            missing_ref_cfg.append("RefConfig.ik_position_only")
        if cfg.mass is None:
            missing_ref_cfg.append("RefConfig.mass")
        if cfg.inertia_diag is None:
            missing_ref_cfg.append("RefConfig.inertia_diag")
        if cfg.mink_num_iter is None:
            missing_ref_cfg.append("RefConfig.mink_num_iter")
        if cfg.mink_damping is None:
            missing_ref_cfg.append("RefConfig.mink_damping")
        if cfg.q_start_idx is None:
            missing_ref_cfg.append("RefConfig.q_start_idx")
        if cfg.qd_start_idx is None:
            missing_ref_cfg.append("RefConfig.qd_start_idx")
        if missing_ref_cfg:
            raise ValueError(
                "Missing required compliance reference configuration: "
                + ", ".join(missing_ref_cfg)
            )

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

        def actuator_to_joint_fn(
            actuator_pos: npt.NDArray[np.float32],
        ) -> npt.NDArray[np.float32]:
            out = np.asarray(actuator_pos, dtype=np.float32)
            if bias is not None:
                out = out - bias
            if scale is not None:
                if np.any(np.abs(scale) < 1e-8):
                    raise ValueError(
                        "Cannot invert joint_to_actuator mapping with near-zero scale."
                    )
                out = out / scale
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
            dt=float(cfg.dt),
            model=model,
            data=data,
            site_names=site_names,
            actuator_indices=actuator_indices,
            joint_indices=joint_indices,
            joint_to_actuator_fn=joint_to_actuator_fn,
            actuator_to_joint_fn=actuator_to_joint_fn,
            default_motor_pos=default_motor_pos,
            default_qpos=default_qpos,
            fixed_model_xml_path=cfg.fixed_model_xml_path,
            q_start_idx=int(cfg.q_start_idx),
            qd_start_idx=int(cfg.qd_start_idx),
            ik_position_only=bool(cfg.ik_position_only),
            mass=float(cfg.mass),
            inertia_diag=np.asarray(cfg.inertia_diag, dtype=np.float32),
            mink_num_iter=int(cfg.mink_num_iter),
            mink_damping=float(cfg.mink_damping),
            avoid_self_collision=bool(cfg.avoid_self_collision),
        )
        self._last_state = self.compliance_ref.get_default_state()
        if default_qpos is not None and default_qpos.size == model.nq:
            self.wrench_sim.set_qpos(default_qpos)
            self.wrench_sim.forward()

    def _smooth_motor_torques(
        self, motor_torques: npt.NDArray[np.float32]
    ) -> npt.NDArray[np.float32]:
        motor_torques_arr = np.asarray(motor_torques, dtype=np.float32).reshape(-1)
        if motor_torques_arr.shape[0] != int(self.wrench_sim.model.nu):
            raise ValueError(
                f"motor_torques length {motor_torques_arr.shape[0]} "
                f"!= model.nu {self.wrench_sim.model.nu}."
            )
        if (
            self._motor_torque_ema is None
            or self._motor_torque_ema.shape != motor_torques_arr.shape
        ):
            self._motor_torque_ema = motor_torques_arr.copy()
        else:
            alpha = self._motor_torque_ema_alpha
            self._motor_torque_ema = (
                alpha * motor_torques_arr + (1.0 - alpha) * self._motor_torque_ema
            ).astype(np.float32)
        return self._motor_torque_ema.copy()

    def step(
        self,
        command_matrix: npt.NDArray[np.float32],
        motor_torques: npt.NDArray[np.float32],
        qpos: npt.NDArray[np.float32],
        base_pos: Optional[npt.NDArray[np.float32]] = None,
        base_quat: Optional[npt.NDArray[np.float32]] = None,
        use_estimated_wrench: bool = True,
    ) -> tuple[Dict[str, npt.NDArray[np.float32]], Optional[ComplianceState]]:
        """Run one loop and return estimated wrenches and optional compliance state."""
        command_matrix = np.asarray(command_matrix, dtype=np.float32).copy()
        self.sync_qpos(qpos)

        base_pos_est = (
            np.asarray(base_pos, dtype=np.float32).copy()
            if base_pos is not None
            else None
        )
        base_quat_est = (
            np.asarray(base_quat, dtype=np.float32).copy()
            if base_quat is not None
            else None
        )
        if self.config.base_body_name:
            import mujoco

            base_body_id = mujoco.mj_name2id(
                self.wrench_sim.model,
                mujoco.mjtObj.mjOBJ_BODY,
                self.config.base_body_name,
            )
            if base_body_id >= 0:
                base_pos_est = np.asarray(
                    self.wrench_sim.data.xpos[base_body_id], dtype=np.float32
                )
                base_quat_est = np.asarray(
                    self.wrench_sim.data.xquat[base_body_id], dtype=np.float32
                )
                # print(
                #     f"[MinimumCompliance] {self.config.base_body_name} pos: {base_pos}"
                # )

        wrenches: Dict[str, npt.NDArray[np.float32]] = {}
        motor_torques_arr = self._smooth_motor_torques(motor_torques)
        bias = self.wrench_sim.bias_torque()
        actuator_trnid = np.asarray(
            self.wrench_sim.model.actuator_trnid[:, 0], dtype=np.int32
        )
        valid_act = actuator_trnid >= 0
        default_motor_idx = np.flatnonzero(valid_act).astype(np.int32)
        default_joint_idx = np.asarray(
            self.wrench_sim.model.jnt_dofadr[actuator_trnid[valid_act]], dtype=np.int32
        )
        for site in self.config.site_names:
            jacp, jacr = self.wrench_sim.site_jacobian(site)
            if self.config.motor_indices_by_site is None:
                motor_idx = default_motor_idx
            else:
                motor_idx = np.asarray(
                    self.config.motor_indices_by_site[site], dtype=np.int32
                )
            if self.config.joint_indices_by_site is None:
                if self.config.motor_indices_by_site is None:
                    joint_idx = default_joint_idx
                else:
                    trnid_sel = np.asarray(actuator_trnid[motor_idx], dtype=np.int32)
                    if np.any(trnid_sel < 0):
                        raise ValueError(
                            f"Actuator(s) without valid joint mapping in motor_indices_by_site[{site!r}]."
                        )
                    joint_idx = np.asarray(
                        self.wrench_sim.model.jnt_dofadr[trnid_sel], dtype=np.int32
                    )
            else:
                joint_idx = np.asarray(
                    self.config.joint_indices_by_site[site], dtype=np.int32
                )

            if self.config.motor_indices_by_site is None:
                tau_raw = motor_torques_arr[motor_idx]
            else:
                gear = (
                    self.config.gear_ratios_by_site.get(site)
                    if self.config.gear_ratios_by_site is not None
                    else None
                )
                if gear is None:
                    tau_raw = motor_torques_arr[motor_idx]
                else:
                    gear = np.asarray(gear, dtype=np.float32)
                    if gear.shape[0] != motor_idx.shape[0]:
                        raise ValueError(
                            f"gear_ratios_by_site[{site!r}] length {gear.shape[0]} "
                            f"!= motor_indices length {motor_idx.shape[0]}."
                        )
                    tau_raw = motor_torques_arr[motor_idx] * gear

            tau_bias = bias[joint_idx]
            if tau_raw.shape[0] != tau_bias.shape[0]:
                raise ValueError(
                    f"Shape mismatch at site {site!r}: tau_raw {tau_raw.shape} vs tau_bias {tau_bias.shape}. "
                    "Check motor_indices_by_site / joint_indices_by_site alignment."
                )
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

        state_ref: Optional[ComplianceState] = None
        if use_estimated_wrench:
            for idx, site in enumerate(self.config.site_names):
                wrench = wrenches.get(site)
                if wrench is None:
                    continue
                command_matrix[idx, COMMAND_LAYOUT.measured_force] = wrench[:3]
                command_matrix[idx, COMMAND_LAYOUT.measured_torque] = wrench[3:6]

        if self.compliance_ref is not None:
            if self._last_state is None:
                self._last_state = self.compliance_ref.get_default_state()
            state_ref = self.compliance_ref.get_state_ref(
                command_matrix=command_matrix,
                last_state=self._last_state,
                model=self.wrench_sim.model,
                data=self.wrench_sim.data,
                base_pos=base_pos_est,
                base_quat=base_quat_est,
            )
            self._last_state = state_ref
        return wrenches, state_ref

    def close(self) -> None:
        self.wrench_sim.close()
