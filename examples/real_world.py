"""Real-world backend for compliance control.

This backend mirrors the minimal ``BaseSim`` surface used by the runners:
- ``step()`` to write commands and refresh hardware state,
- ``sync()`` to keep the main loop alive,
- ``close()`` to release resources.
"""

from __future__ import annotations

import glob
import importlib
import importlib.util
import os
import platform
import warnings
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Sequence

import mujoco
import numpy as np
import yaml

from minimalist_compliance_control.utils import deep_update
from minimalist_compliance_control.wrench_sim import WrenchSim


class RealWorld:
    """Dynamixel/ARX-backed real-world control backend."""

    def __init__(
        self,
        wrench_sim: WrenchSim,
        control_dt: float,
        default_config_path: str,
        robot_config_path: str,
        motors_config_path: str | None = None,
        port_pattern: str | None = None,
        baudrate: int = 2_000_000,
        return_delay: int = 1,
        use_imu: bool = False,
        imu_input_freq: float = 200.0,
        imu_output_freq: float = 200.0,
    ) -> None:
        self.wrench_sim = wrench_sim
        self.model = wrench_sim.model
        self.data = wrench_sim.data
        self.name = "real"
        self.control_dt = float(control_dt)
        if self.control_dt <= 0.0:
            raise ValueError("control_dt must be > 0.")

        self._robot_hint = os.path.basename(robot_config_path).lower()
        self._is_arx = "arx" in self._robot_hint
        self._controller = self._load_controller_module(is_arx=self._is_arx)

        (
            self._kp,
            self._kd,
            self._ki,
            self._zero_pos,
            self.motor_control_mode,
            self._kt,
            self._kv,
            self._r_winding,
        ) = self._load_motor_hw_params(
            default_config_path=default_config_path,
            robot_config_path=robot_config_path,
            motors_config_path=motors_config_path,
        )
        if len(self._kp) != int(self.model.nu):
            raise ValueError(
                f"Hardware motor parameter length {len(self._kp)} != model.nu {self.model.nu}."
            )

        (
            self._motor_to_qpos_mat,
            self._motor_to_qpos_bias,
            self._qpos_motor_mask,
            self._motor_to_qvel_mat,
            self._qvel_motor_mask,
        ) = self._build_motor_joint_maps()
        self._port_pattern = (
            port_pattern if port_pattern is not None else self._default_port_pattern()
        )
        self.controllers = self._controller.create_controllers(
            self._port_pattern,
            self._kp,
            self._kd,
            self._ki,
            self._zero_pos,
            self.motor_control_mode,
            int(baudrate),
            int(return_delay),
        )
        if len(self.controllers) == 0:
            raise RuntimeError(
                f"No motor controllers found for pattern '{self._port_pattern}'."
            )
        self._controller.initialize(self.controllers)

        motor_ids = self._controller.get_motor_ids(self.controllers)
        self._controller_lengths = [len(motor_ids[key]) for key in sorted(motor_ids)]
        self._controller_split_idx = np.cumsum(self._controller_lengths)[:-1]
        motor_ids_flat = np.array(
            sum((motor_ids[key] for key in sorted(motor_ids)), []),
            dtype=np.int32,
        )
        self._sort_idx = np.argsort(motor_ids_flat)
        self._unsort_idx = np.argsort(self._sort_idx)
        self.refresh_control_modes()

        self._motor_cur_limits: np.ndarray | None = None
        if hasattr(self._controller, "get_motor_current_limits"):
            try:
                cur_limits = self._controller.get_motor_current_limits(self.controllers)
                limits_all: list[float] = []
                for key in sorted(motor_ids):
                    limits_all.extend(cur_limits.get(key, []))
                if limits_all:
                    self._motor_cur_limits = np.asarray(limits_all, dtype=np.float32)[
                        self._sort_idx
                    ]
            except Exception:
                self._motor_cur_limits = None

        self.imu = None
        if use_imu:
            try:
                ThreadedIMU = self._load_threaded_imu_class()
                self.imu = ThreadedIMU(
                    input_freq=float(imu_input_freq),
                    output_freq=float(imu_output_freq),
                )
                self.imu.start()
            except Exception as exc:
                warnings.warn(f"IMU initialization failed: {exc}", RuntimeWarning)
                self.imu = None

        self._refresh_state(retries=1)

    def _default_port_pattern(self) -> str:
        if self._is_arx:
            return os.environ.get("ARX5_INTERFACE", "can[0-9]+")
        if "leap" in self._robot_hint:
            if platform.system() == "Darwin":
                return r"cu\.usbserial-.*"
            return r"ttyUSB[0-9]+"
        if platform.system() == "Darwin":
            return r"cu\.usbserial-.*"
        return r"tty(?:CH9344)?USB[0-9]+"

    def _load_controller_module(self, is_arx: bool):
        if is_arx:
            try:
                return importlib.import_module("real_world.dynamixel.arx5_controller")
            except Exception:
                arx_path = (
                    Path(__file__).resolve().parents[1]
                    / "real_world"
                    / "dynamixel"
                    / "arx5_controller.py"
                )
                spec = importlib.util.spec_from_file_location(
                    "arx5_controller", str(arx_path)
                )
                if spec is None or spec.loader is None:
                    raise RuntimeError(f"Failed loading ARX controller from {arx_path}")
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                return module

        try:
            return importlib.import_module("dynamixel_cpp")
        except ModuleNotFoundError:
            pass

        dyn_path = os.environ.get("MCC_DYNAMIXEL_CPP_PATH", "").strip()
        if dyn_path == "":
            candidates = glob.glob(
                str(
                    Path(__file__).resolve().parents[1]
                    / "real_world"
                    / "dynamixel"
                    / "dynamixel_cpp*.so"
                )
            )
            if not candidates:
                tb_root = os.environ.get(
                    "MCC_TODDLERBOT_INTERNAL_PATH",
                    "/Users/haochen/Projects/toddlerbot_internal",
                )
                candidates = glob.glob(
                    os.path.join(
                        tb_root, "toddlerbot", "actuation", "dynamixel_cpp*.so"
                    )
                )
            if not candidates:
                raise RuntimeError(
                    "Cannot find dynamixel_cpp extension. Set MCC_DYNAMIXEL_CPP_PATH "
                    "or install with BUILD_DYNAMIXEL=ON."
                )
            dyn_path = sorted(candidates)[0]

        spec = importlib.util.spec_from_file_location("dynamixel_cpp", dyn_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Failed loading dynamixel_cpp module from {dyn_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def _load_threaded_imu_class(self):
        try:
            mod = importlib.import_module("real_world.IMU")
            return getattr(mod, "ThreadedIMU")
        except Exception:
            imu_path = Path(__file__).resolve().parents[1] / "real_world" / "IMU.py"
            spec = importlib.util.spec_from_file_location(
                "real_world_imu", str(imu_path)
            )
            if spec is None or spec.loader is None:
                raise RuntimeError(f"Failed loading IMU module from {imu_path}")
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return getattr(mod, "ThreadedIMU")

    def _load_motor_hw_params(
        self,
        default_config_path: str,
        robot_config_path: str,
        motors_config_path: str | None,
    ) -> tuple[
        list[float],
        list[float],
        list[float],
        list[float],
        list[str],
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        with open(default_config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        with open(robot_config_path, "r", encoding="utf-8") as f:
            robot_cfg = yaml.safe_load(f)
        if robot_cfg is not None:
            deep_update(config, robot_cfg)
        if motors_config_path:
            with open(motors_config_path, "r", encoding="utf-8") as f:
                motors_cfg = yaml.safe_load(f)
            if motors_cfg is not None:
                deep_update(config, motors_cfg)

        motors_cfg = config.get("motors", {})
        actuator_cfg = config.get("actuators", {})

        kp: list[float] = []
        kd: list[float] = []
        ki: list[float] = []
        zero_pos: list[float] = []
        control_mode: list[str] = []
        kt: list[float] = []
        kv: list[float] = []
        r_winding: list[float] = []

        for i in range(self.model.nu):
            act_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            if act_name is None:
                raise ValueError(f"Actuator index {i} has no name.")
            motor_key = act_name
            if motor_key not in motors_cfg and motor_key.endswith("_act"):
                base_key = motor_key[: -len("_act")]
                if base_key in motors_cfg:
                    motor_key = base_key
            if motor_key not in motors_cfg:
                raise ValueError(f"Missing motor config for actuator '{act_name}'.")
            m_cfg = motors_cfg[motor_key]
            motor_type = str(m_cfg.get("motor", ""))
            a_cfg = actuator_cfg.get(motor_type, {})

            kp.append(float(m_cfg.get("kp", 0.0)))
            kd.append(float(m_cfg.get("kd", 0.0)))
            ki.append(float(m_cfg.get("ki", 0.0)))
            zero_pos.append(float(m_cfg.get("zero_pos", 0.0)))
            control_mode.append(str(m_cfg.get("control_mode", "extended_position")))
            kt.append(float(a_cfg.get("kt", 1.0)))
            kv.append(float(a_cfg.get("kv", 0.0)))
            r_winding.append(float(a_cfg.get("r_winding", 0.0)))

        return (
            kp,
            kd,
            ki,
            zero_pos,
            control_mode,
            np.asarray(kt, dtype=np.float32),
            np.asarray(kv, dtype=np.float32),
            np.asarray(r_winding, dtype=np.float32),
        )

    def _resolve_model_xml_path(self) -> Path | None:
        xml_path_raw = Path(str(self.wrench_sim.config.xml_path))
        candidates: list[Path] = []
        if xml_path_raw.is_absolute():
            candidates.append(xml_path_raw)
        else:
            candidates.append(Path.cwd() / xml_path_raw)
            candidates.append(Path(__file__).resolve().parents[1] / xml_path_raw)
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        return None

    def _build_motor_joint_maps(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # Map scalar joint states (qpos/qvel) from actuator motor angles.
        nu = int(self.model.nu)
        row_by_joint: dict[str, np.ndarray] = {}
        bias_by_joint: dict[str, float] = {}

        trnid = np.asarray(self.model.actuator_trnid[:, 0], dtype=np.int32)
        for act_idx, joint_id in enumerate(trnid):
            if joint_id < 0:
                continue
            joint_name = mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, int(joint_id)
            )
            if joint_name is None:
                continue
            row = np.zeros(nu, dtype=np.float32)
            row[act_idx] = 1.0
            row_by_joint[joint_name] = row
            bias_by_joint[joint_name] = 0.0

        eq_rules: list[tuple[str, str, float, float]] = []
        tendon_rules: list[list[tuple[str, float]]] = []
        xml_path = self._resolve_model_xml_path()
        if xml_path is not None:
            try:
                root = ET.parse(xml_path).getroot()
                eq_elem = root.find("equality")
                if eq_elem is not None:
                    for elem in eq_elem.findall("joint"):
                        joint_1 = elem.attrib.get("joint1", "").strip()
                        joint_2 = elem.attrib.get("joint2", "").strip()
                        polycoef_text = elem.attrib.get("polycoef", "0 1 0 0 0")
                        polycoef = [float(x) for x in polycoef_text.split()]
                        while len(polycoef) < 5:
                            polycoef.append(0.0)
                        if abs(polycoef[2]) > 1e-8 or abs(polycoef[3]) > 1e-8:
                            continue
                        if abs(polycoef[4]) > 1e-8:
                            continue
                        if joint_1 and joint_2:
                            eq_rules.append(
                                (joint_1, joint_2, float(polycoef[0]), float(polycoef[1]))
                            )

                tendon_elem = root.find("tendon")
                if tendon_elem is not None:
                    for fixed in tendon_elem.findall("fixed"):
                        joints: list[tuple[str, float]] = []
                        for elem in fixed.findall("joint"):
                            joint_name = elem.attrib.get("joint", "").strip()
                            coef = float(elem.attrib.get("coef", "1.0"))
                            if joint_name:
                                joints.append((joint_name, coef))
                        if len(joints) >= 2:
                            tendon_rules.append(joints)
            except Exception:
                pass

        for _ in range(8):
            changed = False

            for joint_1, joint_2, c0, c1 in eq_rules:
                if joint_1 in row_by_joint:
                    continue
                if joint_2 not in row_by_joint:
                    continue
                row_by_joint[joint_1] = row_by_joint[joint_2] * float(c1)
                bias_by_joint[joint_1] = float(c0) + float(c1) * float(
                    bias_by_joint.get(joint_2, 0.0)
                )
                changed = True

            for joints in tendon_rules:
                unknown = [
                    (name, coef)
                    for name, coef in joints
                    if name not in row_by_joint and abs(coef) > 1e-8
                ]
                if len(unknown) != 1:
                    continue
                dep_joint, dep_coef = unknown[0]
                accum_row = np.zeros(nu, dtype=np.float32)
                accum_bias = 0.0
                valid = True
                for joint_name, coef in joints:
                    if joint_name == dep_joint:
                        continue
                    if joint_name not in row_by_joint:
                        valid = False
                        break
                    accum_row += float(coef) * row_by_joint[joint_name]
                    accum_bias += float(coef) * float(bias_by_joint.get(joint_name, 0.0))
                if not valid:
                    continue
                row_by_joint[dep_joint] = -accum_row / float(dep_coef)
                bias_by_joint[dep_joint] = -accum_bias / float(dep_coef)
                changed = True

            if not changed:
                break

        qpos_mat = np.zeros((int(self.model.nq), nu), dtype=np.float32)
        qpos_bias = np.zeros(int(self.model.nq), dtype=np.float32)
        qpos_mask = np.zeros(int(self.model.nq), dtype=bool)
        qvel_mat = np.zeros((int(self.model.nv), nu), dtype=np.float32)
        qvel_mask = np.zeros(int(self.model.nv), dtype=bool)

        for joint_name, row in row_by_joint.items():
            joint_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name
            )
            if joint_id < 0:
                continue
            jnt_type = int(self.model.jnt_type[joint_id])
            if jnt_type not in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
                continue
            qpos_adr = int(self.model.jnt_qposadr[joint_id])
            qvel_adr = int(self.model.jnt_dofadr[joint_id])
            qpos_mat[qpos_adr] = np.asarray(row, dtype=np.float32)
            qpos_bias[qpos_adr] = float(bias_by_joint.get(joint_name, 0.0))
            qpos_mask[qpos_adr] = True
            qvel_mat[qvel_adr] = np.asarray(row, dtype=np.float32)
            qvel_mask[qvel_adr] = True

        return qpos_mat, qpos_bias, qpos_mask, qvel_mat, qvel_mask

    def get_qpos(self, motor_pos: np.ndarray) -> np.ndarray:
        motor_pos_arr = np.asarray(motor_pos, dtype=np.float32).reshape(-1)
        if motor_pos_arr.shape[0] != self.model.nu:
            raise ValueError(
                f"motor_pos shape {motor_pos_arr.shape[0]} must equal model.nu {self.model.nu}"
            )
        qpos = np.asarray(self.data.qpos, dtype=np.float32).copy()
        if np.any(self._qpos_motor_mask):
            qpos[self._qpos_motor_mask] = (
                self._motor_to_qpos_mat[self._qpos_motor_mask] @ motor_pos_arr
                + self._motor_to_qpos_bias[self._qpos_motor_mask]
            )
        return qpos

    def _get_qvel_from_motor_vel(self, motor_vel: np.ndarray) -> np.ndarray:
        motor_vel_arr = np.asarray(motor_vel, dtype=np.float32).reshape(-1)
        if motor_vel_arr.shape[0] != self.model.nu:
            raise ValueError(
                f"motor_vel shape {motor_vel_arr.shape[0]} must equal model.nu {self.model.nu}"
            )
        qvel = np.asarray(self.data.qvel, dtype=np.float32).copy()
        if np.any(self._qvel_motor_mask):
            qvel[self._qvel_motor_mask] = (
                self._motor_to_qvel_mat[self._qvel_motor_mask] @ motor_vel_arr
            )
        return qvel

    def _flatten_states(self, motor_state: dict, key: str) -> np.ndarray:
        vec: list[float] = []
        for ctrl_key in sorted(motor_state):
            vec.extend(motor_state[ctrl_key].get(key, []))
        arr = np.asarray(vec, dtype=np.float32)
        return arr[self._sort_idx]

    def refresh_control_modes(self) -> None:
        modes = np.asarray(self.motor_control_mode, dtype=object)[self._unsort_idx]
        self.motor_control_mode_sorted = modes
        self.motor_control_mode_split = np.split(modes, self._controller_split_idx)

    def set_motor_control_mode(self, control_mode: str | Sequence[str]) -> None:
        if isinstance(control_mode, str):
            new_modes = [control_mode] * len(self.motor_control_mode)
        else:
            new_modes = list(control_mode)
        if len(new_modes) != len(self.motor_control_mode):
            raise ValueError("Control mode length must match motor count.")

        self.motor_control_mode = new_modes
        self.refresh_control_modes()
        if not hasattr(self._controller, "set_motor_control_mode"):
            return

        mode_vecs = [m.tolist() for m in self.motor_control_mode_split]
        self._controller.set_motor_control_mode(self.controllers, mode_vecs)
        if hasattr(self._controller, "set_motor_pwm"):
            pwm_vecs = np.split(
                np.full(len(self.motor_control_mode), 100.0, dtype=np.float32),
                self._controller_split_idx,
            )
            self._controller.set_motor_pwm(
                self.controllers, [v.tolist() for v in pwm_vecs]
            )
        if self._motor_cur_limits is not None and hasattr(
            self._controller, "set_motor_cur"
        ):
            cur_vecs = np.split(
                self._motor_cur_limits[self._unsort_idx], self._controller_split_idx
            )
            self._controller.set_motor_cur(
                self.controllers, [v.tolist() for v in cur_vecs]
            )

    def _motor_cur_to_torque(
        self,
        motor_cur: np.ndarray,
        motor_vel: np.ndarray,
        motor_pwm: np.ndarray,
        motor_vin: np.ndarray,
    ) -> np.ndarray:
        if self._is_arx:
            return motor_cur.astype(np.float32)

        motor_duty = np.clip(motor_pwm / 100.0, -1.0, 1.0)
        applied_voltage = motor_duty * motor_vin
        back_emf = np.zeros_like(motor_vel, dtype=np.float32)
        valid_kv = self._kv != 0.0
        back_emf[valid_kv] = motor_vel[valid_kv] / self._kv[valid_kv]

        i_est = motor_cur.astype(np.float32) / 1000.0
        valid_r = self._r_winding != 0.0
        missing_cur = ~np.isfinite(i_est)
        compute_mask = missing_cur & valid_r
        i_est[compute_mask] = (
            applied_voltage[compute_mask] - back_emf[compute_mask]
        ) / self._r_winding[compute_mask]
        return (self._kt * i_est).astype(np.float32)

    def _refresh_state(self, retries: int = 1) -> None:
        motor_state = self._controller.get_motor_states(self.controllers, int(retries))
        pos = self._flatten_states(motor_state, "pos")
        vel = self._flatten_states(motor_state, "vel")
        cur = self._flatten_states(motor_state, "cur")
        pwm = self._flatten_states(motor_state, "pwm")
        vin = self._flatten_states(motor_state, "vin")

        if pos.shape[0] != self.model.nu:
            raise RuntimeError(
                f"Motor state size {pos.shape[0]} != model.nu {self.model.nu}."
            )
        tau = self._motor_cur_to_torque(cur, vel, pwm, vin)

        self.data.qpos[:] = self.get_qpos(pos)
        self.data.qvel[:] = self._get_qvel_from_motor_vel(vel)
        self.data.actuator_force[:] = tau
        self.data.time += self.control_dt
        mujoco.mj_forward(self.model, self.data)

    def get_observation(self, retries: int = 1) -> dict[str, Any]:
        motor_state = self._controller.get_motor_states(self.controllers, int(retries))
        motor_pos = self._flatten_states(motor_state, "pos")
        motor_vel = self._flatten_states(motor_state, "vel")
        motor_cur = self._flatten_states(motor_state, "cur")
        motor_pwm = self._flatten_states(motor_state, "pwm")
        motor_vin = self._flatten_states(motor_state, "vin")
        motor_tor = self._motor_cur_to_torque(
            motor_cur, motor_vel, motor_pwm, motor_vin
        )
        qpos = self.get_qpos(motor_pos)
        qvel = self._get_qvel_from_motor_vel(motor_vel)

        obs: dict[str, Any] = {
            "time": float(self.data.time),
            "motor_pos": motor_pos,
            "motor_vel": motor_vel,
            "qpos": qpos,
            "qvel": qvel,
            "motor_cur": motor_cur,
            "motor_tor": motor_tor,
            "motor_pwm": motor_pwm,
            "motor_vin": motor_vin,
        }
        if self.imu is not None:
            imu_state = self.imu.get_latest_state()
            if imu_state is not None:
                obs["imu"] = imu_state
        return obs

    def set_motor_target(self, motor_target: Sequence[float] | np.ndarray) -> None:
        target = np.asarray(motor_target, dtype=np.float32).reshape(-1)
        if target.shape[0] != self.model.nu:
            raise ValueError(
                f"control shape {target.shape[0]} must equal model.nu {self.model.nu}"
            )

        controller_order = target[self._unsort_idx]
        target_split = np.split(controller_order, self._controller_split_idx)

        has_mode_switch = hasattr(self._controller, "set_motor_control_mode")
        if not has_mode_switch:
            self._controller.set_motor_pos(
                self.controllers, [v.tolist() for v in target_split]
            )
            return

        controllers_pos: list[Any] = []
        pos_vecs: list[list[float]] = []
        controllers_cur: list[Any] = []
        cur_vecs: list[list[float]] = []
        controllers_pwm: list[Any] = []
        pwm_vecs: list[list[float]] = []

        for ctrl, mode_vec, vec in zip(
            self.controllers, self.motor_control_mode_split, target_split, strict=False
        ):
            mode_arr = np.asarray(mode_vec, dtype=object)
            is_cur = mode_arr == "current"
            is_pwm = mode_arr == "pwm"
            is_pos = ~(is_cur | is_pwm)

            if np.any(is_pos):
                pos_masked = np.where(is_pos, vec, 0.0)
                controllers_pos.append(ctrl)
                pos_vecs.append(pos_masked.tolist())
            if np.any(is_cur):
                cur_masked = np.where(is_cur, vec, 0.0)
                controllers_cur.append(ctrl)
                cur_vecs.append(cur_masked.tolist())
            if np.any(is_pwm):
                pwm_masked = np.where(is_pwm, vec, 100.0)
                controllers_pwm.append(ctrl)
                pwm_vecs.append(pwm_masked.tolist())

        if controllers_pos:
            self._controller.set_motor_pos(controllers_pos, pos_vecs)
        if controllers_cur and hasattr(self._controller, "set_motor_cur"):
            self._controller.set_motor_cur(controllers_cur, cur_vecs)
        if controllers_pwm and hasattr(self._controller, "set_motor_pwm"):
            self._controller.set_motor_pwm(controllers_pwm, pwm_vecs)

    def step(self) -> None:
        self.set_motor_target(np.asarray(self.data.ctrl, dtype=np.float32))
        self._refresh_state(retries=1)

    def sync(self) -> bool:
        return True

    def close(self) -> None:
        if self.imu is not None:
            try:
                self.imu.close()
            except Exception:
                pass
            self.imu = None
        if hasattr(self, "controllers"):
            try:
                self._controller.close(self.controllers)
            except Exception:
                pass
