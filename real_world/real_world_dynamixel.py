"""Real hardware backend used by policy/run_policy.py.

This backend intentionally does not depend on wrench simulation. It follows the
legacy toddlerbot real-world behavior for current->torque conversion,
observation timing, and IMU warmup.
"""

from __future__ import annotations

import copy
import os
import platform
import re
import time
import xml.etree.ElementTree as ET
from types import ModuleType
from typing import Any, Dict, Sequence

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

from sim.base_sim import BaseSim, Obs


def _import_dynamixel_cpp() -> ModuleType:
    try:
        import dynamixel_cpp  # type: ignore

        return dynamixel_cpp  # type: ignore[return-value]
    except ImportError as exc:
        print(f"Failed to import dynamixel_cpp: {exc}")
        raise


def _resolve_robot_xml_root(xml_path: str) -> ET.Element:
    """Resolve robot XML root from a scene XML that directly includes robot XML."""
    scene_root = ET.parse(xml_path).getroot()
    include_elem = scene_root.find(".//include")
    if include_elem is None:
        return scene_root

    include_file = include_elem.attrib.get("file", "")
    if not include_file:
        return scene_root

    include_path = os.path.abspath(
        os.path.join(os.path.dirname(xml_path), include_file)
    )
    return ET.parse(include_path).getroot()


class RealWorldDynamixel(BaseSim):
    """Real-world backend for Dynamixel-based robots."""

    def __init__(
        self,
        robot: str,
        control_dt: float,
        xml_path: str,
        merged_config: dict[str, Any] | None = None,
    ) -> None:
        self.name = "real_world"
        self.robot_name = str(robot)
        self.xml_path = str(xml_path)
        self.is_toddlerbot = "toddlerbot" in self.robot_name.lower()
        self.control_dt = float(control_dt)

        if merged_config is None:
            raise ValueError(
                "RealWorldDynamixel requires merged_config to be provided."
            )
        self.config = copy.deepcopy(merged_config)

        motors_cfg = self.config.get("motors", {})
        if not isinstance(motors_cfg, dict) or not motors_cfg:
            raise ValueError("Merged motor config is empty.")
        robot_root = _resolve_robot_xml_root(xml_path)
        self.motor_ordering = [
            motor_name
            for motor_name in motors_cfg
            if robot_root.find(f".//joint[@name='{motor_name}']") is not None
        ]
        if not self.motor_ordering:
            raise ValueError(
                f"No motor joints from merged config found in XML tree: {xml_path}"
            )

        actuators_cfg = self.config.get("actuators", {})
        if not isinstance(actuators_cfg, dict):
            actuators_cfg = {}
        self.gain_backdrive = float(actuators_cfg.get("gain_backdrive", 1.0))

        self.motor_groups: np.ndarray = np.asarray(
            [str(motors_cfg[name].get("group", "arm")) for name in self.motor_ordering],
            dtype=object,
        )
        self.motor_kv = np.zeros(len(self.motor_ordering), dtype=np.float32)
        self.motor_r_winding = np.ones(len(self.motor_ordering), dtype=np.float32)
        self.motor_kt = np.ones(len(self.motor_ordering), dtype=np.float32)
        self.cur_sensor_mask = np.ones(len(self.motor_ordering), dtype=bool)

        motor_kp: list[float] = []
        motor_kd: list[float] = []
        motor_ki: list[float] = []
        motor_zero_pos: list[float] = []
        motor_control_mode: list[str] = []
        for i, name in enumerate(self.motor_ordering):
            m_cfg = motors_cfg[name]
            motor_type = str(m_cfg.get("motor", ""))
            a_cfg = actuators_cfg.get(motor_type, {})
            motor_kp.append(float(m_cfg.get("kp", 0.0)))
            motor_kd.append(float(m_cfg.get("kd", 0.0)))
            motor_ki.append(float(m_cfg.get("ki", 0.0)))
            motor_zero_pos.append(float(m_cfg.get("zero_pos", 0.0)))
            motor_control_mode.append(
                str(m_cfg.get("control_mode", "extended_position"))
            )
            self.motor_kv[i] = float(a_cfg.get("kv", 1.0))
            self.motor_r_winding[i] = float(a_cfg.get("r_winding", 1.0))
            self.motor_kt[i] = float(a_cfg.get("kt", 1.0))
            self.cur_sensor_mask[i] = bool(m_cfg.get("cur_sensor", True))

        self.motor_control_mode = motor_control_mode
        self.position_control_mode = list(self.motor_control_mode)

        self.imu = None
        try:
            from real_world.IMU import ThreadedIMU

            self.imu = ThreadedIMU()
            self.imu.start()
        except Exception as exc:
            print(f"IMU fails to initialize: {exc}")
            self.imu = None

        self.controller, port_pattern, baudrate = self._load_controller_backend()

        self.controllers = self.controller.create_controllers(
            port_pattern,
            motor_kp,
            motor_kd,
            motor_ki,
            motor_zero_pos,
            self.motor_control_mode,
            baudrate,
            1,
        )
        self.controller.initialize(self.controllers)
        self.motor_ids = self.controller.get_motor_ids(self.controllers)

        motor_ids_flat = np.array(
            sum((self.motor_ids[key] for key in sorted(self.motor_ids)), []),
            dtype=np.int32,
        )
        n_motor = len(self.motor_ordering)
        if motor_ids_flat.size != n_motor:
            raise ValueError(
                f"Motor ID count {motor_ids_flat.size} != configured motor count {n_motor}."
            )
        if np.unique(motor_ids_flat).size != n_motor:
            raise ValueError(
                "Motor IDs contain duplicates; cannot build stable ordering."
            )
        self.motor_sort_idx = np.argsort(motor_ids_flat)
        self.motor_unsort_idx = np.argsort(self.motor_sort_idx)

        self.motor_lens = [len(self.motor_ids[key]) for key in sorted(self.motor_ids)]
        self.motor_split_idx = np.cumsum(self.motor_lens)[:-1]
        self.refresh_control_modes()

        self.motor_cur_limits = None
        try:
            cur_limits = self.controller.get_motor_current_limits(self.controllers)
            all_limits: list[float] = []
            for key in sorted(self.motor_ids.keys()):
                all_limits.extend(cur_limits.get(key, []))
            if all_limits:
                self.motor_cur_limits = np.asarray(all_limits, dtype=np.float32)[
                    self.motor_sort_idx
                ]
        except Exception as exc:
            print(f"Failed to read current limits: {exc}")

        self.drive_state = np.ones(len(self.motor_ordering), dtype=np.float32)
        self._build_motor_to_qpos_map(robot_root)

        if self.imu is not None:
            imu_data = self.imu.get_latest_state()
            counter = 0
            while not imu_data:
                counter += 1
                print(
                    f"\rWaiting for real-world observation data... [{counter}]",
                    end="",
                    flush=True,
                )
                imu_data = self.imu.get_latest_state()
            print("\nData received.")

    def _load_controller_backend(self) -> tuple[Any, str, int]:
        controller = _import_dynamixel_cpp()
        if "leap" in self.robot_name.lower():
            port_pattern = (
                r"cu\.usbserial-.*"
                if platform.system() == "Darwin"
                else r"ttyUSB[0-9]+"
            )
        else:
            port_pattern = (
                r"cu\.usbserial-.*"
                if platform.system() == "Darwin"
                else r"tty(?:CH9344)?USB[0-9]+"
            )
        return controller, port_pattern, 2_000_000

    def _estimate_motor_torque_inputs(
        self,
        motor_cur_arr: np.ndarray,
        motor_vel_arr: np.ndarray,
        motor_pwm_arr: np.ndarray,
        motor_vin_arr: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        gain_back = np.ones_like(motor_cur_arr, dtype=np.float32)
        gain_back[self.motor_groups == "arm"] = float(self.gain_backdrive)

        motor_duty = np.clip(motor_pwm_arr / 100.0, -1.0, 1.0)
        applied_voltage = motor_duty * motor_vin_arr

        cur_mask = self.cur_sensor_mask

        back_emf = np.zeros_like(motor_vel_arr, dtype=np.float32)
        valid_kv = self.motor_kv != 0.0
        back_emf[valid_kv] = motor_vel_arr[valid_kv] / self.motor_kv[valid_kv]

        i_est = np.zeros_like(motor_vel_arr, dtype=np.float32)
        i_est[cur_mask] = motor_cur_arr[cur_mask] / 1000.0
        compute_mask = (~cur_mask) & (self.motor_r_winding != 0.0)
        i_est[compute_mask] = (
            applied_voltage[compute_mask] - back_emf[compute_mask]
        ) / self.motor_r_winding[compute_mask]
        tau_est = self.motor_kt * i_est
        return gain_back, i_est, tau_est

    def refresh_control_modes(self) -> None:
        modes = np.asarray(self.motor_control_mode, dtype=object)[self.motor_unsort_idx]
        self.motor_control_mode_sorted = modes
        self.motor_control_mode_split = np.split(modes, self.motor_split_idx)

    def set_motor_control_mode(self, control_mode: str | Sequence[str]) -> None:
        if isinstance(control_mode, str):
            new_modes = [control_mode] * len(self.motor_control_mode)
        else:
            new_modes = [str(m) for m in control_mode]
        if len(new_modes) != len(self.motor_control_mode):
            raise ValueError("Control mode list length must match number of motors.")

        self.motor_control_mode = new_modes
        self.refresh_control_modes()

        if not hasattr(self.controller, "set_motor_control_mode"):
            print("Motor control mode switching is not supported by this controller.")
            return

        mode_vecs = [m.tolist() for m in self.motor_control_mode_split]
        self.controller.set_motor_control_mode(self.controllers, mode_vecs)

        pwm_vecs = np.split(
            np.full(len(self.motor_control_mode), 100.0, dtype=np.float32),
            self.motor_split_idx,
        )
        self.controller.set_motor_pwm(
            self.controllers,
            [vec.tolist() for vec in pwm_vecs],
        )

        if self.motor_cur_limits is not None:
            cur_vecs = np.split(
                self.motor_cur_limits[self.motor_unsort_idx],
                self.motor_split_idx,
            )
            self.controller.set_motor_cur(
                self.controllers,
                [vec.tolist() for vec in cur_vecs],
            )

    def step(self) -> None:
        """No-op for real hardware backend (kept for interface symmetry)."""

    def sync(self) -> bool:
        return True

    def motor_cur_to_tor(
        self,
        motor_cur: np.ndarray,
        motor_vel: np.ndarray,
        motor_pwm: np.ndarray,
        motor_vin: np.ndarray,
        eps_vel: float = 0.05,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        motor_cur_arr = np.asarray(motor_cur, dtype=np.float32).copy()
        motor_vel_arr = np.asarray(motor_vel, dtype=np.float32)
        motor_pwm_arr = np.asarray(motor_pwm, dtype=np.float32)
        motor_vin_arr = np.asarray(motor_vin, dtype=np.float32)
        gain_back, i_est, tau_est = self._estimate_motor_torque_inputs(
            motor_cur_arr=motor_cur_arr,
            motor_vel_arr=motor_vel_arr,
            motor_pwm_arr=motor_pwm_arr,
            motor_vin_arr=motor_vin_arr,
        )

        prev = np.asarray(self.drive_state, dtype=np.float32)
        drive_state = np.where(
            np.abs(motor_vel_arr) > float(eps_vel),
            np.sign(tau_est * motor_vel_arr),
            prev,
        ).astype(np.float32)
        self.drive_state = drive_state

        gain = np.where(drive_state > 0, 1.0, gain_back).astype(np.float32)
        return (
            i_est.astype(np.float32),
            drive_state,
            (gain * tau_est).astype(np.float32),
        )

    def _build_motor_to_qpos_map(self, xml_root: ET.Element) -> None:
        nu = len(self.motor_ordering)
        motor_index = {name: idx for idx, name in enumerate(self.motor_ordering)}

        def get_transmission(motor_name: str) -> str:
            if motor_name.endswith("_drive"):
                return "spur_gear"
            if motor_name.endswith("_act"):
                return "parallel_linkage"
            if re.search(r"_act_[12]$", motor_name):
                return "bevel_gear"
            if motor_name.endswith("_rack"):
                return "rack_and_pinion"
            return "none"

        def get_gear_ratio(name: str) -> float:
            joint_equality = xml_root.find(f".//equality/joint[@joint2='{name}']")
            if joint_equality is None:
                return 1.0
            polycoef = joint_equality.attrib.get("polycoef", "0 1 0 0 0").split()
            if len(polycoef) < 2:
                return 1.0
            return float(polycoef[1])

        motor_gear_ratios = {
            motor_name: get_gear_ratio(motor_name) for motor_name in self.motor_ordering
        }

        bevel_motor_names = [
            name
            for name in self.motor_ordering
            if get_transmission(name) == "bevel_gear"
        ]
        if bevel_motor_names:
            bevel_motor_names.sort(
                key=lambda name: (
                    0
                    if name.endswith("_act_1")
                    else 1
                    if name.endswith("_act_2")
                    else motor_index[name] + 2
                )
            )
        bevel_joint_names = ("waist_roll", "waist_yaw") if bevel_motor_names else ()

        motor_to_joint_map: dict[str, tuple[str, float]] = {}
        for motor_name in self.motor_ordering:
            transmission = get_transmission(motor_name)
            if transmission == "bevel_gear":
                continue
            if transmission == "spur_gear":
                joint_name = motor_name.replace("_drive", "_driven")
                ratio = float(motor_gear_ratios[motor_name])
            elif transmission == "parallel_linkage":
                joint_name = motor_name.replace("_act", "")
                ratio = 1.0
            elif transmission == "rack_and_pinion":
                joint_name = motor_name.replace("_rack", "_pinion")
                ratio = float(motor_gear_ratios[motor_name])
            else:
                joint_name = motor_name
                ratio = 1.0
            motor_to_joint_map[motor_name] = (joint_name, ratio)

        joint_ordering: list[str] = []
        bevel_pending = False
        bevel_added = False
        for motor_name in self.motor_ordering:
            transmission = get_transmission(motor_name)
            if transmission == "bevel_gear":
                if not bevel_pending:
                    bevel_pending = True
                elif not bevel_added:
                    joint_ordering.extend(bevel_joint_names)
                    bevel_added = True
                continue
            joint_ordering.append(motor_to_joint_map[motor_name][0])

        joint_name_to_index = {name: idx for idx, name in enumerate(joint_ordering)}
        motor_to_joint_mat = np.zeros((len(joint_ordering), nu), dtype=np.float32)
        for motor_name, (joint_name, ratio) in motor_to_joint_map.items():
            if joint_name not in joint_name_to_index:
                continue
            motor_to_joint_mat[
                joint_name_to_index[joint_name], motor_index[motor_name]
            ] = ratio

        if bevel_motor_names and all(
            name in joint_name_to_index for name in bevel_joint_names
        ):
            act_1_idx, act_2_idx = [motor_index[name] for name in bevel_motor_names]
            roll_idx, yaw_idx = [
                joint_name_to_index[name] for name in bevel_joint_names
            ]
            roll_coef = float(
                self.config.get("kinematics", {}).get("waist_roll_coef", 1.0)
            )
            yaw_coef = float(
                self.config.get("kinematics", {}).get("waist_yaw_coef", 1.0)
            )
            motor_to_joint_mat[roll_idx, act_1_idx] = -roll_coef
            motor_to_joint_mat[roll_idx, act_2_idx] = roll_coef
            motor_to_joint_mat[yaw_idx, act_1_idx] = yaw_coef
            motor_to_joint_mat[yaw_idx, act_2_idx] = yaw_coef

        parallel_linkage_joint_names = [
            joint_name
            for motor_name, (joint_name, _) in motor_to_joint_map.items()
            if motor_name.endswith("_act")
        ]
        rack_and_pinion_joint_names = [
            joint_name
            for motor_name, (joint_name, _) in motor_to_joint_map.items()
            if motor_name.endswith("_rack")
        ]

        dof_names = list(
            dict.fromkeys(
                joint_ordering
                + self.motor_ordering
                + [f"{name}_front" for name in parallel_linkage_joint_names]
                + [f"{name}_back" for name in parallel_linkage_joint_names]
                + [f"{name}_mirror" for name in rack_and_pinion_joint_names]
            )
        )
        dof_index = {name: idx for idx, name in enumerate(dof_names)}
        motor_to_dof_mat = np.zeros((len(dof_names), nu), dtype=np.float32)

        for joint_name in joint_ordering:
            dof_idx = dof_index[joint_name]
            joint_idx = joint_name_to_index[joint_name]
            motor_to_dof_mat[dof_idx] = motor_to_joint_mat[joint_idx]
        for motor_name in self.motor_ordering:
            if motor_name in joint_name_to_index:
                continue
            dof_idx = dof_index[motor_name]
            motor_to_dof_mat[dof_idx, motor_index[motor_name]] = 1.0
        for joint_name in parallel_linkage_joint_names:
            joint_idx = joint_name_to_index[joint_name]
            for suffix in ("_front", "_back"):
                dof_idx = dof_index[joint_name + suffix]
                motor_to_dof_mat[dof_idx] = -motor_to_joint_mat[joint_idx]
        for joint_name in rack_and_pinion_joint_names:
            joint_idx = joint_name_to_index[joint_name]
            dof_idx = dof_index[joint_name + "_mirror"]
            motor_to_dof_mat[dof_idx] = motor_to_joint_mat[joint_idx]
        self._motor_to_dof_mat = motor_to_dof_mat
        model = mujoco.MjModel.from_xml_path(self.xml_path)
        self._qpos_dim = int(model.nq)
        self._qvel_dim = int(model.nv)
        self._qpos_base = np.zeros(self._qpos_dim, dtype=np.float32)

        self._has_floating_base = bool(
            np.any(model.jnt_type == int(mujoco.mjtJoint.mjJNT_FREE))
        )
        if int(model.nkey) > 0:
            self._qpos_base = np.asarray(model.key_qpos[0], dtype=np.float32)
        elif self._has_floating_base and self._qpos_dim >= 7:
            base_pos = np.asarray(
                self.config.get("kinematics", {}).get("zero_pos", [0.0, 0.0, 0.0]),
                dtype=np.float32,
            ).reshape(-1)
            if base_pos.size >= 3:
                self._qpos_base[:3] = base_pos[:3]
            self._qpos_base[3] = 1.0

        qpos_joint_idx: list[int] = []
        qvel_joint_idx: list[int] = []
        dof_src_idx: list[int] = []
        for joint_name, src_idx in dof_index.items():
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id < 0:
                continue
            joint_type = int(model.jnt_type[joint_id])
            if joint_type not in (
                int(mujoco.mjtJoint.mjJNT_HINGE),
                int(mujoco.mjtJoint.mjJNT_SLIDE),
            ):
                continue
            qpos_joint_idx.append(int(model.jnt_qposadr[joint_id]))
            qvel_joint_idx.append(int(model.jnt_dofadr[joint_id]))
            dof_src_idx.append(int(src_idx))

        self._qpos_joint_idx = np.asarray(qpos_joint_idx, dtype=np.int32)
        self._qvel_joint_idx = np.asarray(qvel_joint_idx, dtype=np.int32)
        self._qmap_dof_idx = np.asarray(dof_src_idx, dtype=np.int32)
        if (
            self._qpos_joint_idx.shape[0] != self._qvel_joint_idx.shape[0]
            or self._qpos_joint_idx.shape[0] != self._qmap_dof_idx.shape[0]
        ):
            raise ValueError("Inconsistent qpos/qvel/dof joint mapping dimensions.")

        expected_joint_names: list[str] = []
        for joint_id in range(int(model.njnt)):
            joint_type = int(model.jnt_type[joint_id])
            if joint_type not in (
                int(mujoco.mjtJoint.mjJNT_HINGE),
                int(mujoco.mjtJoint.mjJNT_SLIDE),
            ):
                continue
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            if name:
                expected_joint_names.append(name)
        missing_joint_names = [
            name for name in expected_joint_names if name not in dof_index
        ]
        if missing_joint_names:
            raise ValueError(
                "Missing hinge/slide joints in dof mapping: "
                + ", ".join(missing_joint_names)
            )

    def get_qpos(
        self,
        motor_pos: np.ndarray,
        root_quat: np.ndarray | None = None,
    ) -> np.ndarray:
        motor_pos_arr = np.asarray(motor_pos, dtype=np.float32).reshape(-1)
        if motor_pos_arr.shape[0] != len(self.motor_ordering):
            raise ValueError(
                f"motor_pos length {motor_pos_arr.shape[0]} != {len(self.motor_ordering)}"
            )
        dof_arr = self._motor_to_dof_mat @ motor_pos_arr
        qpos = self._qpos_base.copy()
        if (
            self._has_floating_base
            and root_quat is not None
            and np.asarray(root_quat).shape[0] == 4
        ):
            qpos[3:7] = np.asarray(root_quat, dtype=np.float32)
        if self._qpos_joint_idx.size > 0:
            qpos[self._qpos_joint_idx] = dof_arr[self._qmap_dof_idx]
        return qpos

    def get_qvel(self, motor_vel: np.ndarray) -> np.ndarray:
        motor_vel_arr = np.asarray(motor_vel, dtype=np.float32).reshape(-1)
        if motor_vel_arr.shape[0] != len(self.motor_ordering):
            raise ValueError(
                f"motor_vel length {motor_vel_arr.shape[0]} != {len(self.motor_ordering)}"
            )
        dof_vel = self._motor_to_dof_mat @ motor_vel_arr
        qvel = np.zeros(self._qvel_dim, dtype=np.float32)
        if self._qvel_joint_idx.size > 0:
            qvel[self._qvel_joint_idx] = dof_vel[self._qmap_dof_idx]
        return qvel

    def get_observation(self, retries: int = 0) -> Obs:
        motor_state = self.controller.get_motor_states(self.controllers, int(retries))

        all_motor_pos: list[float] = []
        all_motor_vel: list[float] = []
        all_motor_cur: list[float] = []
        all_motor_pwm: list[float] = []
        all_motor_vin: list[float] = []
        all_motor_temp: list[float] = []

        for key in sorted(self.motor_ids.keys()):
            data = motor_state[key]
            all_motor_pos.extend(data["pos"])
            all_motor_vel.extend(data["vel"])
            all_motor_cur.extend(data["cur"])
            all_motor_pwm.extend(data["pwm"])
            all_motor_vin.extend(data["vin"])
            all_motor_temp.extend(data["temp"])

        motor_pos = np.asarray(all_motor_pos, dtype=np.float32)[self.motor_sort_idx]
        motor_vel = np.asarray(all_motor_vel, dtype=np.float32)[self.motor_sort_idx]
        motor_cur_raw = np.asarray(all_motor_cur, dtype=np.float32)[self.motor_sort_idx]
        motor_pwm = np.asarray(all_motor_pwm, dtype=np.float32)[self.motor_sort_idx]
        motor_vin = np.asarray(all_motor_vin, dtype=np.float32)[self.motor_sort_idx]
        motor_temp = np.asarray(all_motor_temp, dtype=np.float32)[self.motor_sort_idx]

        motor_cur, motor_drive, motor_tor = self.motor_cur_to_tor(
            motor_cur=motor_cur_raw,
            motor_vel=motor_vel,
            motor_pwm=motor_pwm,
            motor_vin=motor_vin,
        )

        quat, ang_vel = None, None
        if self.imu is not None:
            latest = self.imu.get_latest_state()
            if latest is not None:
                quat, ang_vel = latest

        rot = None
        if quat is not None:
            rot = R.from_quat(np.asarray(quat, dtype=np.float32), scalar_first=True)

        return Obs(
            ang_vel=ang_vel,
            time=float(time.monotonic()),
            motor_pos=motor_pos,
            motor_vel=motor_vel,
            motor_tor=motor_tor,
            qpos=self.get_qpos(motor_pos),
            qvel=self.get_qvel(motor_vel),
            rot=rot,
            motor_cur=motor_cur,
            motor_drive=motor_drive,
            motor_pwm=motor_pwm,
            motor_vin=motor_vin,
            motor_temp=motor_temp,
        )

    def set_motor_target(self, motor_angles: Dict[str, float] | np.ndarray) -> None:
        if isinstance(motor_angles, dict):
            if all(name in motor_angles for name in self.motor_ordering):
                motor_pos = np.asarray(
                    [motor_angles[name] for name in self.motor_ordering],
                    dtype=np.float32,
                )
            else:
                motor_pos = np.asarray(list(motor_angles.values()), dtype=np.float32)
        else:
            motor_pos = np.asarray(motor_angles, dtype=np.float32)

        if motor_pos.shape[0] != len(self.motor_ordering):
            raise ValueError(
                f"motor target len {motor_pos.shape[0]} != {len(self.motor_ordering)}"
            )

        reordered_pos = motor_pos[self.motor_unsort_idx]
        pos_splits = np.split(reordered_pos, self.motor_split_idx)

        controllers_pos = []
        pos_vecs = []
        controllers_cur = []
        cur_vecs = []
        controllers_pwm = []
        pwm_vecs = []

        for ctrl, mode_vec, vec in zip(
            self.controllers,
            self.motor_control_mode_split,
            pos_splits,
        ):
            mode_arr = np.asarray(mode_vec)
            is_cur = mode_arr == "current"
            is_pwm = mode_arr == "pwm"
            is_pos = ~(is_cur | is_pwm)

            if np.any(is_pos):
                controllers_pos.append(ctrl)
                pos_vecs.append(np.where(is_pos, vec, 0.0).tolist())
            if np.any(is_cur):
                controllers_cur.append(ctrl)
                cur_vecs.append(np.where(is_cur, vec, 0.0).tolist())
            if np.any(is_pwm):
                controllers_pwm.append(ctrl)
                pwm_vecs.append(np.where(is_pwm, vec, 100.0).tolist())

        if controllers_pos:
            self.controller.set_motor_pos(controllers_pos, pos_vecs)
        if controllers_cur:
            self.controller.set_motor_cur(controllers_cur, cur_vecs)
        if controllers_pwm:
            self.controller.set_motor_pwm(controllers_pwm, pwm_vecs)

    def close(self) -> None:
        if self.imu is not None:
            try:
                self.imu.close()
            except Exception:
                pass
            self.imu = None

        try:
            self.controller.close(self.controllers)
        except Exception:
            pass
