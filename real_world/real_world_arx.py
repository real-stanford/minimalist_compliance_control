"""Real-world ARX backend used by policy/run_policy.py.

This backend is intentionally simplified for a fixed-base robot arm setup.
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict

import mujoco
import numpy as np

from sim.base_sim import BaseSim, Obs

MotorState = Dict[str, Dict[str, list[float]]]


class RealWorldARX(BaseSim):
    """ARX hardware backend with direct arm-joint mapping."""

    def __init__(
        self,
        robot: str,
        control_dt: float,
        xml_path: str,
        merged_config: dict[str, Any] | None = None,
    ) -> None:
        if "arx" not in str(robot).lower():
            raise ValueError("RealWorldARX requires an arx robot name.")
        # Kept for call-site compatibility with other real-world backends.
        del merged_config

        self.name = "real_world"
        self.robot_name = str(robot)
        self.control_dt = float(control_dt)
        self.xml_path = str(xml_path)

        self.model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.data = mujoco.MjData(self.model)

        model_actuators: list[str] = []
        for i in range(int(self.model.nu)):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            if name is None:
                raise ValueError(f"Actuator {i} has no name in XML: {self.xml_path}")
            model_actuators.append(str(name))

        # Use the full actuator order from the ARX MuJoCo model (including gripper)
        # so policy action dimension matches model.nu.
        self.motor_ordering = model_actuators
        self._n_motor = len(self.motor_ordering)
        motor_kp = [0.0] * self._n_motor
        motor_kd = [0.0] * self._n_motor
        motor_ki = [0.0] * self._n_motor
        motor_zero_pos = [0.0] * self._n_motor
        self.motor_control_mode = ["position"] * self._n_motor

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

        self._build_motor_order_indices()
        self._build_qpos_qvel_indices()

        self.drive_state = np.ones(self._n_motor, dtype=np.float32)

    def _load_controller_backend(self) -> tuple[Any, str, int]:
        from real_world import arx_controller as controller

        return controller, os.getenv("ARX5_INTERFACE", "(?:slcan|can)[0-9]+"), 0

    def _build_motor_order_indices(self) -> None:
        motor_ids_flat = np.array(
            sum((self.motor_ids[key] for key in sorted(self.motor_ids)), []),
            dtype=np.int32,
        )
        if motor_ids_flat.size != self._n_motor:
            raise ValueError(
                f"Motor ID count {motor_ids_flat.size} != configured motor count {self._n_motor}."
            )
        if np.unique(motor_ids_flat).size != self._n_motor:
            raise ValueError(
                "Motor IDs contain duplicates; cannot build stable ordering."
            )

        self.motor_sort_idx = np.argsort(motor_ids_flat)
        self.motor_unsort_idx = np.argsort(self.motor_sort_idx)
        self.motor_lens = [len(self.motor_ids[key]) for key in sorted(self.motor_ids)]
        self.motor_split_idx = np.cumsum(self.motor_lens)[:-1]

    def _build_qpos_qvel_indices(self) -> None:
        self._qpos_base = np.zeros(int(self.model.nq), dtype=np.float32)
        if int(self.model.nkey) > 0:
            self._qpos_base[:] = np.asarray(self.model.key_qpos[0], dtype=np.float32)

        actuator_idx = np.arange(self._n_motor, dtype=np.int32)
        joint_ids = np.asarray(
            self.model.actuator_trnid[actuator_idx, 0], dtype=np.int32
        )
        if np.any(joint_ids < 0):
            missing = [
                self.motor_ordering[i] for i, jid in enumerate(joint_ids) if jid < 0
            ]
            raise ValueError(
                "ARX actuator(s) without mapped joint in model: " + ", ".join(missing)
            )

        self._motor_qpos_idx = np.asarray(
            self.model.jnt_qposadr[joint_ids], dtype=np.int32
        )
        self._motor_qvel_idx = np.asarray(
            self.model.jnt_dofadr[joint_ids], dtype=np.int32
        )

    def _as_motor_vec(self, value: np.ndarray, label: str) -> np.ndarray:
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
        if arr.shape[0] != self._n_motor:
            raise ValueError(f"{label} length {arr.shape[0]} != {self._n_motor}")
        return arr

    def step(self) -> None:
        """No-op for hardware backend."""

    def sync(self) -> bool:
        return True

    def motor_cur_to_tor(
        self,
        motor_cur: np.ndarray,
        motor_vel: np.ndarray,
        eps_vel: float = 0.05,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        motor_cur_arr = self._as_motor_vec(motor_cur, "motor_cur")
        motor_vel_arr = self._as_motor_vec(motor_vel, "motor_vel")

        prev = np.asarray(self.drive_state, dtype=np.float32)
        drive_state = np.where(
            np.abs(motor_vel_arr) > float(eps_vel),
            np.sign(motor_cur_arr * motor_vel_arr),
            prev,
        ).astype(np.float32)
        self.drive_state = drive_state
        return motor_cur_arr.copy(), drive_state, motor_cur_arr.copy()

    def get_qpos(self, motor_pos: np.ndarray) -> np.ndarray:
        motor_pos_arr = self._as_motor_vec(motor_pos, "motor_pos")
        qpos = self._qpos_base.copy()
        qpos[self._motor_qpos_idx] = motor_pos_arr
        return qpos

    def get_qvel(self, motor_vel: np.ndarray) -> np.ndarray:
        motor_vel_arr = self._as_motor_vec(motor_vel, "motor_vel")
        qvel = np.zeros(int(self.model.nv), dtype=np.float32)
        qvel[self._motor_qvel_idx] = motor_vel_arr
        return qvel

    def _flatten_motor_field(
        self,
        motor_state: MotorState,
        key_name: str,
    ) -> np.ndarray:
        values: list[float] = []
        for key in sorted(self.motor_ids.keys()):
            block = motor_state.get(key, {})
            values.extend(block.get(key_name, []))
        arr = np.asarray(values, dtype=np.float32).reshape(-1)
        if arr.shape[0] != self._n_motor:
            raise ValueError(
                f"ARX state[{key_name!r}] length {arr.shape[0]} != expected {self._n_motor}."
            )
        return arr[self.motor_sort_idx]

    def get_observation(self, retries: int = 0) -> Obs:
        motor_state = self.controller.get_motor_states(self.controllers, int(retries))

        fields = {
            key: self._flatten_motor_field(motor_state, key)
            for key in ("pos", "vel", "cur", "pwm", "vin", "temp")
        }
        motor_pos = fields["pos"]
        motor_vel = fields["vel"]
        motor_cur_raw = fields["cur"]
        motor_pwm = fields["pwm"]
        motor_vin = fields["vin"]
        motor_temp = fields["temp"]

        motor_cur, motor_drive, motor_tor = self.motor_cur_to_tor(
            motor_cur=motor_cur_raw,
            motor_vel=motor_vel,
        )

        return Obs(
            ang_vel=np.zeros(3, dtype=np.float32),
            time=float(time.monotonic()),
            motor_pos=motor_pos,
            motor_vel=motor_vel,
            motor_tor=motor_tor,
            qpos=self.get_qpos(motor_pos),
            qvel=self.get_qvel(motor_vel),
            rot=None,
            motor_cur=motor_cur,
            motor_drive=motor_drive,
            motor_pwm=motor_pwm,
            motor_vin=motor_vin,
            motor_temp=motor_temp,
        )

    def set_motor_target(self, motor_angles: Dict[str, float] | np.ndarray) -> None:
        if isinstance(motor_angles, dict):
            key_set = set(motor_angles.keys())
            expected = set(self.motor_ordering)
            missing = [name for name in self.motor_ordering if name not in key_set]
            extra = sorted(key_set - expected)
            if missing or extra:
                details: list[str] = []
                if missing:
                    details.append(f"missing keys: {missing}")
                if extra:
                    details.append(f"unexpected keys: {extra}")
                raise ValueError(
                    "motor target dict must exactly match motor_ordering; "
                    + "; ".join(details)
                )
            target = np.asarray(
                [motor_angles[name] for name in self.motor_ordering],
                dtype=np.float32,
            )
        else:
            target = np.asarray(motor_angles, dtype=np.float32).reshape(-1)

        target = self._as_motor_vec(target, "motor target")

        target_sorted_for_ctrl = target[self.motor_unsort_idx]
        pos_vecs = [
            vec.tolist()
            for vec in np.split(target_sorted_for_ctrl, self.motor_split_idx)
        ]
        self.controller.set_motor_pos(self.controllers, pos_vecs)

    def close(self) -> None:
        try:
            self.controller.close(self.controllers)
        except Exception:
            pass
