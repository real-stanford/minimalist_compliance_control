"""Real-world G1 backend for policy/run_policy.py."""

from __future__ import annotations

import time
from typing import Dict

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

from sim.base_sim import BaseSim, Obs
from real_world import g1_controller


class RealWorldG1(BaseSim):
    """G1 hardware backend using Unitree SDK2 low-level topics."""

    def __init__(
        self,
        control_dt: float,
        xml_path: str,
        net_if: str = "",
    ) -> None:
        self.name = "real_world"
        self.control_dt = float(control_dt)
        self.xml_path = str(xml_path)
        self.model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.data = mujoco.MjData(self.model)

        self.motor_ordering = []
        for i in range(int(self.model.nu)):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            if name is None:
                raise ValueError(f"Actuator {i} has no name in XML: {self.xml_path}")
            self.motor_ordering.append(str(name))
        self._motor_name_to_idx = {
            name: i for i, name in enumerate(self.motor_ordering)
        }

        trnid = np.asarray(self.model.actuator_trnid[:, 0], dtype=np.int32)
        if np.any(trnid < 0):
            raise ValueError("G1 model has actuator(s) without mapped joints.")
        self._qpos_adr = np.asarray(self.model.jnt_qposadr[trnid], dtype=np.int32)
        self._qvel_adr = np.asarray(self.model.jnt_dofadr[trnid], dtype=np.int32)

        self._qpos_base = np.zeros(int(self.model.nq), dtype=np.float32)
        if int(self.model.nkey) > 0:
            self._qpos_base[:] = np.asarray(self.model.key_qpos[0], dtype=np.float32)
        elif int(self.model.nq) >= 7:
            self._qpos_base[3] = 1.0

        kp = np.ones(int(self.model.nu), dtype=np.float32) * 40.0
        kd = np.ones(int(self.model.nu), dtype=np.float32) * 1.0
        g1_default_kp = np.asarray(
            [
                60,
                60,
                60,
                100,
                40,
                40,  # legs
                60,
                60,
                60,
                100,
                40,
                40,  # legs
                60,
                40,
                40,  # waist
                40,
                40,
                40,
                40,
                40,
                40,
                40,  # left arm
                40,
                40,
                40,
                40,
                40,
                40,
                40,  # right arm
            ],
            dtype=np.float32,
        )
        g1_default_kd = np.asarray(
            [
                1,
                1,
                1,
                2,
                1,
                1,  # legs
                1,
                1,
                1,
                2,
                1,
                1,  # legs
                1,
                1,
                1,  # waist
                1,
                1,
                1,
                1,
                1,
                1,
                1,  # left arm
                1,
                1,
                1,
                1,
                1,
                1,
                1,  # right arm
            ],
            dtype=np.float32,
        )
        default_gain_by_joint = {
            name: (float(g1_default_kp[i]), float(g1_default_kd[i]))
            for i, name in enumerate(g1_controller.G1_BODY_JOINT_NAMES)
        }
        for i, name in enumerate(self.motor_ordering):
            gains = default_gain_by_joint.get(str(name))
            if gains is None:
                continue
            kp[i], kd[i] = gains

        interface = str(net_if).strip()
        self.controller = g1_controller
        self.controllers = self.controller.create_controllers(
            interface,
            kp.tolist(),
            kd.tolist(),
            actuator_names=self.motor_ordering,
            control_dt=self.control_dt,
        )
        self.controller.initialize(self.controllers)
        self.motor_ids = self.controller.get_motor_ids(self.controllers)

        self._last_motor_pos = np.zeros(int(self.model.nu), dtype=np.float32)
        self._last_motor_vel = np.zeros(int(self.model.nu), dtype=np.float32)
        self._last_motor_tor = np.zeros(int(self.model.nu), dtype=np.float32)
        self._last_motor_cur = np.zeros(int(self.model.nu), dtype=np.float32)
        self._last_motor_pwm = np.zeros(int(self.model.nu), dtype=np.float32)
        self._last_motor_vin = np.zeros(int(self.model.nu), dtype=np.float32)
        self._last_motor_temp = np.zeros(int(self.model.nu), dtype=np.float32)
        self._last_quat_wxyz = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self._last_gyro = np.zeros(3, dtype=np.float32)

    def step(self) -> None:
        """No-op for hardware backend."""

    def sync(self) -> bool:
        return True

    def _flatten_state(
        self, motor_state: Dict[str, Dict[str, list[float]]]
    ) -> Dict[str, np.ndarray]:
        all_pos = []
        all_vel = []
        all_cur = []
        all_pwm = []
        all_vin = []
        all_temp = []
        for key in sorted(self.motor_ids.keys()):
            block = motor_state.get(key, {})
            all_pos.extend(block.get("pos", []))
            all_vel.extend(block.get("vel", []))
            all_cur.extend(block.get("cur", []))
            all_pwm.extend(block.get("pwm", []))
            all_vin.extend(block.get("vin", []))
            all_temp.extend(block.get("temp", []))
        return {
            "pos": np.asarray(all_pos, dtype=np.float32),
            "vel": np.asarray(all_vel, dtype=np.float32),
            "cur": np.asarray(all_cur, dtype=np.float32),
            "pwm": np.asarray(all_pwm, dtype=np.float32),
            "vin": np.asarray(all_vin, dtype=np.float32),
            "temp": np.asarray(all_temp, dtype=np.float32),
        }

    def get_observation(self, retries: int = 0) -> Obs:
        motor_state = self.controller.get_motor_states(self.controllers, int(retries))
        flat = self._flatten_state(motor_state)
        motor_pos = flat["pos"]
        motor_vel = flat["vel"]
        motor_cur = flat["cur"]
        motor_pwm = flat["pwm"]
        motor_vin = flat["vin"]
        motor_temp = flat["temp"]

        if motor_pos.shape[0] != int(self.model.nu):
            raise ValueError(
                f"G1 motor state length {motor_pos.shape[0]} != model.nu {self.model.nu}"
            )

        self._last_motor_pos = motor_pos.copy()
        self._last_motor_vel = motor_vel.copy()
        self._last_motor_cur = motor_cur.copy()
        self._last_motor_tor = motor_cur.copy()
        self._last_motor_pwm = motor_pwm.copy()
        self._last_motor_vin = motor_vin.copy()
        self._last_motor_temp = motor_temp.copy()

        if self.controllers:
            quat_wxyz, gyro = self.controllers[0].get_imu_state()
            self._last_quat_wxyz = np.asarray(quat_wxyz, dtype=np.float32).reshape(4)
            self._last_gyro = np.asarray(gyro, dtype=np.float32).reshape(3)

        qpos = self._qpos_base.copy()
        qpos[self._qpos_adr] = motor_pos
        if int(self.model.nq) >= 7:
            qpos[3:7] = self._last_quat_wxyz
        qvel = np.zeros(int(self.model.nv), dtype=np.float32)
        qvel[self._qvel_adr] = motor_vel

        obs_time = float(time.monotonic())
        if self.controllers and hasattr(self.controllers[0], "get_latest_sample"):
            _, rx_time = self.controllers[0].get_latest_sample()
            if rx_time > 0.0:
                obs_time = float(rx_time)

        rot = R.from_quat(self._last_quat_wxyz, scalar_first=True)
        return Obs(
            ang_vel=self._last_gyro.copy(),
            time=obs_time,
            motor_pos=motor_pos.copy(),
            motor_vel=motor_vel.copy(),
            motor_tor=self._last_motor_tor.copy(),
            qpos=qpos,
            qvel=qvel,
            rot=rot,
            motor_cur=self._last_motor_cur.copy(),
            motor_drive=np.ones_like(motor_pos, dtype=np.float32),
            motor_pwm=self._last_motor_pwm.copy(),
            motor_vin=self._last_motor_vin.copy(),
            motor_temp=self._last_motor_temp.copy(),
        )

    def set_motor_target(self, motor_angles: Dict[str, float] | np.ndarray) -> None:
        if isinstance(motor_angles, dict):
            if all(name in motor_angles for name in self.motor_ordering):
                target = np.asarray(
                    [motor_angles[name] for name in self.motor_ordering],
                    dtype=np.float32,
                )
            else:
                target = np.asarray(list(motor_angles.values()), dtype=np.float32)
        else:
            target = np.asarray(motor_angles, dtype=np.float32).reshape(-1)
        if target.shape[0] != int(self.model.nu):
            raise ValueError(
                f"motor target len {target.shape[0]} != model.nu {self.model.nu}"
            )
        self.controller.set_motor_pos(self.controllers, [target.tolist()])

    def close(self) -> None:
        try:
            self.controller.close(self.controllers)
        except Exception:
            pass
