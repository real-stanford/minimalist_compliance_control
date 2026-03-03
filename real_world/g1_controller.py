"""Unitree G1 controller wrapper with a Dynamixel-like surface API.

This module mirrors the functions used by real_world.real_world_dynamixel:
- create_controllers(...)
- initialize(controllers)
- get_motor_ids(controllers)
- get_motor_states(controllers, retries)
- set_motor_pos/vel/cur/pwm(...)
- close(controllers)
"""

from __future__ import annotations

import os
import time
from threading import Event, Lock, Thread
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import (  # type: ignore
    MotionSwitcherClient,
)
from unitree_sdk2py.core.channel import (  # type: ignore
    ChannelFactoryInitialize,
    ChannelPublisher,
    ChannelSubscriber,
)
from unitree_sdk2py.idl.default import (  # type: ignore
    unitree_hg_msg_dds__LowCmd_,
    unitree_hg_msg_dds__LowState_,
)
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as LowCmdHG  # type: ignore
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import (  # type: ignore
    LowState_ as LowStateHG,
)
from unitree_sdk2py.utils.crc import CRC  # type: ignore

# Unitree HG low-level body joint order (29 DoF, no dexterous hands).
G1_BODY_JOINT_NAMES: Tuple[str, ...] = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)

_DDS_INDEX_BY_JOINT = {name: idx for idx, name in enumerate(G1_BODY_JOINT_NAMES)}


def _pick_default_interface() -> str:
    """Pick an UP + multicast-capable NIC, avoiding loopback/virtual names."""
    net_root = "/sys/class/net"
    try:
        names = sorted(os.listdir(net_root))
    except Exception:
        return "lo"

    blacklist_prefix = ("lo", "docker", "veth", "br-", "virbr", "zt")
    preferred: list[str] = []
    fallback: list[str] = []

    for name in names:
        if name.startswith(blacklist_prefix):
            continue
        flags_path = os.path.join(net_root, name, "flags")
        try:
            with open(flags_path, "r", encoding="utf-8") as f:
                raw = f.read().strip()
            flags = int(raw, 16)
        except Exception:
            continue
        is_up = bool(flags & 0x1)  # IFF_UP
        has_multicast = bool(flags & 0x1000)  # IFF_MULTICAST
        if not is_up:
            continue
        if has_multicast:
            preferred.append(name)
        else:
            fallback.append(name)

    if preferred:
        return preferred[0]
    if fallback:
        return fallback[0]
    return "lo"


def _first_attr(obj: object, names: Sequence[str], default: float = 0.0) -> float:
    for name in names:
        if hasattr(obj, name):
            try:
                return float(getattr(obj, name))
            except Exception:
                return float(default)
    return float(default)


def _set_dq_field(cmd: object, value: float) -> None:
    if hasattr(cmd, "dq"):
        cmd.dq = float(value)
    elif hasattr(cmd, "qd"):
        cmd.qd = float(value)


def _init_low_cmd(low_cmd: LowCmdHG, mode_machine: int) -> None:
    if hasattr(low_cmd, "mode_machine"):
        low_cmd.mode_machine = int(mode_machine)
    if hasattr(low_cmd, "mode_pr"):
        low_cmd.mode_pr = 0
    for i in range(len(low_cmd.motor_cmd)):
        motor_cmd = low_cmd.motor_cmd[i]
        if hasattr(motor_cmd, "mode"):
            motor_cmd.mode = 1
        motor_cmd.q = 0.0
        _set_dq_field(motor_cmd, 0.0)
        motor_cmd.kp = 0.0
        motor_cmd.kd = 0.0
        motor_cmd.tau = 0.0


class G1Control:
    """Single-NIC G1 low-level controller."""

    def __init__(
        self,
        interface: str,
        actuator_names: Sequence[str],
        kp: Sequence[float],
        kd: Sequence[float],
        control_dt: float = 0.02,
    ) -> None:
        self.interface = str(interface)
        self.actuator_names = [str(name) for name in actuator_names]
        self.actuator_name_to_index = {
            name: idx for idx, name in enumerate(self.actuator_names)
        }
        self._actuator_to_dds: list[tuple[int, int]] = []
        for joint_name, dds_idx in _DDS_INDEX_BY_JOINT.items():
            act_idx = self.actuator_name_to_index.get(joint_name)
            if act_idx is not None:
                self._actuator_to_dds.append((act_idx, dds_idx))
        if not self._actuator_to_dds:
            max_idx = min(len(self.actuator_names), len(G1_BODY_JOINT_NAMES))
            self._actuator_to_dds = [(i, i) for i in range(max_idx)]

        self.control_dt = float(control_dt)
        self.lock = Lock()
        self._tx_dt = 0.002
        self._tx_stop = Event()
        self._tx_thread: Optional[Thread] = None

        n = len(self.actuator_names)
        self.kp = np.zeros(n, dtype=np.float32)
        self.kd = np.zeros(n, dtype=np.float32)
        kp_arr = np.asarray(kp, dtype=np.float32).reshape(-1)
        kd_arr = np.asarray(kd, dtype=np.float32).reshape(-1)
        self.kp[: min(n, kp_arr.size)] = kp_arr[: min(n, kp_arr.size)]
        self.kd[: min(n, kd_arr.size)] = kd_arr[: min(n, kd_arr.size)]
        if np.allclose(self.kp, 0.0):
            self.kp.fill(40.0)
        if np.allclose(self.kd, 0.0):
            self.kd.fill(2.0)

        self.low_cmd = unitree_hg_msg_dds__LowCmd_()
        self.low_state = unitree_hg_msg_dds__LowState_()
        self.mode_machine = 0
        self._connected = False
        self._initialized = False
        self._last_state_tick = 0
        self._last_state_rx_time = 0.0

        self._last_pos = np.zeros(n, dtype=np.float32)
        self._last_vel = np.zeros(n, dtype=np.float32)
        self._last_cur = np.zeros(n, dtype=np.float32)
        self._last_quat_wxyz = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self._last_gyro = np.zeros(3, dtype=np.float32)

        self.lowcmd_publisher_: Optional[ChannelPublisher] = None
        self.low_state_subscriber_: Optional[ChannelSubscriber] = None
        self._motion_switcher: Optional[MotionSwitcherClient] = None

    def initialize_motors(self) -> None:
        ChannelFactoryInitialize(0, self.interface)
        self._release_active_mode()
        self.lowcmd_publisher_ = ChannelPublisher("rt/lowcmd", LowCmdHG)
        self.low_state_subscriber_ = ChannelSubscriber("rt/lowstate", LowStateHG)
        self.lowcmd_publisher_.Init()
        self.low_state_subscriber_.Init(self._low_state_handler, 10)
        self._wait_for_low_state()
        _init_low_cmd(self.low_cmd, self.mode_machine)
        self._start_tx_thread()
        self._initialized = True
        self._refresh_cache()

    def _release_active_mode(self) -> None:
        try:
            self._motion_switcher = MotionSwitcherClient()
            self._motion_switcher.SetTimeout(5.0)
            self._motion_switcher.Init()
            for _ in range(6):
                status, result = self._motion_switcher.CheckMode()
                if status != 0 or not isinstance(result, dict):
                    break
                active_name = str(result.get("name", "")).strip()
                if not active_name:
                    break
                self._motion_switcher.ReleaseMode()
                time.sleep(0.2)
        except Exception:
            self._motion_switcher = None

    def _low_state_handler(self, msg: LowStateHG) -> None:
        now = float(time.monotonic())
        with self.lock:
            self.low_state = msg
            if hasattr(msg, "mode_machine"):
                self.mode_machine = int(msg.mode_machine)
            self._last_state_tick = int(getattr(msg, "tick", self._last_state_tick))
            self._last_state_rx_time = now

    def _start_tx_thread(self) -> None:
        if self._tx_thread is not None and self._tx_thread.is_alive():
            return
        self._tx_stop.clear()
        self._tx_thread = Thread(target=self._tx_loop, name="g1_lowcmd_tx", daemon=True)
        self._tx_thread.start()

    def _stop_tx_thread(self) -> None:
        self._tx_stop.set()
        if self._tx_thread is not None and self._tx_thread.is_alive():
            self._tx_thread.join(timeout=1.0)
        self._tx_thread = None

    def _tx_loop(self) -> None:
        while not self._tx_stop.is_set():
            if self.lowcmd_publisher_ is not None:
                with self.lock:
                    self.low_cmd.mode_machine = int(self.mode_machine)
                    self.low_cmd.crc = CRC().Crc(self.low_cmd)
                    self.lowcmd_publisher_.Write(self.low_cmd)
            time.sleep(self._tx_dt)

    def _wait_for_low_state(self, timeout_s: float = 10.0) -> None:
        start = time.monotonic()
        while True:
            with self.lock:
                tick = int(getattr(self.low_state, "tick", 0))
            if tick > 0:
                self._connected = True
                return
            if time.monotonic() - start > float(timeout_s):
                raise TimeoutError(
                    "Timed out waiting for G1 rt/lowstate stream on interface "
                    f"'{self.interface}'. Try --ip <nic> (e.g., eth0/enp*), or set "
                    "G1_NET_IFACE, and confirm the Unitree low-level service is running."
                )
            time.sleep(self.control_dt)

    def get_latest_sample(self) -> tuple[int, float]:
        """Return latest received lowstate tick and local monotonic receive time."""
        with self.lock:
            return int(self._last_state_tick), float(self._last_state_rx_time)

    def _send_cmd(self) -> None:
        if self.lowcmd_publisher_ is None:
            raise RuntimeError("G1 publisher is not initialized.")
        with self.lock:
            self.low_cmd.mode_machine = int(self.mode_machine)
            self.low_cmd.crc = CRC().Crc(self.low_cmd)
            self.lowcmd_publisher_.Write(self.low_cmd)

    def _refresh_cache(self) -> None:
        pos, vel, cur = self._read_motor_arrays()
        self._last_pos[:] = pos
        self._last_vel[:] = vel
        self._last_cur[:] = cur
        self._last_quat_wxyz, self._last_gyro = self._read_imu()

    def _read_motor_arrays(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = len(self.actuator_names)
        pos = np.asarray(self._last_pos, dtype=np.float32).copy()
        vel = np.asarray(self._last_vel, dtype=np.float32).copy()
        cur = np.asarray(self._last_cur, dtype=np.float32).copy()

        with self.lock:
            low_state = self.low_state
            motor_state = getattr(low_state, "motor_state", [])
            for act_idx, dds_idx in self._actuator_to_dds:
                if dds_idx >= len(motor_state):
                    continue
                state_i = motor_state[dds_idx]
                pos[act_idx] = _first_attr(state_i, ("q",), default=pos[act_idx])
                vel[act_idx] = _first_attr(state_i, ("dq", "qd"), default=0.0)
                cur[act_idx] = _first_attr(
                    state_i, ("tau_est", "tauEst", "tau"), default=0.0
                )

        if pos.shape[0] != n or vel.shape[0] != n or cur.shape[0] != n:
            raise RuntimeError("Internal G1 state buffer size mismatch.")
        return pos, vel, cur

    def _read_imu(self) -> tuple[np.ndarray, np.ndarray]:
        quat = np.asarray(self._last_quat_wxyz, dtype=np.float32).copy()
        gyro = np.asarray(self._last_gyro, dtype=np.float32).copy()
        with self.lock:
            imu_state = getattr(self.low_state, "imu_state", None)
            if imu_state is None:
                return quat, gyro
            quat_raw = np.asarray(
                getattr(imu_state, "quaternion", [1.0, 0.0, 0.0, 0.0]),
                dtype=np.float32,
            ).reshape(-1)
            gyro_raw = np.asarray(
                getattr(imu_state, "gyroscope", [0.0, 0.0, 0.0]), dtype=np.float32
            ).reshape(-1)
        if quat_raw.size >= 4:
            quat = quat_raw[:4].astype(np.float32)
        if gyro_raw.size >= 3:
            gyro = gyro_raw[:3].astype(np.float32)
        return quat, gyro

    def get_imu_state(self) -> tuple[np.ndarray, np.ndarray]:
        self._last_quat_wxyz, self._last_gyro = self._read_imu()
        return (
            np.asarray(self._last_quat_wxyz, dtype=np.float32).copy(),
            np.asarray(self._last_gyro, dtype=np.float32).copy(),
        )

    def get_motor_ids(self) -> List[int]:
        return list(range(len(self.actuator_names)))

    def get_state(self, retries: int = 0) -> Dict[str, List[float]]:
        if not self._initialized:
            raise RuntimeError("G1 controller not initialized.")

        self._refresh_cache()
        zeros = np.zeros(len(self.actuator_names), dtype=np.float32)
        return {
            "pos": self._last_pos.astype(np.float32).tolist(),
            "vel": self._last_vel.astype(np.float32).tolist(),
            "cur": self._last_cur.astype(np.float32).tolist(),
            "pwm": zeros.tolist(),
            "vin": zeros.tolist(),
            "temp": zeros.tolist(),
        }

    def set_pos(self, pos_vec: Sequence[float]) -> None:
        if not self._initialized:
            raise RuntimeError("G1 controller not initialized.")

        vec = np.asarray(pos_vec, dtype=np.float32).reshape(-1)
        n = len(self.actuator_names)
        if vec.size < n:
            padded = np.asarray(self._last_pos, dtype=np.float32).copy()
            padded[: vec.size] = vec
            vec = padded
        elif vec.size > n:
            vec = vec[:n]

        with self.lock:
            if hasattr(self.low_cmd, "mode_pr"):
                self.low_cmd.mode_pr = 0
            for act_idx, dds_idx in self._actuator_to_dds:
                if dds_idx >= len(self.low_cmd.motor_cmd):
                    continue
                motor_cmd = self.low_cmd.motor_cmd[dds_idx]
                if hasattr(motor_cmd, "mode"):
                    motor_cmd.mode = 1
                motor_cmd.q = float(vec[act_idx])
                _set_dq_field(motor_cmd, 0.0)
                motor_cmd.kp = float(self.kp[act_idx])
                motor_cmd.kd = float(self.kd[act_idx])
                motor_cmd.tau = 0.0

        self._send_cmd()
        self._last_pos[:] = vec.astype(np.float32)

    def set_vel(self, vel_vec: Sequence[float]) -> None:
        pass

    def set_pd(self, kp_vec: Sequence[float], kd_vec: Sequence[float]) -> None:
        kp_arr = np.asarray(kp_vec, dtype=np.float32).reshape(-1)
        kd_arr = np.asarray(kd_vec, dtype=np.float32).reshape(-1)
        n = len(self.actuator_names)
        self.kp[: min(n, kp_arr.size)] = kp_arr[: min(n, kp_arr.size)]
        self.kd[: min(n, kd_arr.size)] = kd_arr[: min(n, kd_arr.size)]

    def disable_motors(self) -> None:
        if not self._initialized:
            return
        with self.lock:
            motor_state = getattr(self.low_state, "motor_state", [])
            for dds_idx in range(min(len(self.low_cmd.motor_cmd), len(motor_state))):
                state_i = motor_state[dds_idx]
                motor_cmd = self.low_cmd.motor_cmd[dds_idx]
                if hasattr(motor_cmd, "mode"):
                    motor_cmd.mode = 0
                motor_cmd.q = _first_attr(state_i, ("q",), default=0.0)
                _set_dq_field(motor_cmd, 0.0)
                motor_cmd.kp = 0.0
                motor_cmd.kd = 2.0
                motor_cmd.tau = 0.0
        self._send_cmd()

    def close_motors(self) -> None:
        try:
            self.disable_motors()
        except Exception:
            pass
        self._stop_tx_thread()


def create_controllers(
    port_pattern: str,
    kp: Sequence[float],
    kd: Sequence[float],
    ki: Sequence[float] | None = None,
    zero_pos: Sequence[float] | None = None,
    control_mode: Sequence[str] | None = None,
    baudrate: int = 0,
    return_delay: int = 0,
    actuator_names: Sequence[str] | None = None,
    control_dt: float = 0.02,
) -> List[G1Control]:
    interface = (
        str(port_pattern).strip() or str(os.environ.get("G1_NET_IFACE", "")).strip()
    )
    if not interface:
        interface = _pick_default_interface()
    names = (
        [str(name) for name in actuator_names]
        if actuator_names is not None
        else list(G1_BODY_JOINT_NAMES)
    )
    ctrl = G1Control(
        interface=interface,
        actuator_names=names,
        kp=kp,
        kd=kd,
        control_dt=float(control_dt),
    )
    return [ctrl]


def initialize(controllers: Iterable[G1Control]) -> None:
    for ctrl in controllers:
        if ctrl:
            ctrl.initialize_motors()


def get_motor_ids(controllers: Sequence[G1Control]) -> Dict[str, List[int]]:
    return {
        f"controller_{i}": ctrl.get_motor_ids() for i, ctrl in enumerate(controllers)
    }


def get_motor_states(
    controllers: Sequence[G1Control], retries: int = 0
) -> Dict[str, Dict[str, List[float]]]:
    empty = {"pos": [], "vel": [], "cur": [], "pwm": [], "vin": [], "temp": []}
    out: Dict[str, Dict[str, List[float]]] = {}
    for i, ctrl in enumerate(controllers):
        key = f"controller_{i}"
        try:
            out[key] = ctrl.get_state()
        except Exception:
            out[key] = dict(empty)
    return out


def get_motor_current_limits(
    controllers: Sequence[G1Control], retries: int = 0
) -> Dict[str, List[float]]:
    return {
        f"controller_{i}": [0.0] * len(ctrl.actuator_names)
        for i, ctrl in enumerate(controllers)
    }


def set_motor_pos(
    controllers: Sequence[G1Control], pos_vecs: Sequence[Sequence[float]]
) -> None:
    for ctrl, pos in zip(controllers, pos_vecs):
        ctrl.set_pos(pos)


def set_motor_vel(
    controllers: Sequence[G1Control], vel_vecs: Sequence[Sequence[float]]
) -> None:
    for ctrl, vel in zip(controllers, vel_vecs):
        ctrl.set_vel(vel)


def set_motor_pd(
    controllers: Sequence[G1Control],
    kp_vecs: Sequence[Sequence[float]],
    kd_vecs: Sequence[Sequence[float]],
) -> None:
    for ctrl, kp_vec, kd_vec in zip(controllers, kp_vecs, kd_vecs):
        ctrl.set_pd(kp_vec, kd_vec)


def set_motor_cur(
    controllers: Sequence[G1Control], cur_vecs: Sequence[Sequence[float]]
) -> None:
    pass


def set_motor_pwm(
    controllers: Sequence[G1Control], pwm_vecs: Sequence[Sequence[float]]
) -> None:
    pass


def set_motor_control_mode(
    controllers: Sequence[G1Control], mode_vecs: Sequence[Sequence[str]]
) -> None:
    pass


def disable_motors(controllers: Sequence[G1Control]) -> None:
    for ctrl in controllers:
        ctrl.disable_motors()


def close(controllers: Sequence[G1Control]) -> None:
    for ctrl in controllers:
        try:
            ctrl.disable_motors()
        finally:
            ctrl.close_motors()
