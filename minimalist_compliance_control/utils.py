import os
import select
import sys
import termios
import threading
import time
import tty
import warnings
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

import mujoco
import numpy as np
import yaml

AXIS_BINDINGS = {
    "w": (0, +1),
    "x": (0, -1),
    "a": (1, +1),
    "d": (1, -1),
    "q": (2, +1),
    "z": (2, -1),
}
VALID_KEYBOARD_COMMANDS = {"c", "l", "r", "b"}


@dataclass
class MotorParams:
    kp: np.ndarray
    kd: np.ndarray
    tau_max: np.ndarray
    q_dot_max: np.ndarray
    tau_q_dot_max: np.ndarray
    q_dot_tau_max: np.ndarray
    tau_brake_max: np.ndarray
    kd_min: np.ndarray
    passive_active_ratio: float


class KeyboardTeleop:
    def __init__(
        self,
        num_sites: int,
        site_names: Optional[Sequence[str]] = None,
        pos_step: float = 0.01,
        rot_step_deg: float = 5.0,
    ) -> None:
        self.num_sites = int(num_sites)
        if site_names is None:
            self.site_names = [f"site_{i}" for i in range(self.num_sites)]
        else:
            if len(site_names) != self.num_sites:
                raise ValueError(
                    f"site_names length {len(site_names)} must match num_sites {self.num_sites}."
                )
            self.site_names = [str(name) for name in site_names]
        self.pos_step = float(pos_step)
        self.rot_step = np.deg2rad(float(rot_step_deg))
        self._lock = threading.Lock()
        self.active_idx = 0
        self.rotation_mode = False
        self.force_perturbation_enabled = False
        self.pos_offsets = np.zeros((self.num_sites, 3), dtype=np.float32)
        self.rot_offsets = np.zeros((self.num_sites, 3), dtype=np.float32)

    def _print_target(self) -> None:
        idx = self.active_idx
        name = self.site_names[idx]
        x, y, z = self.pos_offsets[idx]
        roll, pitch, yaw = np.rad2deg(self.rot_offsets[idx])
        print(
            f"[teleop] site {idx} ({name}) target -> "
            f"x: {float(x):.3f}, y: {float(y):.3f}, z: {float(z):.3f}, "
            f"roll: {float(roll):.1f} deg, pitch: {float(pitch):.1f} deg, yaw: {float(yaw):.1f} deg"
        )

    def handle_char(self, char: str) -> None:
        c = char.lower()
        with self._lock:
            if c == "p":
                self.rotation_mode = not self.rotation_mode
                mode = "rotation" if self.rotation_mode else "position"
                print(f"[teleop] mode: {mode}")
                return
            if c == "n" and self.num_sites > 1:
                self.active_idx = (self.active_idx + 1) % self.num_sites
                print(f"[teleop] active site index: {self.active_idx}")
                return
            if c == "r":
                self.pos_offsets[self.active_idx, :] = 0.0
                self.rot_offsets[self.active_idx, :] = 0.0
                print(f"[teleop] reset site index: {self.active_idx}")
                return
            if c == "f":
                self.force_perturbation_enabled = not self.force_perturbation_enabled
                state = "ON" if self.force_perturbation_enabled else "OFF"
                print(f"[teleop] random force perturbation: {state}")
                return
            if c not in AXIS_BINDINGS:
                return
            axis_idx, direction = AXIS_BINDINGS[c]
            if self.rotation_mode:
                self.rot_offsets[self.active_idx, axis_idx] += direction * self.rot_step
                self._print_target()
            else:
                self.pos_offsets[self.active_idx, axis_idx] += direction * self.pos_step
                self._print_target()

    def snapshot(self) -> tuple[np.ndarray, np.ndarray, bool]:
        with self._lock:
            return (
                self.pos_offsets.copy(),
                self.rot_offsets.copy(),
                bool(self.force_perturbation_enabled),
            )


class KeyboardListener:
    def __init__(self, teleop: KeyboardTeleop) -> None:
        self.teleop = teleop
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._fd: Optional[int] = None
        self._old_term_settings = None

    def start(self) -> bool:
        if self._thread is not None:
            return True
        if not sys.stdin.isatty():
            warnings.warn(
                "Keyboard teleop disabled: stdin is not a TTY.",
                RuntimeWarning,
                stacklevel=2,
            )
            return False
        try:
            self._fd = sys.stdin.fileno()
            self._old_term_settings = termios.tcgetattr(self._fd)
            tty.setcbreak(self._fd)
        except Exception as exc:
            warnings.warn(
                f"Keyboard teleop disabled: failed to configure terminal input: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            self._fd = None
            self._old_term_settings = None
            return False

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return True

    def _run(self) -> None:
        while not self._stop_event.is_set():
            if self._fd is None:
                return
            try:
                ready, _, _ = select.select([sys.stdin], [], [], 0.05)
            except Exception:
                return
            if not ready:
                continue
            try:
                ch = sys.stdin.read(1)
            except Exception:
                return
            if ch:
                self.teleop.handle_char(ch)

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=0.5)
            self._thread = None
        if self._fd is not None and self._old_term_settings is not None:
            try:
                termios.tcsetattr(
                    self._fd,
                    termios.TCSADRAIN,
                    self._old_term_settings,
                )
            except Exception:
                pass
        self._fd = None
        self._old_term_settings = None


@dataclass
class KeyboardCommand:
    command: str
    recv_time: float


class KeyboardControlReceiver:
    """Non-blocking stdin receiver for single-char keyboard commands."""

    def __init__(self, port: int = 5592) -> None:
        _ = port
        self.enabled = False
        self._fd: Optional[int] = None
        self._old_term_settings = None

        if not sys.stdin.isatty():
            warnings.warn(
                "Model-based keyboard control disabled: stdin is not a TTY.",
                RuntimeWarning,
                stacklevel=2,
            )
            return
        try:
            self._fd = sys.stdin.fileno()
            self._old_term_settings = termios.tcgetattr(self._fd)
            tty.setcbreak(self._fd)
        except Exception as exc:
            warnings.warn(
                f"Model-based keyboard control disabled: failed to configure stdin ({exc}).",
                RuntimeWarning,
                stacklevel=2,
            )
            self._fd = None
            self._old_term_settings = None
            return

        self.enabled = True
        print("[model_based] Keyboard control active on stdin (c/l/r/b).")
        print("[model_based] Focus terminal to send model-based commands.")

    def close(self) -> None:
        if self._fd is not None and self._old_term_settings is not None:
            try:
                termios.tcsetattr(
                    self._fd,
                    termios.TCSADRAIN,
                    self._old_term_settings,
                )
            except Exception:
                pass
        self._fd = None
        self._old_term_settings = None
        self.enabled = False

    def poll_command(self) -> Optional[KeyboardCommand]:
        if not self.enabled or self._fd is None:
            return None
        try:
            ready, _, _ = select.select([self._fd], [], [], 0.0)
        except Exception:
            return None
        if not ready:
            return None
        try:
            raw = os.read(self._fd, 32)
        except Exception:
            return None
        if not raw:
            return None

        cmd: Optional[str] = None
        for ch in raw.decode(errors="ignore").lower():
            c = ch.strip()
            if c in VALID_KEYBOARD_COMMANDS:
                cmd = c
        if cmd is None:
            return None
        return KeyboardCommand(command=cmd, recv_time=time.time())


def deep_update(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_motor_params(
    model: mujoco.MjModel,
    default_config_path: str,
    robot_config_path: str,
    motors_config_path: Optional[str] = None,
) -> MotorParams:
    with open(default_config_path, "r") as f:
        config = yaml.safe_load(f)
    with open(robot_config_path, "r") as f:
        robot_cfg = yaml.safe_load(f)
    if robot_cfg is not None:
        deep_update(config, robot_cfg)
    if motors_config_path:
        with open(motors_config_path, "r") as f:
            motor_cfg = yaml.safe_load(f)
        if motor_cfg is not None:
            deep_update(config, motor_cfg)

    kp_ratio = float(config["actuators"]["kp_ratio"])
    kd_ratio = float(config["actuators"]["kd_ratio"])
    passive_active_ratio = float(config["actuators"]["passive_active_ratio"])

    names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        for i in range(model.nu)
    ]
    kp = []
    kd = []
    tau_max = []
    q_dot_max = []
    tau_q_dot_max = []
    q_dot_tau_max = []
    tau_brake_max = []
    kd_min = []
    for name in names:
        motor_key = name
        if motor_key not in config["motors"] and motor_key.endswith("_act"):
            base_key = motor_key[: -len("_act")]
            if base_key in config["motors"]:
                motor_key = base_key
        if motor_key not in config["motors"]:
            raise ValueError(f"Missing motor config for actuator '{name}'")
        m_cfg = config["motors"][motor_key]
        motor_type = m_cfg["motor"]
        act_cfg = config["actuators"][motor_type]
        kp.append(float(m_cfg.get("kp", 0.0)) / kp_ratio)
        kd.append(float(m_cfg.get("kd", 0.0)) / kd_ratio)
        tau_max.append(float(act_cfg["tau_max"]))
        q_dot_max.append(float(act_cfg["q_dot_max"]))
        tau_q_dot_max.append(float(act_cfg["tau_q_dot_max"]))
        q_dot_tau_max.append(float(act_cfg["q_dot_tau_max"]))
        tau_brake_max.append(float(act_cfg["tau_brake_max"]))
        kd_min.append(float(act_cfg["kd_min"]))

    return MotorParams(
        kp=np.asarray(kp, dtype=np.float32),
        kd=np.asarray(kd, dtype=np.float32),
        tau_max=np.asarray(tau_max, dtype=np.float32),
        q_dot_max=np.asarray(q_dot_max, dtype=np.float32),
        tau_q_dot_max=np.asarray(tau_q_dot_max, dtype=np.float32),
        q_dot_tau_max=np.asarray(q_dot_tau_max, dtype=np.float32),
        tau_brake_max=np.asarray(tau_brake_max, dtype=np.float32),
        kd_min=np.asarray(kd_min, dtype=np.float32),
        passive_active_ratio=passive_active_ratio,
    )


def compute_clamped_motor_torque(
    target_motor_pos: np.ndarray,
    q: np.ndarray,
    q_dot: np.ndarray,
    q_dot_dot: np.ndarray,
    motor_params: MotorParams,
) -> np.ndarray:
    error = target_motor_pos - q
    real_kp = np.where(
        q_dot_dot * error < 0,
        motor_params.kp * motor_params.passive_active_ratio,
        motor_params.kp,
    )
    tau_m = real_kp * error - (motor_params.kd_min + motor_params.kd) * q_dot
    abs_q_dot = np.abs(q_dot)
    slope = (motor_params.tau_q_dot_max - motor_params.tau_max) / (
        motor_params.q_dot_max - motor_params.q_dot_tau_max
    )
    taper_limit = motor_params.tau_max + slope * (
        abs_q_dot - motor_params.q_dot_tau_max
    )
    tau_acc_limit = np.where(
        abs_q_dot <= motor_params.q_dot_tau_max, motor_params.tau_max, taper_limit
    )
    tau_m_clamped = np.where(
        np.logical_and(
            abs_q_dot > motor_params.q_dot_max, q_dot * target_motor_pos > 0
        ),
        np.where(
            q_dot > 0,
            np.ones_like(tau_m) * -motor_params.tau_brake_max,
            np.ones_like(tau_m) * motor_params.tau_brake_max,
        ),
        np.where(
            q_dot > 0,
            np.clip(tau_m, -motor_params.tau_brake_max, tau_acc_limit),
            np.clip(tau_m, -tau_acc_limit, motor_params.tau_brake_max),
        ),
    )
    return tau_m_clamped
