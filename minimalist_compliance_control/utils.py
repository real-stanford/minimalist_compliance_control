import os
import select
import sys
import termios
import threading
import time
import tty
import warnings
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np
import numpy.typing as npt

AXIS_BINDINGS = {
    "w": (0, +1),
    "x": (0, -1),
    "a": (1, +1),
    "d": (1, -1),
    "q": (2, +1),
    "z": (2, -1),
}
VALID_KEYBOARD_COMMANDS = {"c", "l", "r", "b"}


def _symmetrize(matrix: npt.ArrayLike) -> npt.NDArray[np.float32]:
    arr = np.asarray(matrix, dtype=np.float32)
    return (0.5 * (arr + np.swapaxes(arr, -1, -2))).astype(np.float32)


def _matrix_sqrt(matrix: npt.ArrayLike) -> npt.NDArray[np.float32]:
    sym = _symmetrize(matrix)
    eigvals, eigvecs = np.linalg.eigh(sym)
    eigvals_clipped = np.clip(eigvals, 0.0, None)
    sqrt_vals = np.sqrt(eigvals_clipped)[..., None, :]
    scaled_vecs = eigvecs * sqrt_vals
    sqrt_matrix = np.matmul(scaled_vecs, np.swapaxes(eigvecs, -1, -2))
    return _symmetrize(sqrt_matrix)


def ensure_matrix(
    value: npt.ArrayLike | float | Iterable[float],
) -> npt.NDArray[np.float32]:
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim == 0:
        return np.eye(3, dtype=np.float32) * float(arr)
    if arr.ndim == 1:
        if arr.shape[0] != 3:
            raise ValueError("Gain vectors must have length 3.")
        return np.diag(arr.astype(np.float32))
    if arr.ndim >= 2:
        if arr.shape[-2:] != (3, 3):
            raise ValueError("Gain matrices must have trailing shape (3, 3).")
        return arr.astype(np.float32)
    raise ValueError("Unsupported gain array shape.")


def get_damping_matrix(
    stiffness: npt.ArrayLike,
    inertia_like: npt.ArrayLike | float | Iterable[float],
) -> npt.NDArray[np.float32]:
    stiffness_matrix = ensure_matrix(stiffness)
    inertia_matrix = ensure_matrix(inertia_like)
    mass_sqrt = _matrix_sqrt(inertia_matrix)
    stiffness_sqrt = _matrix_sqrt(stiffness_matrix)
    damping = 2.0 * np.matmul(mass_sqrt, stiffness_sqrt)
    return _symmetrize(damping).astype(np.float32)


def _interpolate_linear(
    p_start: npt.NDArray[np.float32],
    p_end: npt.NDArray[np.float32],
    duration: float,
    t: float,
) -> npt.NDArray[np.float32]:
    if t <= 0.0:
        return p_start
    if t >= duration:
        return p_end
    return p_start + (p_end - p_start) * (t / duration)


def interpolate_action(
    t: float,
    time_arr: npt.NDArray[np.float32],
    action_arr: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    if t <= float(time_arr[0]):
        return np.asarray(action_arr[0], dtype=np.float32)
    if t >= float(time_arr[-1]):
        return np.asarray(action_arr[-1], dtype=np.float32)

    idx = int(np.searchsorted(time_arr, t, side="right") - 1)
    idx = max(0, min(idx, len(time_arr) - 2))
    p_start = np.asarray(action_arr[idx], dtype=np.float32)
    p_end = np.asarray(action_arr[idx + 1], dtype=np.float32)
    duration = float(time_arr[idx + 1] - time_arr[idx])
    return _interpolate_linear(
        p_start, p_end, max(duration, 1e-6), t - float(time_arr[idx])
    )


def get_action_traj(
    start_time: float,
    start_action: npt.ArrayLike,
    end_action: npt.ArrayLike,
    duration: float,
    dt: float,
    end_time: float = 0.0,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    traj_duration = float(max(duration, 0.0))
    traj_dt = float(max(dt, 1e-6))
    n_steps = max(int(traj_duration / traj_dt), 2)
    traj_time = np.linspace(
        float(start_time),
        float(start_time) + traj_duration,
        n_steps,
        endpoint=True,
        dtype=np.float32,
    )

    action_start = np.asarray(start_action, dtype=np.float32).reshape(-1)
    action_end = np.asarray(end_action, dtype=np.float32).reshape(-1)
    traj_action = np.zeros(
        (traj_time.shape[0], action_start.shape[0]), dtype=np.float32
    )

    hold_time = float(np.clip(end_time, 0.0, traj_duration))
    blend_duration = max(traj_duration - hold_time, 0.0)
    for i, t_now in enumerate(traj_time):
        t_rel = float(t_now - start_time)
        if t_rel < blend_duration:
            traj_action[i] = _interpolate_linear(
                action_start, action_end, max(blend_duration, 1e-6), t_rel
            )
        else:
            traj_action[i] = action_end
    return traj_time, traj_action


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
