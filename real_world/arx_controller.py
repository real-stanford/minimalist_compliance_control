"""
ARX5 controller wrapper that mirrors the :mod:`toddlerbot.actuation.dynamixel_cpp`
surface API. The goal is drop-in compatibility with consumers like
``toddlerbot.sim.real_world.RealWorld`` that expect functions:

- create_controllers(...) -> list of controller objects
- initialize(controllers)
- get_motor_ids(controllers) -> {"controller_0": [...], ...}
- get_motor_states(controllers, retries)
- set_motor_pos/vel/pd(...), disable_motors(...), close(...)

This module binds to the ARX5 SDK Python wrapper (``arx5_interface``). Set
``ARX5_SDK_PATH`` if the SDK is not already on the PYTHONPATH.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np

# Ensure arx5_interface is importable.
sdk_path = os.environ.get("ARX5_SDK_PATH", "").strip()
if sdk_path:
    sdk_dir = Path(sdk_path).expanduser()
    if not sdk_dir.exists():
        raise ImportError(f"ARX5_SDK_PATH is set but does not exist: {sdk_dir}")
    sdk_dir_str = str(sdk_dir)
    if sdk_dir_str not in sys.path:
        sys.path.append(sdk_dir_str)
try:
    import arx5_interface as arx5  # type: ignore
except ImportError as exc:  # pragma: no cover - environment dependent
    raise ImportError(
        "arx5_interface not found. Install the ARX5 SDK Python bindings and make "
        "sure it is importable, or set ARX5_SDK_PATH to the SDK python directory "
        "(e.g. /path/to/arx5-sdk/python)."
    ) from exc


def _discover_ports(port_pattern: str) -> List[str]:
    """Return interface names matching ``port_pattern`` (e.g., ``can[0-9]+``)."""
    try:
        pattern = re.compile(port_pattern)
    except re.error as exc:
        raise RuntimeError(f"Invalid ARX5 interface pattern: {port_pattern!r}") from exc
    net_dir = Path("/sys/class/net")
    all_interfaces = (
        [entry.name for entry in net_dir.iterdir()] if net_dir.exists() else []
    )
    interfaces = [name for name in all_interfaces if pattern.fullmatch(name)]

    # Only treat as literal interface when no regex metacharacters are present.
    is_regex = bool(re.search(r"[.^$*+?{}\[\]|()\\]", port_pattern))
    if (
        not interfaces
        and port_pattern
        and not is_regex
        and port_pattern in all_interfaces
    ):
        interfaces = [port_pattern]
    return interfaces


def _has_ament_index(prefix: str) -> bool:
    if not prefix:
        return False
    return (
        Path(prefix) / "share" / "ament_index" / "resource_index" / "packages"
    ).is_dir()


def _ensure_ament_prefix_path() -> None:
    existing = os.environ.get("AMENT_PREFIX_PATH", "").strip()
    if existing:
        # Accept first valid path in the AMENT_PREFIX_PATH chain.
        for item in existing.split(os.pathsep):
            if _has_ament_index(item.strip()):
                return

    candidates: List[str] = []
    user_hint = os.environ.get("ARX5_AMENT_PREFIX_PATH", "").strip()
    if user_hint:
        candidates.append(user_hint)

    # Active interpreter environment (if running inside conda/venv).
    conda_prefix = os.environ.get("CONDA_PREFIX", "").strip()
    if conda_prefix:
        candidates.append(conda_prefix)

    # Common local ARX env naming convention (e.g., .../envs/arx-py310).
    exe_env = str(Path(sys.executable).resolve().parent.parent)
    candidates.append(exe_env)
    if "/envs/" in exe_env:
        env_root, _ = exe_env.rsplit("/envs/", 1)
        candidates.append(f"{env_root}/envs/arx-py310")

    seen = set()
    for prefix in candidates:
        prefix = prefix.strip()
        if not prefix or prefix in seen:
            continue
        seen.add(prefix)
        if _has_ament_index(prefix):
            os.environ["AMENT_PREFIX_PATH"] = prefix
            return

    raise RuntimeError(
        "ARX5 runtime requires AMENT_PREFIX_PATH to point to a ROS ament prefix. "
        "Could not auto-detect one. Set it explicitly, e.g. "
        "`export AMENT_PREFIX_PATH=/path/to/conda/envs/arx-py310` "
        "or activate the ARX SDK conda env before launching mcc-run-policy."
    )


class ARX5Control:
    """Per-interface ARX5 controller mirroring the DynamixelControl methods."""

    def __init__(
        self,
        interface: str,
        robot_model: "str",
        robot_config: "arx5.RobotConfig",
        controller_config: "arx5.ControllerConfig",
        controller: "arx5.Arx5JointController",
    ):
        self.interface = interface
        self.robot_model = robot_model
        self.robot_config = robot_config
        self.controller_config = controller_config
        self.controller = controller
        self._num_motors: int | None = None

    # Dynamixel-like API
    def initialize_motors(self) -> None:
        """Bring the arm to a known state."""
        self.controller.reset_to_home()

    def get_motor_ids(self) -> List[int]:
        if self._num_motors is None:
            default_num = int(getattr(self.robot_config, "joint_dof", 0))
            motor_ids_cfg = list(getattr(self.robot_config, "motor_id", []))
            if motor_ids_cfg:
                default_num = max(default_num, len(motor_ids_cfg))
            gripper_motor_id = getattr(self.robot_config, "gripper_motor_id", None)
            if gripper_motor_id is not None and gripper_motor_id not in motor_ids_cfg:
                default_num += 1
            try:
                state = self.get_state(retries=0)
                inferred = int(len(state.get("pos", [])))
                self._num_motors = inferred if inferred > 0 else default_num
            except Exception:
                self._num_motors = default_num
        return list(range(self._num_motors))

    def get_state(self, retries: int = 0) -> Dict[str, List[float]]:
        """Fetch current joint+gripper state."""
        del retries
        joint_state = self.controller.get_joint_state()
        arm_pos = np.asarray(joint_state.pos(), dtype=np.float64)
        arm_vel = np.asarray(joint_state.vel(), dtype=np.float64)
        arm_tor = np.asarray(joint_state.torque(), dtype=np.float64)

        gripper_pos = float(getattr(joint_state, "gripper_pos", 0.0))
        gripper_vel = float(getattr(joint_state, "gripper_vel", 0.0))
        gripper_tor = float(getattr(joint_state, "gripper_torque", 0.0))

        pos = (*arm_pos.tolist(), gripper_pos)
        vel = (*arm_vel.tolist(), gripper_vel)
        torque = (*arm_tor.tolist(), gripper_tor)
        zeros = [0.0] * len(pos)

        return {
            "pos": list(pos),
            "vel": list(vel),
            "cur": list(torque),
            "pwm": zeros,
            "vin": zeros,
            "temp": zeros,
        }

    def set_pos(self, pos_vec: Sequence[float]) -> None:
        """Command joint (and optional gripper) positions."""
        dof = self.robot_config.joint_dof
        vec = list(pos_vec)
        if len(vec) < dof:
            vec.extend([0.0] * (dof - len(vec)))
        cmd = arx5.JointState(dof)
        cmd.pos()[:] = np.array(vec[:dof], dtype=np.float64)
        if len(vec) > dof:
            cmd.gripper_pos = float(vec[dof])
        self.controller.set_joint_cmd(cmd)

    def set_vel(self, vel_vec: Sequence[float]) -> None:
        """Command joint (and optional gripper) velocities."""
        dof = self.robot_config.joint_dof
        vec = list(vel_vec)
        if len(vec) < dof:
            vec.extend([0.0] * (dof - len(vec)))
        cmd = arx5.JointState(dof)
        cmd.vel()[:] = np.array(vec[:dof], dtype=np.float64)
        if len(vec) > dof:
            cmd.gripper_vel = float(vec[dof])
        self.controller.set_joint_cmd(cmd)

    def set_pd(self, kp_vec: Sequence[float], kd_vec: Sequence[float]) -> None:
        """Update per-joint and gripper gains."""
        dof = self.robot_config.joint_dof
        kp_arr = np.array(kp_vec, dtype=np.float64)
        kd_arr = np.array(kd_vec, dtype=np.float64)
        gain = self.controller.get_gain()
        if kp_arr.size >= dof:
            gain.kp()[:] = kp_arr[:dof]
        if kd_arr.size >= dof:
            gain.kd()[:] = kd_arr[:dof]
        if kp_arr.size > dof:
            gain.gripper_kp = float(kp_arr[dof])
        if kd_arr.size > dof:
            gain.gripper_kd = float(kd_arr[dof])
        self.controller.set_gain(gain)

    def disable_motors(self) -> None:
        """Switch to damping / torque-off."""
        self.controller.set_to_damping()

    def close_motors(self) -> None:
        """Graceful shutdown."""
        self.controller.set_to_damping()


def create_controllers(
    port_pattern: str,
    kp: Sequence[float],
    kd: Sequence[float],
    ki: Sequence[float],
    zero_pos: Sequence[float],
    control_mode: Sequence[str],
    baudrate: int,
    return_delay: int,
    model: str | None = None,
    controller_type: str = "joint_controller",
    background_send_recv: bool = True,
    controller_dt: float | None = None,
    log_level: "arx5.LogLevel | None" = None,
) -> List[ARX5Control]:
    """Instantiate ARX5 controllers for each matching interface."""
    del kp, kd, ki, zero_pos, control_mode, baudrate, return_delay
    _ensure_ament_prefix_path()
    model_name = model or os.environ.get("ARX5_MODEL", "X5")
    interfaces = _discover_ports(port_pattern)
    if not interfaces:
        net_dir = Path("/sys/class/net")
        all_ifaces = (
            sorted([entry.name for entry in net_dir.iterdir()])
            if net_dir.exists()
            else []
        )
        raise RuntimeError(
            "No ARX5 interfaces matched "
            f"'{port_pattern}'. Detected network interfaces: {all_ifaces}. "
            "Set ARX5_INTERFACE to a real CAN iface (for example 'can0') and ensure it is up "
            "(for example: `sudo ip link set can0 up type can bitrate 1000000`)."
        )

    robot_config = arx5.RobotConfigFactory.get_instance().get_config(model_name)
    controller_config = arx5.ControllerConfigFactory.get_instance().get_config(
        controller_type, robot_config.joint_dof
    )
    controller_config.background_send_recv = background_send_recv
    if controller_dt is not None:
        controller_config.controller_dt = controller_dt

    controllers: List[ARX5Control] = []
    errors: List[str] = []
    for interface in interfaces:
        try:
            controller = arx5.Arx5JointController(
                robot_config, controller_config, interface
            )
            if log_level is not None:
                controller.set_log_level(log_level)
            controllers.append(
                ARX5Control(
                    interface=interface,
                    robot_model=model_name,
                    robot_config=robot_config,
                    controller_config=controller_config,
                    controller=controller,
                )
            )
        except Exception as exc:
            errors.append(f"{interface}: {exc}")

    if not controllers:
        raise RuntimeError(
            "Failed to create ARX5 controller on matched interfaces "
            f"{interfaces}. Errors: {errors}"
        )

    return controllers


def initialize(controllers: Iterable[ARX5Control]) -> None:
    for ctrl in controllers:
        if ctrl:
            ctrl.initialize_motors()


def get_motor_ids(controllers: Sequence[ARX5Control]) -> Dict[str, List[int]]:
    return {
        f"controller_{i}": ctrl.get_motor_ids() for i, ctrl in enumerate(controllers)
    }


def get_motor_states(
    controllers: Sequence[ARX5Control], retries: int = 0
) -> Dict[str, Dict[str, List[float]]]:
    empty = {"pos": [], "vel": [], "cur": [], "pwm": [], "vin": [], "temp": []}
    states: Dict[str, Dict[str, List[float]]] = {}
    for i, ctrl in enumerate(controllers):
        key = f"controller_{i}"
        try:
            states[key] = ctrl.get_state(retries)
        except Exception:
            states[key] = dict(empty)
    return states


def set_motor_pos(
    controllers: Sequence[ARX5Control], pos_vecs: Sequence[Sequence[float]]
) -> None:
    for ctrl, pos in zip(controllers, pos_vecs):
        ctrl.set_pos(pos)


def set_motor_vel(
    controllers: Sequence[ARX5Control], vel_vecs: Sequence[Sequence[float]]
) -> None:
    for ctrl, vel in zip(controllers, vel_vecs):
        ctrl.set_vel(vel)


def set_motor_pd(
    controllers: Sequence[ARX5Control],
    kp_vecs: Sequence[Sequence[float]],
    kd_vecs: Sequence[Sequence[float]],
) -> None:
    for ctrl, kp_vec, kd_vec in zip(controllers, kp_vecs, kd_vecs):
        ctrl.set_pd(kp_vec, kd_vec)


def disable_motors(controllers: Sequence[ARX5Control]) -> None:
    for ctrl in controllers:
        ctrl.disable_motors()


def close(controllers: Sequence[ARX5Control]) -> None:
    """Disable torque and close all specified controllers."""
    for ctrl in controllers:
        try:
            ctrl.disable_motors()
        finally:
            ctrl.close_motors()
