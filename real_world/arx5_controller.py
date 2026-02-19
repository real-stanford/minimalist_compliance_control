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
_DEFAULT_SDK_PATH = Path(
    os.environ.get("ARX5_SDK_PATH", "/home/jgoler/arx5-sdk/python")
)
if _DEFAULT_SDK_PATH.exists():
    sys.path.append(str(_DEFAULT_SDK_PATH))
try:
    import arx5_interface as arx5  # type: ignore
except ImportError as exc:  # pragma: no cover - environment dependent
    raise ImportError(
        "arx5_interface not found. Install the ARX5 SDK Python bindings or set ARX5_SDK_PATH."
    ) from exc


def _discover_ports(port_pattern: str) -> List[str]:
    """Return interface names matching ``port_pattern`` (e.g., ``can[0-9]+``)."""
    pattern = re.compile(port_pattern)
    net_dir = Path("/sys/class/net")
    interfaces = (
        [entry.name for entry in net_dir.iterdir() if pattern.fullmatch(entry.name)]
        if net_dir.exists()
        else []
    )

    # If nothing matched but a literal interface was provided, keep it.
    if not interfaces and port_pattern:
        interfaces = [port_pattern]
    return interfaces


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

    # Dynamixel-like API
    def initialize_motors(self) -> None:
        """Bring the arm to a known state."""
        self.controller.reset_to_home()

    def get_motor_ids(self) -> List[int]:
        return list(range(self.robot_config.joint_dof))

    def get_state(self, retries: int = 0) -> Dict[str, List[float]]:
        """Fetch current joint+gripper state."""
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
        try:
            self.controller.set_to_damping()
        finally:
            # Explicit close hook is not exposed; rely on Python GC.
            pass


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
    model_name = model or os.environ.get("ARX5_MODEL", "X5")
    interfaces = _discover_ports(port_pattern)
    if not interfaces:
        raise RuntimeError(f"No ARX5 interfaces matched pattern '{port_pattern}'")

    robot_config = arx5.RobotConfigFactory.get_instance().get_config(model_name)
    controller_config = arx5.ControllerConfigFactory.get_instance().get_config(
        controller_type, robot_config.joint_dof
    )
    controller_config.background_send_recv = background_send_recv
    if controller_dt is not None:
        controller_config.controller_dt = controller_dt

    controllers: List[ARX5Control] = []
    for interface in interfaces:
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
