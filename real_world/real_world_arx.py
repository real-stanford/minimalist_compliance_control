"""Real-world ARX backend used by policy/run_policy.py."""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from real_world.real_world_dynamixel import RealWorldDynamixel


class RealWorldARX(RealWorldDynamixel):
    """ARX backend with arx5 controller wiring and torque-native current signals."""

    def __init__(
        self,
        robot: str,
        control_dt: float,
        xml_path: str,
        merged_config: dict[str, Any] | None = None,
    ) -> None:
        if "arx" not in str(robot).lower():
            raise ValueError("RealWorldARX requires an arx robot name.")
        super().__init__(
            robot=robot,
            control_dt=control_dt,
            xml_path=xml_path,
            merged_config=merged_config,
        )

    def _load_controller_backend(self) -> tuple[Any, str, int]:
        from real_world.dynamixel import arx5_controller as controller

        return controller, os.getenv("ARX5_INTERFACE", "can[0-9]+"), 0

    def _estimate_motor_torque_inputs(
        self,
        motor_cur_arr: np.ndarray,
        motor_vel_arr: np.ndarray,
        motor_pwm_arr: np.ndarray,
        motor_vin_arr: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        del motor_vel_arr, motor_pwm_arr, motor_vin_arr
        gain_back = np.full_like(motor_cur_arr, float(self.gain_backdrive))
        i_est = motor_cur_arr.copy()
        tau_est = motor_cur_arr.copy()
        return gain_back, i_est, tau_est
