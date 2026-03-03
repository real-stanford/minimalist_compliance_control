"""Shared observation contract and minimal sim interface."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    from typing import Protocol, runtime_checkable
except ImportError:  # pragma: no cover
    from typing_extensions import Protocol, runtime_checkable  # type: ignore


@dataclass
class Obs:
    ang_vel: np.ndarray
    time: float = 0.0
    motor_pos: np.ndarray | None = None
    motor_vel: np.ndarray | None = None
    motor_acc: np.ndarray | None = None
    motor_tor: np.ndarray | None = None
    qpos: np.ndarray | None = None
    qvel: np.ndarray | None = None
    pos: np.ndarray | None = None
    rot: Any = None
    motor_cur: np.ndarray | None = None
    motor_drive: np.ndarray | None = None
    motor_pwm: np.ndarray | None = None
    motor_vin: np.ndarray | None = None
    motor_temp: np.ndarray | None = None


@runtime_checkable
class BaseSim(Protocol):
    model: Any
    data: Any

    def set_motor_target(self, motor_target: np.ndarray) -> None: ...

    def step(self) -> None: ...

    def get_observation(self) -> Obs: ...

    def sync(self) -> bool: ...

    def close(self) -> None: ...
