from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from policy.compliance_model_based_leap import LeapModelBasedPolicy
from policy.compliance_model_based_toddlerbot import ToddlerbotModelBasedPolicy


class ModelBasedPolicy:
    def __init__(
        self,
        name: str,
        robot: str,
        init_motor_pos: npt.ArrayLike,
        *,
        vis: bool = True,
        scene_xml: str = "",
        control_dt: float = 0.02,
        prep_duration: float = 7.0,
    ) -> None:
        robot_name = str(robot).strip().lower()
        self.name = str(name)
        self.robot = robot_name
        if robot_name == "leap":
            self.impl = LeapModelBasedPolicy(
                name=self.name,
                robot=self.robot,
                init_motor_pos=init_motor_pos,
                scene_xml=str(scene_xml),
                control_dt=float(control_dt),
                prep_duration=float(prep_duration),
                vis=bool(vis),
            )
        elif robot_name == "toddlerbot":
            self.impl = ToddlerbotModelBasedPolicy(
                name=self.name,
                robot=self.robot,
                init_motor_pos=init_motor_pos,
                vis=bool(vis),
            )
        else:
            raise ValueError(
                f"Unsupported robot for compliance_model_based: {robot}. "
                "Expected one of: toddlerbot, leap"
            )

        self.control_dt = float(getattr(self.impl, "control_dt", control_dt))
        self.done = bool(getattr(self.impl, "done", False))

    def step(self, obs: Any, sim: Any) -> np.ndarray:
        out = self.impl.step(obs, sim)
        self.done = bool(getattr(self.impl, "done", False))
        return out

    def close(self, exp_folder_path: str = "") -> None:
        close_fn = getattr(self.impl, "close", None)
        if callable(close_fn):
            try:
                close_fn(exp_folder_path=exp_folder_path)
            except TypeError:
                close_fn()
