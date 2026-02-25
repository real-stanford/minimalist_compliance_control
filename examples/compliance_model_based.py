from __future__ import annotations

from typing import Any

import numpy as np

from examples.compliance_model_based_leap import LeapModelBasedPolicy
from examples.compliance_model_based_toddlerbot import ToddlerbotModelBasedPolicy


class ModelBasedPolicy:
    def __init__(
        self,
        *,
        robot: str = "toddlerbot",
        sim: str = "mujoco",
        vis: bool = True,
        scene_xml: str = "",
        duration: float = 120.0,
        control_dt: float = 0.02,
        prep_duration: float = 7.0,
        status_interval: float = 1.0,
    ) -> None:
        if str(sim) != "mujoco":
            raise ValueError(
                "compliance_model_based currently supports only --sim mujoco"
            )

        robot_name = str(robot).strip().lower()
        if robot_name == "leap":
            self.impl = LeapModelBasedPolicy(
                scene_xml=str(scene_xml),
                control_dt=float(control_dt),
                prep_duration=float(prep_duration),
                vis=bool(vis),
            )
        elif robot_name == "toddlerbot":
            self.impl = ToddlerbotModelBasedPolicy(vis=bool(vis))
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
