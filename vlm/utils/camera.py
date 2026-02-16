"""Minimal camera wrapper for standalone VLM policy."""

from __future__ import annotations

import os
from typing import Optional

import numpy as np

try:
    import cv2
except ModuleNotFoundError as exc:  # pragma: no cover
    cv2 = None  # type: ignore[assignment]
    _CV2_IMPORT_ERROR = exc
else:
    _CV2_IMPORT_ERROR = None


class Camera:
    """Lightweight camera capture class with left/right side selection.

    Device selection order:
    1) explicit `device` argument
    2) env var `VLM_LEFT_CAMERA_ID` / `VLM_RIGHT_CAMERA_ID`
    3) default: 0 for left, 1 for right
    """

    def __init__(
        self,
        side: str,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        device: Optional[int] = None,
    ) -> None:
        if cv2 is None:
            raise ModuleNotFoundError(
                "opencv-python is required for Camera support"
            ) from _CV2_IMPORT_ERROR

        self.side = str(side).strip().lower()
        if self.side not in ("left", "right"):
            raise ValueError("side must be 'left' or 'right'")

        if device is None:
            env_name = (
                "VLM_LEFT_CAMERA_ID" if self.side == "left" else "VLM_RIGHT_CAMERA_ID"
            )
            env_val = os.environ.get(env_name)
            if env_val is not None and env_val != "":
                device = int(env_val)
            else:
                device = 0 if self.side == "left" else 1

        self.device = int(device)
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)

        self.cap = cv2.VideoCapture(self.device)
        if not self.cap.isOpened():
            raise RuntimeError(
                f"Failed to open camera device {self.device} ({self.side})."
            )

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(self.width))
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self.height))
        self.cap.set(cv2.CAP_PROP_FPS, float(self.fps))

    def get_frame(self) -> np.ndarray:
        ok, frame = self.cap.read()
        if not ok or frame is None:
            raise RuntimeError(
                f"Failed to capture frame from camera {self.device} ({self.side})."
            )
        return frame

    def close(self) -> None:
        if hasattr(self, "cap") and self.cap is not None:
            self.cap.release()
            self.cap = None

    def __del__(self) -> None:
        self.close()
