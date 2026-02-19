"""Camera interface module for stereo capture and streaming."""

import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import List, Optional, Set

import cv2
import numpy as np
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]
_ASSETS_DIR = _REPO_ROOT / "assets"


def normalize_robot_name(robot: str) -> str:
    """Map broad robot names to camera config keys."""
    name = str(robot).strip().lower()
    if "leap" in name:
        return "leap"
    if "toddlerbot" in name:
        return "toddlerbot"
    return name


def resolve_camera_config_path(
    robot: str, config_override: Optional[str] = None
) -> Path:
    if config_override:
        return Path(config_override)
    return _ASSETS_DIR / f"{normalize_robot_name(robot)}_camera.yml"


def load_robot_camera_config(
    robot: str, config_override: Optional[str] = None
) -> tuple[dict, Path]:
    config_path = resolve_camera_config_path(robot, config_override=config_override)
    if not config_path.exists():
        print(
            f"Camera config not found: {config_path} (using device defaults)",
            file=sys.stderr,
        )
        return {}, config_path

    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Camera config must be a mapping: {config_path}")
    return data, config_path


def load_camera_params(robot: str) -> dict:
    data, config_path = load_robot_camera_config(
        robot, config_override=os.environ.get("MCC_CAMERA_CONFIG")
    )
    if not data:
        return {}

    controls_data = data.get("camera_controls", data)
    if not isinstance(controls_data, dict):
        raise ValueError(f"camera_controls must be a mapping: {config_path}")

    params: dict = {}
    for side in ("left", "right"):
        if side not in controls_data:
            continue
        side_cfg = controls_data.get(side, {})
        if not isinstance(side_cfg, dict):
            raise ValueError(
                f"Camera config missing mapping for '{side}' in {config_path}"
            )
        brightness = side_cfg.get("brightness")
        contrast = side_cfg.get("contrast")
        saturation = side_cfg.get("saturation", 64)
        hue = side_cfg.get("hue", 0)
        if not isinstance(brightness, (int, float)) or not isinstance(
            contrast, (int, float)
        ):
            raise ValueError(
                f"Camera config requires numeric brightness/contrast for '{side}'"
            )
        if not isinstance(saturation, (int, float)):
            raise ValueError(f"Camera config requires numeric saturation for '{side}'")
        if not isinstance(hue, (int, float)):
            raise ValueError(f"Camera config requires numeric hue for '{side}'")

        params[side] = {
            "brightness": int(brightness),
            "contrast": int(contrast),
            "saturation": int(saturation),
            "hue": int(hue),
        }

    return params


def load_intrinsics_from_config(robot: str, side: str) -> np.ndarray:
    data, config_path = load_robot_camera_config(
        robot, config_override=os.environ.get("MCC_CAMERA_CONFIG")
    )
    calibration = data.get("calibration", {})
    matrix_key = "K1" if side == "left" else "K2"
    if isinstance(calibration, dict) and matrix_key in calibration:
        intrinsics = np.asarray(calibration[matrix_key], dtype=np.float32)
        if intrinsics.shape != (3, 3):
            raise ValueError(
                f"Calibration matrix {matrix_key} must be 3x3 in {config_path}"
            )
        return intrinsics

    return np.eye(3, dtype=np.float32)


class Camera:
    """Camera class for capturing stereo images."""

    mjpg_device_ids: Optional[List[int]] = None
    claimed_device_ids: set[int] = set()
    claimed_device_paths: set[str] = set()
    device_control_cache: dict[int, Optional[Set[str]]] = {}

    def __init__(
        self,
        side,
        width=640,
        height=480,
        fps=30,
        robot: Optional[str] = None,
    ):
        """Initializes the camera setup for either the left or right side and configures capture settings.

        Args:
            side (str): Specifies the side of the camera, either 'left' or 'right'.
            width (int, optional): The width of the video capture frame. Defaults to 640.
            height (int, optional): The height of the video capture frame. Defaults to 480.
            fps (int | None, optional): Target frames per second. Defaults to 20.
            robot (str | None, optional): Robot name used to select
                `assets/<robot>_camera.yml`. Defaults to `MCC_ROBOT` or `toddlerbot`.

        Raises:
            Exception: If the camera cannot be opened.
        """
        self.side = side
        self.robot = normalize_robot_name(
            robot or os.environ.get("MCC_ROBOT", "toddlerbot")
        )

        self.camera_id = None
        self.camera_path = None
        self._select_camera_source(width, height)

        self.width = width
        self.height = height
        self.fps = fps

        camera_params = load_camera_params(self.robot)
        side_params = camera_params.get(side, {})
        self.brightness = side_params.get("brightness")
        self.contrast = side_params.get("contrast")
        self.saturation = side_params.get("saturation")
        self.hue = side_params.get("hue")

        # Fall back to device defaults if config is missing.
        if self.brightness is None:
            self.brightness = 0
        if self.contrast is None:
            self.contrast = 0
        if self.saturation is None:
            self.saturation = 64
        if self.hue is None:
            self.hue = 0

        self.frame_lock = threading.Lock()
        self.latest_frame: Optional[np.ndarray] = None
        self.latest_frame_time: Optional[float] = None
        self.frame_event = threading.Event()
        self.frame_buffer: list[tuple[float, np.ndarray]] = []
        self.frame_buffer_size = 10
        self.stop_capture = threading.Event()
        self.capture_thread: Optional[threading.Thread] = None
        self.refresh_lock = threading.Lock()
        self.cap_lock = threading.Lock()
        self.refresh_backoff_until = 0.0
        self.failed_read_count = 0

        self.apply_camera_controls(include_optional=True)
        self._open_capture()
        self.intrinsics = load_intrinsics_from_config(self.robot, side)

        # Transformation from right camera to left camera
        self.eye_transform = (
            np.array([[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]])
            if side == "left"
            else np.array(
                [[0, 0, 1, 0], [-1, 0, 0, -0.033], [0, -1, 0, 0], [0, 0, 0, 1]]
            )
        )
        if not self.cap.isOpened():
            raise Exception("Error: Could not open camera.")
        self.start_capture_thread()

    def _select_camera_source(self, width: int, height: int) -> None:
        by_id_dir = Path("/dev/v4l/by-id")
        if by_id_dir.exists():
            by_id_paths = sorted(str(p) for p in by_id_dir.glob("*-video-index0"))
            available_paths = [
                p for p in by_id_paths if p not in self.claimed_device_paths
            ]
            unique_claim = True
            if not available_paths:
                available_paths = by_id_paths
                unique_claim = False
            if available_paths:
                camera_path = (
                    available_paths[0] if self.side == "left" else available_paths[-1]
                )
                self.camera_path = camera_path
                self.camera_id = None
                if unique_claim:
                    self.claimed_device_paths.add(camera_path)
                return

        mjpg_devices = self.get_mjpg_devices(width, height)
        available_devices = [
            dev_id for dev_id in mjpg_devices if dev_id not in self.claimed_device_ids
        ]
        unique_claim = True
        if not available_devices:
            available_devices = mjpg_devices
            unique_claim = False

        if self.side == "left":
            camera_id = available_devices[-1]
        else:
            camera_id = available_devices[0]
        if unique_claim:
            self.claimed_device_ids.add(camera_id)
        self.camera_id = camera_id
        self.camera_path = None

    def _device_arg(self) -> Optional[str]:
        if self.camera_path is not None:
            return self.camera_path
        if self.camera_id is not None:
            return f"/dev/video{self.camera_id}"
        return None

    def _open_capture(self) -> None:
        with self.cap_lock:
            if self.camera_path is not None and not os.path.exists(self.camera_path):
                return
            self.cap = cv2.VideoCapture(self.camera_path or self.camera_id)
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            if self.fps is not None:
                self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    @classmethod
    def get_mjpg_devices(cls, width: int, height: int) -> List[int]:
        """Return a cached list of MJPG-capable /dev/video ids."""
        if cls.mjpg_device_ids is not None:
            return cls.mjpg_device_ids

        video_devices = sorted(
            int(dev[5:]) for dev in os.listdir("/dev") if dev.startswith("video")
        )

        def probe_with_v4l2(dev_id: int) -> bool:
            try:
                result = subprocess.run(
                    [
                        "v4l2-ctl",
                        f"--device=/dev/video{dev_id}",
                        "--list-formats",
                    ],
                    capture_output=True,
                    text=True,
                    check=True,
                )
            except (subprocess.CalledProcessError, FileNotFoundError):
                return False
            return "MJPG" in result.stdout

        mjpg_devices = []
        for dev_id in video_devices:
            if probe_with_v4l2(dev_id):
                mjpg_devices.append(dev_id)

        if not mjpg_devices:
            raise RuntimeError(
                "Could not find any MJPG-capable camera devices. "
                f"Detected video nodes: {video_devices}"
            )

        cls.mjpg_device_ids = sorted(mjpg_devices)
        return cls.mjpg_device_ids

    @classmethod
    def list_device_controls(cls, dev_id: int) -> Optional[Set[str]]:
        """Return the set of controls reported by v4l2 for a device."""
        if dev_id in cls.device_control_cache:
            return cls.device_control_cache[dev_id]

        try:
            result = subprocess.run(
                [
                    "v4l2-ctl",
                    f"--device=/dev/video{dev_id}",
                    "-L",
                ],
                capture_output=True,
                text=True,
                check=True,
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            controls: Optional[Set[str]] = None
        else:
            controls = set()
            for raw_line in result.stdout.splitlines():
                line = raw_line.strip()
                if not line or "0x" not in line:
                    continue
                controls.add(line.split(None, 1)[0])

        cls.device_control_cache[dev_id] = controls
        return controls

    def resolve_control_name(self, *candidates: str) -> Optional[str]:
        """Pick the first candidate supported by the device or fall back if unknown."""
        if not candidates:
            return None
        if self.camera_id is None:
            return candidates[0]
        available = type(self).list_device_controls(self.camera_id)
        if available is None:
            return candidates[0]
        for name in candidates:
            if name in available:
                return name
        return None

    def start_capture_thread(self) -> None:
        """Start a background thread that continuously grabs frames from the device."""
        if self.capture_thread is not None:
            return

        def _capture_loop() -> None:
            while not self.stop_capture.is_set():
                with self.cap_lock:
                    ret, frame = self.cap.read()
                if not ret:
                    self.failed_read_count += 1
                    if self.failed_read_count >= 30:
                        self.refresh()
                        self.failed_read_count = 0
                    time.sleep(0.01)
                    continue
                self.failed_read_count = 0
                with self.frame_lock:
                    self.latest_frame = frame
                    self.latest_frame_time = time.time()
                    self.frame_buffer.append((self.latest_frame_time, frame.copy()))
                    if len(self.frame_buffer) > self.frame_buffer_size:
                        self.frame_buffer = self.frame_buffer[-self.frame_buffer_size :]
                self.frame_event.set()

        self.capture_thread = threading.Thread(
            target=_capture_loop, name=f"{self.side}_camera_capture", daemon=True
        )
        self.capture_thread.start()

    def apply_camera_controls(self, include_optional: bool = False) -> None:
        """Apply controls via v4l2."""
        device_arg = self._device_arg()

        if device_arg is None:
            return

        controls = {
            "brightness": self.brightness,
            "contrast": self.contrast,
            "saturation": self.saturation,
            "hue": self.hue,
        }

        optional_segments: list[str] = []
        if include_optional:
            exposure_ctrl = self.resolve_control_name("exposure_auto", "auto_exposure")
            if exposure_ctrl is not None:
                optional_segments.append(f"{exposure_ctrl}=3")

            wb_ctrl = self.resolve_control_name(
                "white_balance_temperature_auto", "white_balance_automatic"
            )
            if wb_ctrl is not None:
                optional_segments.append(f"{wb_ctrl}=1")

        required_segments: list[str] = []
        for key, value in controls.items():
            if value is None:
                continue
            control_name = self.resolve_control_name(key)
            if control_name is not None:
                required_segments.append(f"{control_name}={value}")

        if optional_segments:
            control_segments = optional_segments + required_segments
        else:
            control_segments = list(required_segments)

        if not control_segments:
            return

        command = [
            "v4l2-ctl",
            f"--device={device_arg}",
            "--set-ctrl=" + ",".join(control_segments),
        ]
        try:
            subprocess.run(
                command,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                "v4l2-ctl command not found. Please install v4l-utils to adjust camera controls."
            ) from exc
        except subprocess.CalledProcessError as exc:
            error_msg = exc.stderr.strip() if exc.stderr else "Unknown error"
            if include_optional and optional_segments:
                # Optional controls may be unsupported; retry without them quietly.
                # Re-run without optional segments in case they are unsupported.
                self.apply_camera_controls(include_optional=False)
                return
            raise RuntimeError(
                f"Failed to set controls on {device_arg}: {error_msg}"
            ) from exc

    def set_brightness(self, brightness: int) -> None:
        """Update the brightness setting and reapply camera controls."""
        if brightness == self.brightness:
            return
        self.brightness = brightness
        self.apply_camera_controls()

    def set_contrast(self, contrast: int | None) -> None:
        """Update contrast and apply camera controls."""
        if contrast == self.contrast:
            return
        self.contrast = contrast
        self.apply_camera_controls()

    def set_saturation(self, saturation: int | None) -> None:
        """Update saturation and apply camera controls."""
        if saturation == self.saturation:
            return
        self.saturation = saturation
        self.apply_camera_controls()

    def set_hue(self, hue: int | None) -> None:
        """Update hue and apply camera controls."""
        if hue == self.hue:
            return
        self.hue = hue
        self.apply_camera_controls()

    def get_frame(self):
        """Return the most recent frame captured by the background thread."""
        if self.capture_thread is None:
            raise RuntimeError("Camera capture thread is not running.")

        if not self.frame_event.wait(timeout=1.0):
            self.refresh()
            self.frame_event.wait(timeout=1.0)

        with self.frame_lock:
            frame = None if self.latest_frame is None else self.latest_frame.copy()
            frame_time = self.latest_frame_time

        if frame is None:
            self.refresh()
            with self.frame_lock:
                if self.latest_frame is None:
                    raise RuntimeError(
                        "Camera capture has not produced any frames yet."
                    )
                return self.latest_frame.copy()

        if frame_time is not None and (time.time() - frame_time) > 1.0:
            self.refresh()
            with self.frame_lock:
                if self.latest_frame is None:
                    return frame
                return self.latest_frame.copy()
        return frame

    def get_synced_frames(
        self, other: "Camera", max_dt: float = 0.02
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return a time-synchronized frame pair within max_dt seconds."""
        self.frame_event.wait(timeout=1.0)
        other.frame_event.wait(timeout=1.0)

        with self.frame_lock:
            self_buffer = list(self.frame_buffer)
        with other.frame_lock:
            other_buffer = list(other.frame_buffer)

        if not self_buffer or not other_buffer:
            return self.get_frame(), other.get_frame()

        best = None
        best_dt = float("inf")
        for t_a, f_a in self_buffer:
            for t_b, f_b in other_buffer:
                dt = abs(t_a - t_b)
                if dt < best_dt:
                    best_dt = dt
                    best = (f_a, f_b)
        if best is None or best_dt > max_dt:
            print(f"[Camera] Synced frame dt={best_dt:.4f}s (fallback).")
            return self.get_frame(), other.get_frame()
        print(f"[Camera] Synced frame dt={best_dt:.4f}s.")
        return best

    def refresh(self) -> None:
        """Reopen the camera if frames appear stale."""
        with self.refresh_lock:
            now = time.time()
            if now < self.refresh_backoff_until:
                return
            self.refresh_backoff_until = now + 0.5

            try:
                with self.cap_lock:
                    self.cap.release()
            except Exception:
                pass
            try:
                type(self).mjpg_device_ids = None
                if self.camera_path in self.claimed_device_paths:
                    self.claimed_device_paths.discard(self.camera_path)
                if self.camera_id in self.claimed_device_ids:
                    self.claimed_device_ids.discard(self.camera_id)
                self._select_camera_source(self.width, self.height)
            except Exception:
                # Keep existing camera_id if probing fails.
                return

            if self.camera_path is not None and not os.path.exists(self.camera_path):
                self.refresh_backoff_until = now + 2.0
                return
            self._open_capture()
            try:
                self.apply_camera_controls()
            except Exception:
                pass

    def get_jpeg(self):
        """Converts the current video frame to JPEG format and returns it along with the RGB frame.

        Returns:
            tuple: A tuple containing:
                - jpeg (numpy.ndarray): The encoded JPEG image.
                - frame_rgb (numpy.ndarray): The RGB representation of the current video frame.
        """
        frame = self.get_frame()
        frame_rgb = np.array(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), dtype=np.uint8)
        # Encode the frame as a JPEG with quality of 90
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 50]
        _, jpeg = cv2.imencode(".jpg", frame_rgb, encode_param)
        return jpeg, frame_rgb

    def close(self):
        """Releases the video capture object and closes all OpenCV windows.

        This method should be called to properly release the resources associated with the video capture and to close any OpenCV windows that were opened during the process.
        """
        self.stop_capture.set()
        if self.capture_thread is not None:
            self.capture_thread.join(timeout=2.0)
            self.capture_thread = None
        self.frame_event.set()
        with self.cap_lock:
            self.cap.release()
        cv2.destroyAllWindows()

        if (
            hasattr(self, "camera_id")
            and self.camera_id is not None
            and self.camera_id in self.claimed_device_ids
        ):
            self.claimed_device_ids.discard(self.camera_id)
        if self.camera_path in self.claimed_device_paths:
            self.claimed_device_paths.discard(self.camera_path)
