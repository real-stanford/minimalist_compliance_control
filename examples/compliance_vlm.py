from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import gin
import joblib
import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation as R

from minimalist_compliance_control.compliance_ref import COMMAND_LAYOUT
from minimalist_compliance_control.controller import (
    ComplianceController,
    ComplianceRefConfig,
    ControllerConfig,
)
from minimalist_compliance_control.wrench_estimation import WrenchEstimateConfig
from real_world.camera import Camera
from vlm.affordance.affordance_predictor import AffordancePredictor
from vlm.affordance.plan_ee_pose import plan_end_effector_poses
from vlm.utils.comm_utils import ZMQNode
from vlm.utils.math_utils import ensure_matrix, get_damping_matrix

LEAP_DRAW_POS = np.array(
    [2.23, 0, 0, 0.4, 2.23, 0, 0, 0.4, 2.23, 0, 0, 0.4, 0.0, -1.57, 0.0, 0.0],
    dtype=np.float32,
)


def ComplianceVLMInput(**kwargs):
    return SimpleNamespace(**kwargs)


def ComplianceVLMOutput(**kwargs):
    return SimpleNamespace(**kwargs)


@gin.configurable
def get_motor_config_paths(
    default_config_path: Optional[str] = None,
    robot_config_path: Optional[str] = None,
    motors_config_path: Optional[str] = None,
) -> tuple[Optional[str], Optional[str], Optional[str]]:
    return default_config_path, robot_config_path, motors_config_path


def _to_hwc_u8(image: np.ndarray, *, size_hw: tuple[int, int]) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim != 3:
        arr = np.zeros((size_hw[0], size_hw[1], 3), dtype=np.uint8)
    if arr.shape[0] in (1, 3) and arr.shape[2] not in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    if arr.dtype != np.uint8:
        max_v = float(arr.max()) if arr.size else 0.0
        if max_v <= 1.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0.0, 255.0).astype(np.uint8)
    return arr


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compliance VLM per-tick policy")
    parser.add_argument("--robot-name", type=str, default="toddlerbot_2xm")
    parser.add_argument(
        "--site-names", type=str, default="", help="Comma-separated site names"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="waiting",
        choices=["waiting", "wiping", "drawing"],
    )
    parser.add_argument("--object", type=str, default="black ink. vase")
    parser.add_argument("--mode-control-port", type=int, default=5591)
    parser.add_argument("--disable-zmq", action="store_true")
    parser.add_argument("--use-camera-stream", action="store_true")
    parser.add_argument("--disable-video", action="store_true")
    parser.add_argument("--image-height", type=int, default=480)
    parser.add_argument("--image-width", type=int, default=640)
    return parser


class ComplianceVLMPolicy:
    """Per-tick compliance VLM policy with (obs, sim) -> (control_inputs, action)."""

    def __init__(
        self,
        args: argparse.Namespace,
        *,
        robot: str,
        sim: str,
        vis: bool,
        plot: bool,
    ) -> None:
        del plot
        self.args = args
        self.robot = str(robot)
        self.sim_backend = str(sim)
        self.vis = bool(vis)
        self.repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        os.chdir(self.repo_root)

        config_name = "leap.gin" if self.robot == "leap" else "toddlerbot.gin"
        config_path = os.path.join(self.repo_root, "config", config_name)
        gin.clear_config()
        gin.parse_config_file(config_path)
        gin.bind_parameter(
            "WrenchSimConfig.view", bool(self.vis and self.sim_backend == "mujoco")
        )
        gin.bind_parameter("WrenchSimConfig.render", False)

        self.controller = ComplianceController(
            config=ControllerConfig(),
            estimate_config=WrenchEstimateConfig(),
            ref_config=ComplianceRefConfig(),
        )
        if self.controller.compliance_ref is None:
            raise ValueError("Controller compliance_ref must be configured.")
        self.model = self.controller.wrench_sim.model
        self.data = self.controller.wrench_sim.data
        self.control_dt = float(self.controller.ref_config.dt)
        self.site_names = tuple(self.controller.config.site_names)
        self.num_sites = len(self.site_names)
        default_state = self.controller.compliance_ref.get_default_state()
        self.target_motor_pos = np.asarray(default_state.motor_pos, dtype=np.float32)

        (
            default_config_path,
            robot_config_path,
            motors_config_path,
        ) = get_motor_config_paths()
        if not default_config_path or not robot_config_path:
            raise ValueError(
                "get_motor_config_paths.default_config_path and robot_config_path must be configured."
            )
        if self.robot == "leap" and not motors_config_path:
            raise ValueError(
                "get_motor_config_paths.motors_config_path must be configured for leap."
            )
        self.default_config_path = str(default_config_path)
        self.robot_config_path = str(robot_config_path)
        self.motors_config_path = (
            str(motors_config_path) if motors_config_path is not None else None
        )

        trnid = np.asarray(self.model.actuator_trnid[:, 0], dtype=np.int32)
        self.qpos_adr = np.asarray(self.model.jnt_qposadr[trnid], dtype=np.int32)
        self.qvel_adr = np.asarray(self.model.jnt_dofadr[trnid], dtype=np.int32)

        self._vlm_input_cls = ComplianceVLMInput
        site_names_override = None
        if len(args.site_names.strip()) > 0:
            site_names_override = [
                s.strip() for s in args.site_names.split(",") if s.strip()
            ]
        self._init_vlm_core(
            site_names=site_names_override or list(self.site_names),
            robot_name=str(args.robot_name),
            control_dt=self.control_dt,
            mode_control_port=int(args.mode_control_port),
            enable_mode_control=not bool(args.disable_zmq),
            use_camera_stream=bool(args.use_camera_stream),
            record_video=not bool(args.disable_video),
        )
        if args.mode == "wiping":
            self.set_mode(True, object_label=None, site_names=None)
        elif args.mode == "drawing":
            self.set_mode(False, object_label=str(args.object), site_names=None)
        self._closed = False

    @classmethod
    def from_argv(
        cls,
        argv: Sequence[str],
        *,
        robot: str,
        sim: str,
        vis: bool,
        plot: bool,
    ) -> "ComplianceVLMPolicy":
        args = build_parser().parse_args(list(argv))
        return cls(args, robot=robot, sim=sim, vis=vis, plot=plot)

    def _init_vlm_core(
        self,
        site_names: Optional[List[str]] = None,
        robot_name: str = "toddlerbot_2xm",
        control_dt: float = 0.02,
        mode_control_port: int = 5591,
        enable_mode_control: bool = True,
        predictor_model: str = "gemini-2.5-pro",
        predictor_provider: str = "gemini",
        use_camera_stream: bool = False,
        camera_left_device: Optional[int] = None,
        camera_right_device: Optional[int] = None,
        fixed_draw_trajectory_path: str = "",
        fixed_wipe_trajectory_path: str = "",
        record_video: bool = True,
        default_head_pos_world: npt.ArrayLike = (0.0, 0.0, 0.0),
        default_head_quat_world_wxyz: npt.ArrayLike = (1.0, 0.0, 0.0, 0.0),
    ) -> None:
        self.robot_name = str(robot_name)
        self.control_dt = float(control_dt)
        self.mass = 1.0
        self.inertia_diag = np.array([1.0, 1.0, 1.0], dtype=np.float32)

        lower_name = self.robot_name.lower()
        if "arx" in lower_name:
            default_site = ["ee_site"]
            self.force_only = False
            self.estimate_full_wrench = True
            self.normal_pos_stiffness = 80.0
            self.tangent_pos_stiffness = 400.0
            self.fixed_contact_force = 5.0
        elif "leap" in lower_name:
            default_site = ["mf_tip"]
            self.force_only = True
            self.estimate_full_wrench = True
            self.ref_motor_pos = LEAP_DRAW_POS.copy()
            self.normal_pos_stiffness = 20.0
            self.tangent_pos_stiffness = 200.0
            self.fixed_contact_force = 0.2
        else:
            default_site = ["left_hand_center"]
            self.force_only = False
            self.estimate_full_wrench = False
            self.normal_pos_stiffness = 80.0
            self.tangent_pos_stiffness = 400.0
            self.fixed_contact_force = 5.0

        self.wrench_site_names = (
            list(site_names) if site_names is not None else list(default_site)
        )
        if len(self.wrench_site_names) == 0:
            raise ValueError("site_names cannot be empty")

        self.num_sites = len(self.wrench_site_names)
        self.pose_command = np.zeros((self.num_sites, 6), dtype=np.float32)
        self.wrench_command = np.zeros((self.num_sites, 6), dtype=np.float32)
        self.pos_stiffness = np.zeros((self.num_sites, 9), dtype=np.float32)
        self.rot_stiffness = np.zeros((self.num_sites, 9), dtype=np.float32)
        self.pos_damping = np.zeros((self.num_sites, 9), dtype=np.float32)
        self.rot_damping = np.zeros((self.num_sites, 9), dtype=np.float32)
        self._pose_initialized = False

        self.site_normal_axis: Dict[str, str] = {
            name: "+x" for name in self.wrench_site_names
        }
        normal_overrides: Dict[str, str] = {
            "left_hand_center": "+z",
            "right_hand_center": "+z",
            "if_tip": "-x",
            "mf_tip": "-x",
            "rf_tip": "-x",
            "th_tip": "-x",
        }
        for site_name in self.wrench_site_names:
            if site_name in normal_overrides:
                self.site_normal_axis[site_name] = normal_overrides[site_name]

        self.set_stiffness(
            pos_stiffness=[400.0, 400.0, 400.0],
            rot_stiffness=[40.0, 40.0, 40.0],
        )

        self.status = "waiting"
        self.target_object_label = ""
        self.tool = ""
        self.target_site_names = list(default_site)
        self.target_site_indices: List[int] = []
        self._refresh_target_indices()

        self.traj_start_time: Optional[float] = None
        self.traj_v_max_contact = 0.02
        self.traj_v_max_free = 0.1
        self.trajectory_plans: Dict[str, Tuple[np.ndarray, ...]] = {}

        self.debug_output_dir = tempfile.TemporaryDirectory(prefix="compliance_vlm_")
        self.prediction_counter = 0
        self.prediction_executor = ThreadPoolExecutor(max_workers=1)
        self.prediction_future: Optional[Future] = None
        self.prediction_requested = False

        default_draw = Path(
            "results/toddlerbot_2xm_compliance_vlm_real_world_[time_str]/affordance_draw"
        )
        default_wipe = Path(
            "results/toddlerbot_2xm_compliance_vlm_real_world_[time_str]/affordance_wipe"
        )
        self.fixed_draw_trajectory_path = (
            Path(fixed_draw_trajectory_path)
            if fixed_draw_trajectory_path
            else (default_draw / "trajectory.lz4")
        )
        self.fixed_wipe_trajectory_path = (
            Path(fixed_wipe_trajectory_path)
            if fixed_wipe_trajectory_path
            else (default_wipe / "trajectory.lz4")
        )
        self.use_fixed_trajectory = False
        self.fixed_trajectory_active = False

        self.video_logging_active = False
        self.record_video = bool(record_video)
        self.record_left_only = False
        self.video_writer: Optional[cv2.VideoWriter] = None
        self.video_temp_dir: Optional[tempfile.TemporaryDirectory] = None
        self.video_path: Optional[Path] = None
        self.video_fps: Optional[float] = None
        self.last_left_frame: Optional[np.ndarray] = None
        self.last_right_frame: Optional[np.ndarray] = None
        self.video_capture_thread: Optional[threading.Thread] = None
        self.video_capture_stop: Optional[threading.Event] = None
        self.video_frame_timestamps: List[float] = []

        self.wipe_pause_duration = 2.0
        self.wipe_pause_end_time: Optional[float] = None
        self.wiping_complete = False
        self.args_payload: Dict[str, object] = {}

        self.default_head_pos_world = np.asarray(
            default_head_pos_world, dtype=np.float32
        ).reshape(3)
        self.default_head_quat_world_wxyz = np.asarray(
            default_head_quat_world_wxyz, dtype=np.float32
        ).reshape(4)

        self.left_camera: Optional[Camera] = None
        self.right_camera: Optional[Camera] = None
        if use_camera_stream:
            try:
                if camera_left_device is not None or camera_right_device is not None:
                    print(
                        "[ComplianceVLM] camera_left_device/camera_right_device are ignored with real_world.camera.Camera."
                    )
                self.left_camera = Camera("left", robot=self.robot_name)
                self.right_camera = Camera("right", robot=self.robot_name)
            except Exception as exc:
                self.left_camera = None
                self.right_camera = None
                print(f"[ComplianceVLM] Warning: camera stream disabled: {exc}")

        self.mode_control_port = int(mode_control_port)
        self.mode_control_receiver: Optional[ZMQNode] = None
        if enable_mode_control:
            try:
                self.mode_control_receiver = ZMQNode(
                    type="receiver", port=self.mode_control_port
                )
                print(
                    f"[ComplianceVLM] Mode control listening on port {self.mode_control_port} (w=wipe, d=draw)."
                )
            except Exception as exc:
                self.mode_control_receiver = None
                print(f"[ComplianceVLM] Warning: mode control receiver disabled: {exc}")

        self.predictor: Optional[AffordancePredictor] = None
        try:
            camera_robot = "leap" if "leap" in lower_name else "toddlerbot"
            camera_config_path = Path("assets") / f"{camera_robot}_camera.yml"
            self.predictor = AffordancePredictor(
                provider=predictor_provider,
                model=predictor_model,
                depth_config={"camera_config": str(camera_config_path)},
            )
        except Exception as exc:
            self.predictor = None
            print(
                "[ComplianceVLM] Warning: affordance predictor unavailable "
                f"({exc}). Prediction-dependent modes will not run."
            )

        self.refresh_fixed_trajectory_flag()

    def _refresh_target_indices(self) -> None:
        self.target_site_indices = []
        for site_name in self.target_site_names:
            if site_name in self.wrench_site_names:
                self.target_site_indices.append(self.wrench_site_names.index(site_name))

        if len(self.target_site_indices) == 0:
            self.target_site_names = [self.wrench_site_names[0]]
            self.target_site_indices = [0]

    def _ensure_pose_from_obs(self, x_obs: Optional[npt.NDArray[np.float32]]) -> None:
        if self._pose_initialized or x_obs is None:
            return
        obs_arr = np.asarray(x_obs, dtype=np.float32)
        if obs_arr.ndim != 2 or obs_arr.shape != (self.num_sites, 6):
            return
        self.pose_command[:] = obs_arr
        self._pose_initialized = True

    def set_stiffness(
        self,
        pos_stiffness,
        rot_stiffness,
        pos_damp_ratio: float = 1.0,
        rot_damp_ratio: float = 1.0,
        pos_damping: Optional[list | np.ndarray] = None,
        rot_damping: Optional[list | np.ndarray] = None,
    ) -> None:
        kp_pos = ensure_matrix(pos_stiffness)
        kp_rot = ensure_matrix(rot_stiffness)

        if pos_damping is not None:
            kd_pos = ensure_matrix(pos_damping)
        else:
            kd_pos = get_damping_matrix(kp_pos, self.mass) * pos_damp_ratio

        if rot_damping is not None:
            kd_rot = ensure_matrix(rot_damping)
        else:
            kd_rot = get_damping_matrix(kp_rot, self.inertia_diag) * rot_damp_ratio

        self.pos_stiffness = np.tile(kp_pos.reshape(-1, 9), (self.num_sites, 1))
        self.rot_stiffness = np.tile(kp_rot.reshape(-1, 9), (self.num_sites, 1))
        self.pos_damping = np.tile(kd_pos.reshape(-1, 9), (self.num_sites, 1))
        self.rot_damping = np.tile(kd_rot.reshape(-1, 9), (self.num_sites, 1))

    def reset(self) -> None:
        self.traj_start_time = None
        self.use_fixed_trajectory = False
        self.wipe_pause_end_time = None
        self.prediction_requested = False
        self.status = "waiting"
        self.fixed_trajectory_active = False
        self.wiping_complete = False

        if self.prediction_future is not None:
            self.prediction_future.cancel()
            self.prediction_future = None

    def set_mode(
        self,
        is_wiping: bool,
        object_label: Optional[str] = None,
        site_names: Optional[List[str]] = None,
    ) -> None:
        target_status = "wiping" if is_wiping else "drawing"
        if self.status == target_status and self.status != "waiting":
            return

        self.status = target_status
        self.tool = "eraser" if is_wiping else "pen"
        self.prediction_requested = False
        if is_wiping:
            self.wiping_complete = False

        if self.prediction_future is not None:
            self.prediction_future.cancel()
            self.prediction_future = None

        self.trajectory_plans = {}
        self.traj_start_time = None
        self.wipe_pause_end_time = None
        self.fixed_trajectory_active = False
        self.refresh_fixed_trajectory_flag()

        if is_wiping:
            self.target_object_label = "black ink. vase"
        elif object_label:
            self.target_object_label = object_label

        lower_name = self.robot_name.lower()
        if "leap" in lower_name:
            self.fixed_contact_force = 0.2 if is_wiping else 0.7
        else:
            self.fixed_contact_force = 5.0

        task_description = (
            f"wipe up the {self.target_object_label} on the vase with an eraser."
            if is_wiping
            else f"draw the {self.target_object_label} on the vase using the pen."
        )

        if self.predictor is not None:
            self.predictor.default_task = task_description

        if "arx" in lower_name:
            default_site = ["ee_site"]
        elif "leap" in lower_name:
            default_site = ["mf_tip"] if is_wiping else ["rf_tip", "if_tip"]
        else:
            default_site = ["left_hand_center"] if is_wiping else ["right_hand_center"]

        self.target_site_names = site_names if site_names is not None else default_site
        self._refresh_target_indices()

        self.args_payload["object"] = self.target_object_label
        self.args_payload["site"] = (
            self.target_site_names if self.target_site_names else ""
        )
        self.args_payload["task_description"] = task_description

        print(f"[ComplianceVLM] Switched to {self.status} mode.")

    def get_fixed_trajectory_path(self) -> Optional[Path]:
        if self.status == "wiping":
            return self.fixed_wipe_trajectory_path
        if self.status == "drawing":
            return self.fixed_draw_trajectory_path
        return None

    def refresh_fixed_trajectory_flag(self) -> None:
        path = self.get_fixed_trajectory_path()
        self.use_fixed_trajectory = bool(path and path.exists())

    def _apply_mode_message(self, msg: object) -> None:
        if msg is None:
            return

        mode_val: Optional[str] = None
        label_val: Optional[str] = None
        sites_val: Optional[List[str]] = None

        if isinstance(msg, dict):
            mode_raw = msg.get("mode")
            if mode_raw is not None:
                mode_val = str(mode_raw)
            object_raw = msg.get("object")
            if object_raw is not None:
                label_val = str(object_raw)
            sites = msg.get("site") or msg.get("sites")
            if isinstance(sites, list):
                sites_val = [str(s) for s in sites if str(s).strip()]
        else:
            mode_val = str(msg)

        if mode_val is None:
            return

        mode_str = str(mode_val).strip()
        tokens = mode_str.split()
        cmd = tokens[0].lower() if tokens else ""

        if len(tokens) > 1:
            rest = tokens[1:]
            if "with" in rest:
                idx = rest.index("with")
                label_tokens = rest[:idx]
                site_tokens = rest[idx + 1 :]
            else:
                label_tokens = rest
                site_tokens = []
            if label_val is None and label_tokens:
                label_val = " ".join(label_tokens).strip()
            if sites_val is None and site_tokens:
                sites_val = [token.strip(",") for token in site_tokens if token]

        if cmd in ("w", "wipe", "wiping"):
            self.set_mode(True, None, None)
        elif cmd in ("d", "draw", "drawing"):
            self.set_mode(
                False,
                label_val if label_val else None,
                sites_val if sites_val else None,
            )

    def check_mode_command(self, external_msg: Optional[object] = None) -> None:
        if self.status != "waiting":
            return

        self._apply_mode_message(external_msg)

        if self.mode_control_receiver is None:
            return

        msg = self.mode_control_receiver.get_msg()
        self._apply_mode_message(msg)

    def start_video_logging(self) -> None:
        if self.video_logging_active or self.left_camera is None:
            return
        self.discard_video_recording()
        self.video_logging_active = True
        self.start_video_capture_thread()

    def ensure_video_writer(self, frame: np.ndarray) -> bool:
        if self.video_writer is not None:
            return True

        if self.video_temp_dir is None:
            self.video_temp_dir = tempfile.TemporaryDirectory(
                prefix="compliance_vlm_video_"
            )

        filename = "left_camera.mp4" if self.record_left_only else "stereo_camera.mp4"
        self.video_path = Path(self.video_temp_dir.name) / filename
        height, width = frame.shape[:2]
        fps = max(1.0, float(1.0 / max(self.control_dt, 1e-3)))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(self.video_path), fourcc, fps, (width, height))
        if not writer.isOpened():
            writer.release()
            print(
                "[ComplianceVLM] Warning: failed to open video writer; disabling video logging."
            )
            self.video_logging_active = False
            if self.video_capture_stop is not None:
                self.video_capture_stop.set()
            return False

        self.video_writer = writer
        self.video_fps = fps
        return True

    def start_video_capture_thread(self) -> None:
        if (
            self.video_capture_thread is not None
            and self.video_capture_thread.is_alive()
        ):
            return
        self.video_capture_stop = threading.Event()
        self.video_capture_thread = threading.Thread(
            target=self.video_capture_worker,
            name="ComplianceVLMCapture",
            daemon=True,
        )
        self.video_capture_thread.start()

    def stop_video_capture_thread(self, timeout_s: float = 1.0) -> None:
        if self.video_capture_stop is not None:
            self.video_capture_stop.set()
        if self.video_capture_thread is not None:
            self.video_capture_thread.join(timeout=timeout_s)
        self.video_capture_thread = None
        self.video_capture_stop = None

    def video_capture_worker(self) -> None:
        if self.left_camera is None:
            return
        capture_dt = max(float(self.control_dt), 1e-3)
        next_time = time.monotonic()
        start_time = next_time
        while True:
            if self.video_capture_stop is not None and self.video_capture_stop.is_set():
                break
            now = time.monotonic()
            if now < next_time:
                time.sleep(next_time - now)
            next_time += capture_dt
            try:
                left_frame = self.left_camera.get_frame()
            except Exception:
                self.last_left_frame = None
                continue
            self.last_left_frame = left_frame
            right_frame = None
            if self.right_camera is not None:
                try:
                    right_frame = self.right_camera.get_frame()
                except Exception:
                    right_frame = None
                self.last_right_frame = right_frame

            output_frame = left_frame
            if not self.record_left_only and right_frame is not None:
                if right_frame.shape[0] != left_frame.shape[0]:
                    right_frame = cv2.resize(
                        right_frame, (left_frame.shape[1], left_frame.shape[0])
                    )
                output_frame = np.concatenate([left_frame, right_frame], axis=1)

            if not self.video_logging_active:
                continue
            if not self.ensure_video_writer(output_frame):
                continue
            if self.video_writer is None:
                continue
            self.video_writer.write(output_frame)
            timestamp = time.monotonic() - start_time
            self.video_frame_timestamps.append(float(timestamp))

    def discard_video_recording(self) -> None:
        self.stop_video_capture_thread()
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
        if self.video_temp_dir is not None:
            self.video_temp_dir.cleanup()
            self.video_temp_dir = None
        self.video_path = None
        self.video_fps = None
        self.video_frame_timestamps = []
        self.video_logging_active = False

    def log_camera_frame(
        self, timestamp: float, left_only: Optional[bool] = None
    ) -> None:
        del timestamp
        if left_only is not None:
            self.record_left_only = left_only
        self.start_video_logging()

    def export_camera_video(self, output_dir: Path) -> None:
        self.stop_video_capture_thread()
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None

        timestamps = np.asarray(self.video_frame_timestamps, dtype=np.float64)
        if (
            timestamps.size == 0
            or self.video_path is None
            or not self.video_path.exists()
        ):
            return

        output_dir.mkdir(parents=True, exist_ok=True)
        dest_path = output_dir / self.video_path.name
        if dest_path.exists():
            dest_path.unlink()

        if timestamps.size >= 2:
            duration = max(timestamps[-1] - timestamps[0], 1e-6)
            actual_fps = float((timestamps.size - 1) / duration)
        else:
            actual_fps = float(self.video_fps or 1.0)

        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            shutil.copy2(self.video_path, dest_path)
            print(
                f"[ComplianceVLM] Warning: failed to reopen video; copied {dest_path}."
            )
            return

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(dest_path), fourcc, actual_fps, (width, height))
        if not writer.isOpened():
            writer.release()
            cap.release()
            shutil.copy2(self.video_path, dest_path)
            print(
                "[ComplianceVLM] Warning: failed to open re-encode writer; copied video."
            )
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            writer.write(frame)

        writer.release()
        cap.release()
        print(
            f"[ComplianceVLM] Saved camera video at {actual_fps:.2f} FPS to {dest_path}"
        )

    def get_prediction_output_dir(self, prediction_idx: int) -> Optional[str]:
        if self.debug_output_dir is None:
            return None
        path = Path(self.debug_output_dir.name) / f"prediction_{prediction_idx}"
        path.mkdir(parents=True, exist_ok=True)
        return str(path)

    def _get_head_pose(
        self,
        head_pos_world: Optional[npt.NDArray[np.float32]],
        head_quat_world_wxyz: Optional[npt.NDArray[np.float32]],
    ) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        pos = (
            np.asarray(head_pos_world, dtype=np.float32).reshape(3)
            if head_pos_world is not None
            else self.default_head_pos_world.copy()
        )
        quat = (
            np.asarray(head_quat_world_wxyz, dtype=np.float32).reshape(4)
            if head_quat_world_wxyz is not None
            else self.default_head_quat_world_wxyz.copy()
        )
        return pos, quat

    def maybe_start_prediction(self, inp: ComplianceVLMInput) -> None:
        if not self.prediction_requested or self.prediction_future is not None:
            return
        if self.predictor is None:
            return

        head_pos, head_quat = self._get_head_pose(
            inp.head_pos_world, inp.head_quat_world_wxyz
        )

        left_image = inp.left_image
        if left_image is None:
            left_image = self.last_left_frame
        if left_image is None and self.left_camera is not None:
            try:
                left_image = self.left_camera.get_frame()
            except Exception:
                left_image = None

        right_image = inp.right_image
        if right_image is None:
            right_image = self.last_right_frame
        if right_image is None and self.right_camera is not None:
            try:
                right_image = self.right_camera.get_frame()
            except Exception:
                right_image = None

        if left_image is None or right_image is None:
            return

        output_dir = self.get_prediction_output_dir(self.prediction_counter)
        pose_cur = dict(
            zip(
                self.target_site_names,
                self.pose_command[self.target_site_indices],
                strict=False,
            )
        )

        self.prediction_future = self.prediction_executor.submit(
            self.run_prediction_pipeline,
            head_pos,
            head_quat,
            np.asarray(left_image),
            np.asarray(right_image),
            output_dir,
            pose_cur,
            self.status == "wiping",
        )
        self.prediction_counter += 1
        self.prediction_requested = False

    def request_prediction_after_completion(self) -> None:
        if self.status == "wiping":
            self.prediction_requested = True

    def run_prediction_pipeline(
        self,
        head_pos: npt.NDArray[np.float32],
        head_quat: npt.NDArray[np.float32],
        left_image: np.ndarray,
        right_image: np.ndarray,
        output_dir: Optional[str],
        pose_cur: Dict[str, np.ndarray],
        is_wiping: bool,
    ) -> Optional[Dict[str, Tuple[np.ndarray, ...]]]:
        if self.predictor is None:
            return None

        self.args_payload["head_position"] = head_pos.tolist()
        self.args_payload["head_orientation"] = head_quat.tolist()

        try:
            result = self.predictor.predict(
                left_image=left_image,
                right_image=right_image,
                robot_name=self.robot_name,
                site_names=self.target_site_names,
                is_wiping=is_wiping,
                output_dir=output_dir,
                object_label=self.target_object_label,
            )
            wiping_done = is_wiping and self.predictor.last_wiping_done
        except Exception as exc:
            print(f"[ComplianceVLM] Affordance prediction failed: {exc}")
            return None

        if not result:
            if wiping_done:
                print("[ComplianceVLM] Predictor reports wiping is already complete.")
            else:
                print("[ComplianceVLM] Predictor returned no result.")
            return None

        contact_points_3d, contact_normals = result
        return plan_end_effector_poses(
            contact_points_camera=contact_points_3d,
            contact_normals_camera=contact_normals,
            head_position_world=head_pos,
            head_quaternion_world_wxyz=head_quat,
            tangent_pos_stiffness=self.tangent_pos_stiffness,
            normal_pos_stiffness=self.normal_pos_stiffness,
            tangent_rot_stiffness=self.tangent_rot_stiffness,
            normal_rot_stiffness=self.normal_rot_stiffness,
            contact_force=np.asarray(self.fixed_contact_force, dtype=np.float32),
            pose_cur=pose_cur,
            output_dir=output_dir,
            traj_dt=self.control_dt,
            traj_v_max_contact=self.traj_v_max_contact,
            traj_v_max_free=self.traj_v_max_free,
            tool=self.tool,
            robot_name=self.robot_name,
            mass=self.mass,
            inertia_diag=self.inertia_diag,
        )

    def prepare_fixed_plan(self, inp: ComplianceVLMInput) -> bool:
        path = self.get_fixed_trajectory_path()
        if path is None or not path.exists():
            self.use_fixed_trajectory = False
            return False

        try:
            payload = joblib.load(path)
            contact_points_camera = payload["contact_pos_camera"]
            contact_normals_camera = payload["contact_normals_camera"]
        except Exception as exc:
            print(f"[ComplianceVLM] Failed to load fixed trajectory {path}: {exc}")
            self.use_fixed_trajectory = False
            return False

        head_pos, head_quat = self._get_head_pose(
            inp.head_pos_world, inp.head_quat_world_wxyz
        )
        output_dir = self.get_prediction_output_dir(self.prediction_counter)

        try:
            plans_by_site = plan_end_effector_poses(
                contact_points_camera=contact_points_camera,
                contact_normals_camera=contact_normals_camera,
                head_position_world=head_pos,
                head_quaternion_world_wxyz=head_quat,
                tangent_pos_stiffness=self.tangent_pos_stiffness,
                normal_pos_stiffness=self.normal_pos_stiffness,
                tangent_rot_stiffness=self.tangent_rot_stiffness,
                normal_rot_stiffness=self.normal_rot_stiffness,
                contact_force=np.asarray(self.fixed_contact_force, dtype=np.float32),
                pose_cur=dict(
                    zip(
                        self.target_site_names,
                        self.pose_command[self.target_site_indices],
                        strict=False,
                    )
                ),
                output_dir=output_dir,
                traj_dt=self.control_dt,
                traj_v_max_contact=self.traj_v_max_contact,
                traj_v_max_free=self.traj_v_max_free,
                tool=self.tool,
                robot_name=self.robot_name,
                mass=self.mass,
                inertia_diag=self.inertia_diag,
            )
        except Exception as exc:
            print(
                f"[ComplianceVLM] Failed to build fixed plan with contact force {self.fixed_contact_force}: {exc}"
            )
            self.use_fixed_trajectory = False
            return False

        self.trajectory_plans = plans_by_site
        self.use_fixed_trajectory = False
        self.fixed_trajectory_active = True
        self.traj_start_time = None
        return True

    def _build_command_matrix(
        self, x_wrench: Optional[npt.NDArray[np.float32]]
    ) -> npt.NDArray[np.float32]:
        command_matrix = np.zeros(
            (self.num_sites, COMMAND_LAYOUT.width), dtype=np.float32
        )
        command_matrix[:, COMMAND_LAYOUT.position] = self.pose_command[:, :3]
        command_matrix[:, COMMAND_LAYOUT.orientation] = self.pose_command[:, 3:6]
        command_matrix[:, COMMAND_LAYOUT.kp_pos] = self.pos_stiffness
        command_matrix[:, COMMAND_LAYOUT.kp_rot] = self.rot_stiffness
        command_matrix[:, COMMAND_LAYOUT.kd_pos] = self.pos_damping
        command_matrix[:, COMMAND_LAYOUT.kd_rot] = self.rot_damping
        command_matrix[:, COMMAND_LAYOUT.force] = self.wrench_command[:, :3]
        command_matrix[:, COMMAND_LAYOUT.torque] = self.wrench_command[:, 3:6]

        if x_wrench is not None:
            wrench_arr = np.asarray(x_wrench, dtype=np.float32)
            if wrench_arr.ndim == 2 and wrench_arr.shape[1] >= 6:
                rows = min(self.num_sites, wrench_arr.shape[0])
                command_matrix[:rows, COMMAND_LAYOUT.measured_force] = wrench_arr[
                    :rows, :3
                ]
                command_matrix[:rows, COMMAND_LAYOUT.measured_torque] = wrench_arr[
                    :rows, 3:6
                ]

        return command_matrix

    def _trajectory_status(self, in_pause: bool) -> str:
        if self.status == "waiting":
            return "waiting"
        if in_pause:
            return f"{self.status}_pause"
        if self.trajectory_plans:
            return f"{self.status}_trajectory"
        if self.prediction_future is not None:
            return f"{self.status}_predicting"
        if self.prediction_requested:
            return f"{self.status}_requesting_prediction"
        return self.status

    def _step_vlm_core(self, inp: ComplianceVLMInput) -> ComplianceVLMOutput:
        self._ensure_pose_from_obs(inp.x_obs)
        self.check_mode_command(inp.mode_command)

        if self.status == "waiting":
            self.wipe_pause_end_time = None
            command_matrix = self._build_command_matrix(inp.x_wrench)
            return ComplianceVLMOutput(
                status="waiting",
                pose_command=self.pose_command.copy(),
                command_matrix=command_matrix,
                trajectory_active=False,
            )

        if self.status != "wiping":
            self.wipe_pause_end_time = None
        elif (
            self.wipe_pause_end_time is not None
            and inp.time >= self.wipe_pause_end_time
        ):
            self.wipe_pause_end_time = None
            self.trajectory_plans = {}
            self.traj_start_time = None
            if self.prediction_future is None:
                self.request_prediction_after_completion()

        in_pause = (
            self.status == "wiping"
            and self.wipe_pause_end_time is not None
            and inp.time < self.wipe_pause_end_time
        )

        if (
            not self.trajectory_plans
            and self.prediction_future is None
            and not in_pause
        ):
            if self.use_fixed_trajectory:
                self.prepare_fixed_plan(inp)
            else:
                self.prediction_requested = True

        if not in_pause:
            self.maybe_start_prediction(inp)

        if self.prediction_future is not None and self.prediction_future.done():
            result = self.prediction_future.result()
            wiping_done = (
                self.status == "wiping"
                and self.predictor is not None
                and self.predictor.last_wiping_done
            )
            self.prediction_future = None
            if result is None:
                if wiping_done:
                    print(
                        "[ComplianceVLM] Wiping complete; stopping affordance prediction requests."
                    )
                    self.prediction_requested = False
                    self.status = "waiting"
                    self.wiping_complete = True
                    self.wipe_pause_end_time = None
                else:
                    print(
                        "[ComplianceVLM] Affordance predictor returned no "
                        f"{self.target_object_label} to wipe; waiting for a new target."
                    )
                    self.prediction_requested = True
                self.trajectory_plans = {}
                self.traj_start_time = None
            else:
                self.trajectory_plans = result
                self.traj_start_time = None
                self.prediction_requested = False

        if self.trajectory_plans:
            if self.traj_start_time is None:
                self.traj_start_time = float(inp.time)

            elapsed = max(0.0, float(inp.time) - float(self.traj_start_time))
            trajectory_indices: Dict[str, Tuple[int, int]] = {}

            for site_idx, site_name in zip(
                self.target_site_indices, self.target_site_names, strict=False
            ):
                plan = self.trajectory_plans.get(site_name)
                if plan is None:
                    continue
                (
                    time_samples,
                    _,
                    ee_pos,
                    ee_ori,
                    pos_stiffness,
                    rot_stiffness,
                    pos_damping,
                    rot_damping,
                    command_forces,
                ) = plan

                idx = np.searchsorted(time_samples, elapsed, side="right") - 1
                idx = int(np.clip(idx, 0, len(time_samples) - 1))
                trajectory_indices[site_name] = (idx, len(time_samples))

                self.pose_command[site_idx, 0:3] = ee_pos[idx]
                self.pose_command[site_idx, 3:6] = ee_ori[idx]
                self.pos_stiffness[site_idx] = pos_stiffness[idx].flatten()
                self.rot_stiffness[site_idx] = rot_stiffness[idx].flatten()
                self.pos_damping[site_idx] = pos_damping[idx].flatten()
                self.rot_damping[site_idx] = rot_damping[idx].flatten()
                self.wrench_command[site_idx, 0:3] = command_forces[idx]

            traj_done = bool(trajectory_indices) and all(
                idx >= length - 1 for idx, length in trajectory_indices.values()
            )
            if traj_done and self.status == "wiping" and not self.prediction_future:
                if self.fixed_trajectory_active:
                    self.status = "waiting"
                    self.trajectory_plans = {}
                    self.traj_start_time = None
                    self.wipe_pause_end_time = None
                    self.fixed_trajectory_active = False
                elif self.wipe_pause_duration > 0.0:
                    if self.wipe_pause_end_time is None:
                        self.wipe_pause_end_time = (
                            float(inp.time) + self.wipe_pause_duration
                        )
                else:
                    self.request_prediction_after_completion()
            elif traj_done and self.status == "drawing":
                self.status = "waiting"
                self.trajectory_plans = {}
                self.traj_start_time = None

        if self.record_video:
            self.log_camera_frame(float(inp.time))

        in_pause_now = (
            self.status == "wiping"
            and self.wipe_pause_end_time is not None
            and inp.time < self.wipe_pause_end_time
        )
        command_matrix = self._build_command_matrix(inp.x_wrench)
        return ComplianceVLMOutput(
            status=self._trajectory_status(in_pause_now),
            pose_command=self.pose_command.copy(),
            command_matrix=command_matrix,
            trajectory_active=bool(self.trajectory_plans),
        )

    def _close_vlm_core(self, exp_folder_path: str = "") -> None:
        if self.mode_control_receiver is not None:
            self.mode_control_receiver.close()
            self.mode_control_receiver = None

        if self.left_camera is not None:
            self.left_camera.close()
            self.left_camera = None
        if self.right_camera is not None:
            self.right_camera.close()
            self.right_camera = None

        if self.predictor is not None:
            try:
                self.predictor.close()
            except Exception:
                pass

        if self.debug_output_dir is not None:
            if exp_folder_path:
                src = Path(self.debug_output_dir.name)
                if src.exists():
                    dest_root = Path(exp_folder_path)
                    dest_root.mkdir(parents=True, exist_ok=True)

                    payload_str = json.dumps(self.args_payload, indent=2)
                    copied_prediction_names = set()
                    for idx in range(self.prediction_counter):
                        prediction_src = src / f"prediction_{idx}"
                        if not prediction_src.exists():
                            continue
                        dest_path = dest_root / prediction_src.name
                        if dest_path.exists():
                            if dest_path.is_dir():
                                shutil.rmtree(dest_path)
                            else:
                                dest_path.unlink()
                        shutil.copytree(prediction_src, dest_path)
                        copied_prediction_names.add(prediction_src.name)
                        args_path = dest_path / "args.json"
                        try:
                            args_path.write_text(payload_str)
                        except Exception as exc:
                            print(
                                f"[ComplianceVLM] Warning: failed to write args.json: {exc}"
                            )

                    for item in src.iterdir():
                        if item.name in copied_prediction_names:
                            continue
                        dest_path = dest_root / item.name
                        if item.is_dir():
                            if dest_path.exists():
                                if dest_path.is_dir():
                                    shutil.rmtree(dest_path)
                                else:
                                    dest_path.unlink()
                            shutil.copytree(item, dest_path)
                        else:
                            shutil.copy2(item, dest_path)

            self.debug_output_dir.cleanup()
            self.debug_output_dir = None

        if exp_folder_path:
            self.export_camera_video(Path(exp_folder_path))
        self.discard_video_recording()

        if self.prediction_future is not None:
            self.prediction_future.cancel()
            self.prediction_future = None
        self.prediction_executor.shutdown(wait=False)

    def _build_x_obs(self) -> np.ndarray:
        x_obs = np.zeros((self.num_sites, 6), dtype=np.float32)
        for i, site in enumerate(self.site_names):
            site_id = self.controller.wrench_sim.site_ids[site]
            x_obs[i, :3] = np.asarray(self.data.site_xpos[site_id], dtype=np.float32)
            rot = np.asarray(self.data.site_xmat[site_id], dtype=np.float32).reshape(
                3, 3
            )
            x_obs[i, 3:6] = R.from_matrix(rot).as_rotvec().astype(np.float32)
        return x_obs

    def step(self, obs: Any, sim: Any) -> tuple[dict[str, float], np.ndarray]:
        del sim
        if "qpos" in obs:
            self.controller.wrench_sim.set_qpos(
                np.asarray(obs["qpos"], dtype=np.float32)
            )
            self.controller.wrench_sim.forward()
        elif "motor_pos" in obs:
            self.controller.wrench_sim.set_motor_angles(
                np.asarray(obs["motor_pos"], dtype=np.float32)
            )
            self.controller.wrench_sim.forward()
        left_image = obs.get("left_image", obs.get("image"))
        right_image = obs.get("right_image", left_image)
        if left_image is None:
            left = np.zeros(
                (int(self.args.image_height), int(self.args.image_width), 3),
                dtype=np.uint8,
            )
        else:
            left = _to_hwc_u8(
                np.asarray(left_image),
                size_hw=(int(self.args.image_height), int(self.args.image_width)),
            )
        if right_image is None:
            right = left.copy()
        else:
            right = _to_hwc_u8(
                np.asarray(right_image),
                size_hw=(int(self.args.image_height), int(self.args.image_width)),
            )

        x_obs = self._build_x_obs()
        vlm_out = self._step_vlm_core(
            self._vlm_input_cls(
                time=float(obs.get("time", self.data.time)),
                x_obs=x_obs,
                left_image=left,
                right_image=right,
            )
        )

        cmd = np.asarray(vlm_out.command_matrix, dtype=np.float32)
        if cmd.shape != (self.num_sites, COMMAND_LAYOUT.width):
            raise ValueError(
                f"VLM command_matrix shape {cmd.shape} != ({self.num_sites}, {COMMAND_LAYOUT.width})"
            )
        motor_tor_obs = np.asarray(obs["motor_tor"], dtype=np.float32)
        step_kwargs: dict[str, np.ndarray] = {}
        if "qpos" in obs:
            step_kwargs["qpos"] = np.asarray(obs["qpos"], dtype=np.float32)
        elif "motor_pos" in obs:
            step_kwargs["motor_pos"] = np.asarray(obs["motor_pos"], dtype=np.float32)
        _, state_ref = self.controller.step(
            command_matrix=cmd,
            motor_torques=motor_tor_obs,
            **step_kwargs,
        )
        if state_ref is not None:
            self.target_motor_pos = np.asarray(state_ref.motor_pos, dtype=np.float32)
        return {}, self.target_motor_pos.copy()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._close_vlm_core()
