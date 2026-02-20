"""Compliance affordance policy built on top of the base compliance controller.

This keeps old policy orchestration style while using local VLM + minimalist
compliance modules.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import joblib
import numpy as np
import numpy.typing as npt

from examples.compliance import CompliancePolicy
from real_world.camera import Camera
from vlm.affordance.affordance_predictor import AffordancePredictor
from vlm.affordance.plan_ee_pose import plan_end_effector_poses
from vlm.utils.comm_utils import ZMQNode

LEAP_DRAW_POS = np.array(
    [2.23, 0, 0, 0.4, 2.23, 0, 0, 0.4, 2.23, 0, 0, 0.4, 0.0, -1.57, 0.0, 0.0],
    dtype=np.float32,
)


class ComplianceVLMPolicy(CompliancePolicy):
    """Guides compliance references via affordance-predicted trajectories."""

    def __init__(
        self,
        name: str,
        robot: str,
        init_motor_pos: npt.ArrayLike,
        mode_control_port: int = 5591,
        ip: str = "",
        sim: str = "real",
        vis: bool = False,
        plot: bool = False,
        robot_name: str = "",
        site_names: str = "",
        mode: str = "waiting",
        object: str = "black ink. vase",
        disable_zmq: bool = False,
        use_camera_stream: bool = False,
        disable_video: bool = False,
        image_height: int = 480,
        image_width: int = 640,
        predictor_model: str = "gemini-2.5-pro",
        predictor_provider: str = "gemini",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            name=name,
            robot=robot,
            init_motor_pos=init_motor_pos,
            ip=ip,
            sim=sim,
            vis=vis,
            plot=plot,
            **kwargs,
        )

        self.args = argparse.Namespace(
            robot_name=robot_name,
            site_names=site_names,
            mode=mode,
            object=object,
            mode_control_port=int(mode_control_port),
            disable_zmq=bool(disable_zmq),
            use_camera_stream=bool(use_camera_stream),
            disable_video=bool(disable_video),
            image_height=int(image_height),
            image_width=int(image_width),
        )

        if robot_name:
            self.robot_name = str(robot_name)
        elif self.robot == "leap":
            self.robot_name = "leap_hand"
        elif self.robot == "arx":
            self.robot_name = "arx"
        else:
            self.robot_name = "toddlerbot_2xm"

        self.target_site_names: List[str] = []
        if len(site_names.strip()) > 0:
            self.target_site_names = [
                s.strip() for s in site_names.split(",") if s.strip()
            ]
        if not self.target_site_names:
            if self.robot == "leap":
                self.target_site_names = ["mf_tip"]
            elif self.robot == "arx":
                self.target_site_names = ["ee_site"]
            else:
                self.target_site_names = ["left_hand_center"]

        self.target_site_indices: List[int] = []
        for site_name in self.target_site_names:
            if site_name in self.wrench_site_names:
                self.target_site_indices.append(self.wrench_site_names.index(site_name))

        if self.robot == "leap":
            self.ref_motor_pos = LEAP_DRAW_POS.copy()
            self.normal_pos_stiffness = 20.0
            self.tangent_pos_stiffness = 200.0
            self.normal_rot_stiffness = 10.0
            self.tangent_rot_stiffness = 20.0
            self.fixed_contact_force = 0.2
        else:
            self.normal_pos_stiffness = 80.0
            self.tangent_pos_stiffness = 400.0
            self.normal_rot_stiffness = 40.0
            self.tangent_rot_stiffness = 40.0
            self.fixed_contact_force = 5.0

        self.status = "waiting"
        self.target_object_label = str(object)
        self.tool = "eraser"
        self.trajectory_plans: Dict[str, Tuple[np.ndarray, ...]] = {}
        self.traj_start_time: Optional[float] = None

        self.wipe_pause_duration = 2.0
        self.wipe_pause_end_time: Optional[float] = None
        self.prediction_requested = False
        self.prediction_counter = 0
        self.use_fixed_trajectory = False
        self.fixed_trajectory_active = False
        self.wiping_complete = False

        self.predictor: Optional[AffordancePredictor] = None
        try:
            self.predictor = AffordancePredictor(
                model=str(predictor_model),
                provider=str(predictor_provider),
            )
        except Exception as exc:
            self.predictor = None
            print(f"[ComplianceVLM] predictor disabled: {exc}")

        self.left_camera: Optional[Camera] = None
        self.right_camera: Optional[Camera] = None
        if bool(use_camera_stream):
            try:
                self.left_camera = Camera("left")
                self.right_camera = Camera("right")
            except Exception as exc:
                self.left_camera = None
                self.right_camera = None
                print(f"[ComplianceVLM] camera stream disabled: {exc}")

        self.mode_control_receiver: Optional[ZMQNode] = None
        if not bool(disable_zmq):
            try:
                self.mode_control_receiver = ZMQNode(
                    type="receiver", port=int(mode_control_port)
                )
                print(
                    f"[ComplianceVLM] Mode control listening on port {mode_control_port} (w=wipe, d=draw)."
                )
            except Exception as exc:
                self.mode_control_receiver = None
                print(f"[ComplianceVLM] mode control disabled: {exc}")

        self.debug_output_dir = tempfile.TemporaryDirectory(prefix="compliance_vlm_")
        self.prediction_executor = ThreadPoolExecutor(max_workers=1)
        self.prediction_future: Optional[Future] = None

        self.record_video = not bool(disable_video)
        self.video_logging_active = False
        self.video_writer: Optional[cv2.VideoWriter] = None
        self.video_temp_dir: Optional[tempfile.TemporaryDirectory] = None
        self.video_path: Optional[Path] = None
        self.video_fps: Optional[float] = None
        self.last_left_frame: Optional[np.ndarray] = None
        self.last_right_frame: Optional[np.ndarray] = None
        self.video_capture_thread: Optional[threading.Thread] = None
        self.video_capture_stop: Optional[threading.Event] = None
        self.video_frame_timestamps: List[float] = []

        self.set_stiffness(
            pos_stiffness=[400.0, 400.0, 400.0],
            rot_stiffness=[40.0, 40.0, 40.0],
        )

        if mode == "wiping":
            self.set_mode(True, object_label=None, site_names=None)
        elif mode == "drawing":
            self.set_mode(False, object_label=str(object), site_names=None)

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
        parser = argparse.ArgumentParser(description="Compliance VLM policy")
        parser.add_argument("--robot-name", type=str, default="toddlerbot_2xm")
        parser.add_argument("--site-names", type=str, default="")
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
        args = parser.parse_args(list(argv))
        return cls(
            name="compliance_vlm",
            robot=robot,
            init_motor_pos=np.zeros(0, dtype=np.float32),
            sim=sim,
            vis=vis,
            plot=plot,
            robot_name=args.robot_name,
            site_names=args.site_names,
            mode=args.mode,
            object=args.object,
            mode_control_port=args.mode_control_port,
            disable_zmq=args.disable_zmq,
            use_camera_stream=args.use_camera_stream,
            disable_video=args.disable_video,
            image_height=args.image_height,
            image_width=args.image_width,
        )

    def reset(self) -> None:
        self.traj_start_time = None
        self.trajectory_plans = {}
        self.wipe_pause_end_time = None
        self.prediction_requested = False
        if self.prediction_future is not None:
            self.prediction_future.cancel()
            self.prediction_future = None
        self.status = "waiting"
        self.fixed_trajectory_active = False
        self.wiping_complete = False

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
        if object_label is not None:
            self.target_object_label = str(object_label)
        if site_names is not None and len(site_names) > 0:
            self.target_site_names = [str(x) for x in site_names]
            self.target_site_indices = [
                self.wrench_site_names.index(s)
                for s in self.target_site_names
                if s in self.wrench_site_names
            ]
        self.trajectory_plans = {}
        self.traj_start_time = None
        self.prediction_requested = True
        self.fixed_trajectory_active = False
        self.wiping_complete = False

    def get_fixed_trajectory_path(self) -> Optional[Path]:
        if self.status == "wiping":
            path = Path("results") / "affordance_wipe" / "trajectory.lz4"
        elif self.status == "drawing":
            path = Path("results") / "affordance_draw" / "trajectory.lz4"
        else:
            return None
        return path if path.exists() else None

    def refresh_fixed_trajectory_flag(self) -> None:
        self.use_fixed_trajectory = self.get_fixed_trajectory_path() is not None

    def prepare_fixed_plan(self, obs: Any) -> None:
        path = self.get_fixed_trajectory_path()
        if path is None:
            self.refresh_fixed_trajectory_flag()
            return
        try:
            payload = joblib.load(path)
            plans = payload.get("trajectory_by_site")
            if isinstance(plans, dict):
                self.trajectory_plans = {
                    str(k): tuple(v)
                    for k, v in plans.items()
                    if k in self.target_site_names
                }
                self.traj_start_time = float(obs.time)
                self.fixed_trajectory_active = bool(self.trajectory_plans)
                self.use_fixed_trajectory = True
        except Exception as exc:
            print(f"[ComplianceVLM] failed to load fixed trajectory: {exc}")
            self.use_fixed_trajectory = False
            self.fixed_trajectory_active = False

    def get_prediction_output_dir(self, prediction_idx: int) -> Optional[str]:
        if self.debug_output_dir is None:
            return None
        base = Path(self.debug_output_dir.name)
        out_dir = base / f"prediction_{prediction_idx}"
        out_dir.mkdir(parents=True, exist_ok=True)
        return str(out_dir)

    def check_mode_command(self) -> None:
        if self.mode_control_receiver is None:
            return
        msg = self.mode_control_receiver.get_msg(return_last=True)
        if msg is None:
            return

        cmd = None
        if getattr(msg, "other", None) is not None:
            candidate = msg.other.get("command")
            if isinstance(candidate, str) and len(candidate) > 0:
                cmd = candidate.lower().strip()
        if cmd is None and getattr(msg, "control_inputs", None) is not None:
            candidate = msg.control_inputs.get("command")
            if isinstance(candidate, str) and len(candidate) > 0:
                cmd = candidate.lower().strip()
        if cmd is None:
            return

        if cmd.startswith("w"):
            self.set_mode(True)
        elif cmd.startswith("d"):
            self.set_mode(False)

    def _to_hwc_u8(self, image: np.ndarray) -> np.ndarray:
        arr = np.asarray(image)
        if arr.ndim != 3:
            arr = np.zeros(
                (int(self.args.image_height), int(self.args.image_width), 3),
                dtype=np.uint8,
            )
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

    def _get_stereo_images(self, obs: Any) -> tuple[np.ndarray, np.ndarray]:
        left = None
        right = None
        if self.left_camera is not None:
            try:
                left = self.left_camera.get_frame()
                self.last_left_frame = left
            except Exception:
                left = self.last_left_frame
        if self.right_camera is not None:
            try:
                right = self.right_camera.get_frame()
                self.last_right_frame = right
            except Exception:
                right = self.last_right_frame

        if left is None:
            left = getattr(obs, "left_image", None)
        if left is None:
            left = getattr(obs, "image", None)
        if right is None:
            right = getattr(obs, "right_image", None)
        if right is None:
            right = left

        if left is None:
            left = np.zeros(
                (int(self.args.image_height), int(self.args.image_width), 3),
                dtype=np.uint8,
            )
        if right is None:
            right = left

        return self._to_hwc_u8(np.asarray(left)), self._to_hwc_u8(np.asarray(right))

    def run_prediction_pipeline(
        self,
        head_pos: np.ndarray,
        head_quat: np.ndarray,
        left_image: np.ndarray,
        right_image: np.ndarray,
        output_dir: Optional[str],
        pose_cur_by_site: Dict[str, np.ndarray],
    ) -> Optional[Dict[str, Tuple[np.ndarray, ...]]]:
        if self.predictor is None:
            return None

        prediction = self.predictor.predict(
            left_image=left_image,
            right_image=right_image,
            robot_name=self.robot_name,
            site_names=self.target_site_names,
            is_wiping=(self.status == "wiping"),
            output_dir=output_dir,
            object_label=self.target_object_label,
        )
        if prediction is None:
            return None

        contact_points = {
            site_name: np.asarray(values[0], dtype=np.float32)
            for site_name, values in prediction.items()
            if site_name in self.target_site_names
        }
        contact_normals = {
            site_name: np.asarray(values[1], dtype=np.float32)
            for site_name, values in prediction.items()
            if site_name in self.target_site_names
        }
        if not contact_points:
            return None

        return plan_end_effector_poses(
            contact_points_camera=contact_points,
            contact_normals_camera=contact_normals,
            head_position_world=np.asarray(head_pos, dtype=np.float32),
            head_quaternion_world_wxyz=np.asarray(head_quat, dtype=np.float32),
            tangent_pos_stiffness=float(self.tangent_pos_stiffness),
            normal_pos_stiffness=float(self.normal_pos_stiffness),
            tangent_rot_stiffness=float(self.tangent_rot_stiffness),
            normal_rot_stiffness=float(self.normal_rot_stiffness),
            contact_force=float(self.fixed_contact_force),
            pose_cur=pose_cur_by_site,
            output_dir=output_dir,
            traj_dt=float(self.control_dt),
            traj_v_max_contact=0.02,
            traj_v_max_free=0.1,
            tool=self.tool,
            robot_name=self.robot_name,
            mass=float(self.mass),
            inertia_diag=np.asarray(self.inertia_diag, dtype=np.float32),
        )

    def maybe_start_prediction(self, obs: Any) -> None:
        if self.status == "waiting":
            return
        if self.prediction_future is not None:
            return
        if self.predictor is None:
            return
        if not self.prediction_requested and self.trajectory_plans:
            return
        if self.use_fixed_trajectory and not self.trajectory_plans:
            return

        left_image, right_image = self._get_stereo_images(obs)
        head_pos, head_quat = self.controller.get_head_pose()
        output_dir = self.get_prediction_output_dir(self.prediction_counter)

        pose_cur = {
            site_name: np.asarray(
                self.pose_command[self.wrench_site_names.index(site_name)],
                dtype=np.float32,
            )
            for site_name in self.target_site_names
            if site_name in self.wrench_site_names
        }
        self.prediction_future = self.prediction_executor.submit(
            self.run_prediction_pipeline,
            head_pos,
            head_quat,
            left_image,
            right_image,
            output_dir,
            pose_cur,
        )
        self.prediction_counter += 1
        self.prediction_requested = False

    def request_prediction_after_completion(self) -> None:
        if self.status == "wiping":
            self.prediction_requested = True

    def _consume_prediction(self) -> None:
        if self.prediction_future is None:
            return
        if not self.prediction_future.done():
            return
        try:
            result = self.prediction_future.result()
        except Exception as exc:
            print(f"[ComplianceVLM] prediction failed: {exc}")
            result = None
        self.prediction_future = None

        if result is None:
            if self.status == "wiping":
                self.wipe_pause_end_time = (
                    float(time.monotonic()) + self.wipe_pause_duration
                )
            return

        self.trajectory_plans = result
        self.traj_start_time = None

    def _apply_trajectory(self, now: float) -> None:
        if not self.trajectory_plans:
            return
        if self.traj_start_time is None:
            self.traj_start_time = float(now)

        elapsed = max(0.0, float(now) - float(self.traj_start_time))
        indices: Dict[str, Tuple[int, int]] = {}

        for site_name in self.target_site_names:
            if site_name not in self.wrench_site_names:
                continue
            plan = self.trajectory_plans.get(site_name)
            if plan is None:
                continue

            if len(plan) >= 9:
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
            else:
                (
                    time_samples,
                    _,
                    ee_pos,
                    ee_ori,
                    pos_stiffness,
                    rot_stiffness,
                    pos_damping,
                    rot_damping,
                ) = plan
                command_forces = np.zeros((len(time_samples), 3), dtype=np.float32)

            idx = np.searchsorted(time_samples, elapsed, side="right") - 1
            idx = int(np.clip(idx, 0, len(time_samples) - 1))
            indices[site_name] = (idx, len(time_samples))

            site_idx = self.wrench_site_names.index(site_name)
            self.pose_command[site_idx, 0:3] = np.asarray(ee_pos[idx], dtype=np.float32)
            self.pose_command[site_idx, 3:6] = np.asarray(ee_ori[idx], dtype=np.float32)
            self.pos_stiffness[site_idx] = np.asarray(
                pos_stiffness[idx], dtype=np.float32
            ).reshape(-1)
            self.rot_stiffness[site_idx] = np.asarray(
                rot_stiffness[idx], dtype=np.float32
            ).reshape(-1)
            self.pos_damping[site_idx] = np.asarray(
                pos_damping[idx], dtype=np.float32
            ).reshape(-1)
            self.rot_damping[site_idx] = np.asarray(
                rot_damping[idx], dtype=np.float32
            ).reshape(-1)
            self.wrench_command[site_idx, 0:3] = np.asarray(
                command_forces[idx], dtype=np.float32
            )

        if indices and all(idx >= length - 1 for idx, length in indices.values()):
            if self.status == "wiping":
                self.wiping_complete = True
                if self.fixed_trajectory_active:
                    self.status = "waiting"
                    self.wipe_pause_end_time = None
                    self.fixed_trajectory_active = False
                else:
                    self.wipe_pause_end_time = float(now) + self.wipe_pause_duration
            else:
                self.status = "waiting"
            self.trajectory_plans = {}
            self.traj_start_time = None

    def _start_video_logging(self) -> None:
        if self.video_logging_active or not self.record_video:
            return
        self.video_temp_dir = tempfile.TemporaryDirectory(
            prefix="compliance_vlm_video_"
        )
        self.video_path = Path(self.video_temp_dir.name) / "camera.mp4"
        self.video_fps = max(1.0, 1.0 / max(self.control_dt, 1e-3))
        self.video_logging_active = True

    def start_video_logging(self) -> None:
        self._start_video_logging()

    def ensure_video_writer(self, frame: np.ndarray) -> bool:
        if self.video_writer is not None:
            return True
        if self.video_path is None:
            return False
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            str(self.video_path),
            fourcc,
            float(self.video_fps if self.video_fps is not None else 30.0),
            (int(frame.shape[1]), int(frame.shape[0])),
        )
        if not writer.isOpened():
            writer.release()
            self.video_logging_active = False
            return False
        self.video_writer = writer
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
        capture_dt = max(float(self.control_dt), 1e-3)
        next_t = time.monotonic()
        while True:
            if self.video_capture_stop is not None and self.video_capture_stop.is_set():
                break
            now = time.monotonic()
            if now < next_t:
                time.sleep(next_t - now)
            next_t += capture_dt
            if self.left_camera is None:
                continue
            try:
                frame = self.left_camera.get_frame()
            except Exception:
                frame = None
            if frame is None or not self.video_logging_active:
                continue
            if not self.ensure_video_writer(frame):
                continue
            assert self.video_writer is not None
            if frame.ndim == 3:
                pair = frame
            else:
                pair = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            self.video_writer.write(pair)
            self.video_frame_timestamps.append(float(time.monotonic()))

    def _log_camera_frame(
        self, timestamp_s: float, left: np.ndarray, right: np.ndarray
    ) -> None:
        if not self.video_logging_active:
            return
        frame = np.hstack([left, right])
        if not self.ensure_video_writer(frame):
            return
        assert self.video_writer is not None
        self.video_writer.write(frame)
        self.video_frame_timestamps.append(float(timestamp_s))

    def log_camera_frame(
        self, timestamp_s: float, left: np.ndarray, right: np.ndarray
    ) -> None:
        self._log_camera_frame(timestamp_s, left, right)

    def export_camera_video(self, exp_dir: Path) -> None:
        self.stop_video_capture_thread()
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
        if self.video_path is None or not self.video_path.exists():
            return
        exp_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(self.video_path, exp_dir / "camera.mp4")

        timestamp_path = exp_dir / "camera_timestamps.json"
        with timestamp_path.open("w", encoding="utf-8") as f:
            json.dump(self.video_frame_timestamps, f, indent=2)

    def discard_video_recording(self) -> None:
        self.stop_video_capture_thread()
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
        if self.video_temp_dir is not None:
            self.video_temp_dir.cleanup()
            self.video_temp_dir = None
        self.video_path = None

    def step(
        self,
        obs: Any,
        sim: Any,
    ) -> npt.NDArray[np.float32]:
        action = super().step(obs, sim)
        self.check_mode_command()
        self.refresh_fixed_trajectory_flag()

        if self.status == "waiting":
            self.wipe_pause_end_time = None
            return np.asarray(action, dtype=np.float32)

        if self.status != "wiping":
            self.wipe_pause_end_time = None

        if self.status == "wiping" and self.wipe_pause_end_time is not None:
            if float(obs.time) >= float(self.wipe_pause_end_time):
                self.wipe_pause_end_time = None
                self.trajectory_plans = {}
                self.traj_start_time = None
                self.request_prediction_after_completion()
            else:
                self._consume_prediction()
                return np.asarray(action, dtype=np.float32)

        if self.use_fixed_trajectory and not self.trajectory_plans:
            self.prepare_fixed_plan(obs)
        self.maybe_start_prediction(obs)
        self._consume_prediction()
        self._apply_trajectory(float(obs.time))

        left, right = self._get_stereo_images(obs)
        if self.record_video:
            self.start_video_logging()
            self.log_camera_frame(float(obs.time), left, right)

        return np.asarray(action, dtype=np.float32)

    def close(self, exp_folder_path: str = "") -> None:
        if self.mode_control_receiver is not None:
            self.mode_control_receiver.close()
            self.mode_control_receiver = None

        if self.prediction_future is not None:
            self.prediction_future.cancel()
            self.prediction_future = None
        self.prediction_executor.shutdown(wait=False)

        if self.left_camera is not None:
            try:
                self.left_camera.close()
            except Exception:
                pass
            self.left_camera = None
        if self.right_camera is not None:
            try:
                self.right_camera.close()
            except Exception:
                pass
            self.right_camera = None

        if self.debug_output_dir is not None and exp_folder_path:
            exp_dir = Path(exp_folder_path)
            exp_dir.mkdir(parents=True, exist_ok=True)
            src = Path(self.debug_output_dir.name)
            if src.exists():
                dst = exp_dir / "affordance_debug"
                if dst.exists():
                    shutil.rmtree(dst)
                shutil.copytree(src, dst)

        if exp_folder_path:
            self.export_camera_video(Path(exp_folder_path))
        self.discard_video_recording()

        if self.debug_output_dir is not None:
            self.debug_output_dir.cleanup()
            self.debug_output_dir = None

        super().close(exp_folder_path)
