"""Compliance policy driven by diffusion-predicted pose commands.

This keeps the old policy class structure but removes toddlerbot dependencies.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import queue
import shutil
import tempfile
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

import cv2
import joblib
import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

from diffusion_policy.dp_model import DPModel
from examples.compliance import CompliancePolicy


@dataclass(frozen=True)
class DPConfig:
    use_ddpm: bool
    diffuse_steps: int
    action_horizon: int
    obs_horizon: int
    image_horizon: int
    lowdim_obs_dim: int
    input_channels: int
    obs_source: Optional[List[str]]
    action_source: Optional[List[str]]

    @classmethod
    def from_model(cls, model: DPModel) -> "DPConfig":
        return cls(
            use_ddpm=bool(model.use_ddpm),
            diffuse_steps=int(model.diffuse_steps),
            action_horizon=int(model.action_horizon),
            obs_horizon=int(model.obs_horizon),
            image_horizon=int(model.image_horizon),
            lowdim_obs_dim=int(model.lowdim_obs_dim),
            input_channels=int(model.input_channels),
            obs_source=model.obs_source,
            action_source=model.action_source,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compliance DP policy")
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--num-sites", type=int, default=0)
    parser.add_argument("--image-height", type=int, default=96)
    parser.add_argument("--image-width", type=int, default=96)
    parser.add_argument("--kp-pos", type=float, default=100.0)
    parser.add_argument("--kp-rot", type=float, default=10.0)
    parser.add_argument("--use-camera-stream", action="store_true")
    return parser


def put_latest(queue_obj: mp.Queue, payload) -> None:
    try:
        queue_obj.put_nowait(payload)
    except queue.Full:
        try:
            queue_obj.get_nowait()
        except queue.Empty:
            return
        try:
            queue_obj.put_nowait(payload)
        except queue.Full:
            pass


def run_dp_inference_process(
    ckpt_path: str,
    use_ddpm: bool,
    diffuse_steps: int,
    action_horizon: Optional[int],
    action_drop: int,
    input_queue: mp.Queue,
    output_queue: mp.Queue,
    stop_event,
) -> None:
    dp_model = DPModel(
        ckpt_path,
        use_ddpm=bool(use_ddpm),
        diffuse_steps=int(diffuse_steps),
        action_horizon=action_horizon,
    )
    put_latest(output_queue, ("config", DPConfig.from_model(dp_model).__dict__))

    while not stop_event.is_set():
        try:
            obs_window, image_window, obs_time = input_queue.get(timeout=0.1)
        except queue.Empty:
            continue

        try:
            t1 = time.monotonic()
            action_seq = list(
                dp_model.get_action_from_obs(obs_window, image_deque=image_window)
            )
            t2 = time.monotonic()
        except Exception as exc:
            put_latest(output_queue, ("error", str(exc), float(obs_time)))
            continue

        drop_count = max(0, min(int(action_drop), len(action_seq)))
        action_seq = action_seq[drop_count:]
        put_latest(
            output_queue,
            ("action", action_seq, float(obs_time), float(t2 - t1), int(drop_count)),
        )


class ComplianceDPPolicy(CompliancePolicy):
    """Compliance policy that updates pose commands using a diffusion model."""

    def __init__(
        self,
        name: str,
        robot: str,
        init_motor_pos: npt.ArrayLike,
        ckpt: str = "",
        ip: str = "",
        sim: str = "real",
        vis: bool = False,
        plot: bool = False,
        dt: float = 0.02,
        num_sites: int = 0,
        image_height: int = 96,
        image_width: int = 96,
        kp_pos: float = 100.0,
        kp_rot: float = 10.0,
        use_camera_stream: bool = False,
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
        if not ckpt:
            raise ValueError("ComplianceDPPolicy requires ckpt path.")

        self.use_camera_stream = bool(use_camera_stream)
        self.image_height = int(image_height)
        self.image_width = int(image_width)
        self.args = argparse.Namespace(
            ckpt=str(ckpt),
            dt=float(dt),
            num_sites=int(num_sites),
            image_height=self.image_height,
            image_width=self.image_width,
            kp_pos=float(kp_pos),
            kp_rot=float(kp_rot),
        )

        if int(num_sites) > 0 and int(num_sites) != self.num_sites:
            raise ValueError(
                f"num_sites {num_sites} does not match configured site count {self.num_sites}."
            )

        self.set_stiffness(
            [float(kp_pos), float(kp_pos), float(kp_pos)],
            [float(kp_rot), float(kp_rot), float(kp_rot)],
        )

        self.camera = None
        if self.use_camera_stream:
            try:
                from real_world.camera import Camera

                self.camera = Camera("left")
            except Exception as exc:
                self.camera = None
                print(f"[ComplianceDP] camera stream disabled: {exc}")

        self.action_dt = float(dt)
        self.action_drop = 0
        self.interpolate_action = True
        self.action_blend_alpha = 0.9
        self.action_blend_min_alpha = 0.1
        self.action_blend_ramp_steps = 3
        self.action_blend_ramp_s = self.action_dt * float(self.action_blend_ramp_steps)

        self.expected_channels = 1
        self.lowdim_obs_dim = 0
        self.dp_action_horizon = 1
        self.action_max_age_s = self.action_dt * 2.0

        self.obs_source: Optional[List[str]] = None
        self.action_source: Optional[List[str]] = None

        self.obs_deque: deque = deque([], maxlen=1)
        self.image_deque: deque = deque([], maxlen=1)
        self.model_action_seq: List[Tuple[float, npt.NDArray[np.float32]]] = []
        self.action_seq_timestamp = 0.0
        self.last_status = "init"
        self.action_dim_warned = False
        self.reset_time: Optional[np.ndarray] = None
        self.reset_action: Optional[np.ndarray] = None
        self.reset_start_time: Optional[float] = None
        self.pending_eval_prompt = False
        self.pending_eval_duration: Optional[float] = None
        self.eval_start_time: Optional[float] = None
        self.defer_eval_start = False
        self.trial_start_mono = time.monotonic()
        self.eval_durations: List[float] = []
        self.eval_success_flags: List[bool] = []
        self.eval_success_count = 0
        self.eval_total_count = 0
        self.eval_reset_count = 0
        self.eval_last_reset_time: Optional[float] = None
        self.action_delta_threshold = 0.02
        self.action_stall_duration_s = 1.0
        self.post_prep_grace_s = 30.0
        self.post_prep_time_out_s = 40.0
        self.low_action_start_time: Optional[float] = None
        self.last_model_action: Optional[np.ndarray] = None

        self.dp_output_time_buffer: deque = deque(maxlen=20000)
        self.dp_output_buffer: deque = deque(maxlen=20000)
        self.dp_output_batch_buffer: deque = deque(maxlen=20000)
        self.dp_output_last_time = -float("inf")
        self.dp_output_batch_id = -1

        self.record_video = bool(self.use_camera_stream)
        self.video_logging_active = False
        self.video_writer: Optional[cv2.VideoWriter] = None
        self.video_temp_dir: Optional[tempfile.TemporaryDirectory] = None
        self.video_path: Optional[Path] = None
        self.video_fps: Optional[float] = None
        self.last_camera_frame: Optional[np.ndarray] = None
        self.video_capture_thread: Optional[threading.Thread] = None
        self.video_capture_stop: Optional[threading.Event] = None
        self.video_frame_timestamps: list[float] = []

        ctx = mp.get_context("spawn")
        self.inference_input_queue = ctx.Queue(maxsize=1)
        self.inference_output_queue = ctx.Queue(maxsize=1)
        self.inference_stop_event = ctx.Event()
        self.inference_process = ctx.Process(
            target=run_dp_inference_process,
            name="ComplianceDPInference",
            daemon=True,
            args=(
                str(ckpt),
                True,
                10,
                None,
                self.action_drop,
                self.inference_input_queue,
                self.inference_output_queue,
                self.inference_stop_event,
            ),
        )
        self.inference_process.start()

        cfg = self.read_dp_config_from_process()
        self.obs_source = cfg.get("obs_source")
        self.action_source = cfg.get("action_source")

        self.expected_channels = int(cfg.get("input_channels", 1))
        self.lowdim_obs_dim = int(cfg.get("lowdim_obs_dim", 0))
        self.dp_action_horizon = int(cfg.get("action_horizon", 1))
        self.action_max_age_s = self.action_dt * float(self.dp_action_horizon + 1)

        self.obs_deque = deque([], maxlen=int(cfg.get("obs_horizon", 1)))
        self.image_deque = deque([], maxlen=int(cfg.get("image_horizon", 1)))

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
    ) -> "ComplianceDPPolicy":
        args = build_parser().parse_args(list(argv))
        return cls(
            name="compliance_dp",
            robot=robot,
            init_motor_pos=np.zeros(0, dtype=np.float32),
            sim=sim,
            vis=vis,
            plot=plot,
            ckpt=args.ckpt,
            dt=args.dt,
            num_sites=args.num_sites,
            image_height=args.image_height,
            image_width=args.image_width,
            kp_pos=args.kp_pos,
            kp_rot=args.kp_rot,
            use_camera_stream=args.use_camera_stream,
        )

    def _read_config_from_process(self, timeout_s: float = 30.0) -> dict[str, Any]:
        deadline = time.monotonic() + float(timeout_s)
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError(
                    "Timed out waiting for DP config from inference process"
                )
            try:
                payload = self.inference_output_queue.get(timeout=remaining)
            except queue.Empty:
                continue

            if (
                isinstance(payload, tuple)
                and len(payload) == 2
                and payload[0] == "config"
                and isinstance(payload[1], dict)
            ):
                return dict(payload[1])

            if (
                isinstance(payload, tuple)
                and len(payload) >= 2
                and payload[0] == "error"
            ):
                raise RuntimeError(f"Inference process init failed: {payload[1]}")

    def read_dp_config_from_process(self, timeout_s: float = 30.0) -> dict[str, Any]:
        return self._read_config_from_process(timeout_s=timeout_s)

    def get_expected_channels(self) -> int:
        return int(self.expected_channels)

    def _to_hwc_u8(self, image: np.ndarray) -> np.ndarray:
        arr = np.asarray(image)
        if arr.ndim != 3:
            arr = np.zeros((self.image_height, self.image_width, 3), dtype=np.uint8)
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

    def get_image_obs(self, obs: Any) -> Optional[np.ndarray]:
        if self.camera is not None:
            frame = None
            if (
                self.video_capture_thread is not None
                and self.video_capture_thread.is_alive()
            ):
                frame = self.last_camera_frame
            if frame is None:
                try:
                    frame = self.camera.get_frame()
                except Exception:
                    frame = None
            if frame is not None:
                return self._to_hwc_u8(frame)
        if getattr(obs, "image", None) is None:
            return None
        return self._to_hwc_u8(np.asarray(obs.image))

    def _prepare_image(self, image: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if image is None:
            return None

        arr = np.asarray(image)
        if arr.ndim == 2:
            resized = cv2.resize(arr, (128, 96))[:, 16:112]
            resized = resized.astype(np.float32)
            if float(resized.max()) > 1.0:
                resized /= 255.0
            return resized[None, :, :]

        if arr.ndim != 3:
            return None

        if self.expected_channels == 1:
            gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (128, 96))[:, 16:112]
            resized = resized.astype(np.float32)
            if float(resized.max()) > 1.0:
                resized /= 255.0
            return resized[None, :, :]

        frame_rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(frame_rgb, (128, 96))[:, 16:112]
        resized = resized.astype(np.float32)
        if float(resized.max()) > 1.0:
            resized /= 255.0
        return resized.transpose(2, 0, 1)

    def prepare_image(self, image: np.ndarray) -> Optional[np.ndarray]:
        return self._prepare_image(image)

    def get_x_wrench(self) -> np.ndarray:
        rows = []
        for site_name in self.wrench_site_names:
            wrench = self.wrenches_by_site.get(site_name)
            if wrench is None:
                rows.append(np.zeros(6, dtype=np.float32))
            else:
                rows.append(np.asarray(wrench, dtype=np.float32).reshape(6))
        return np.stack(rows, axis=0)

    def get_obs(
        self,
        obs: Any,
        x_obs: npt.NDArray[np.float32],
    ) -> npt.NDArray[np.float32]:
        if self.obs_source:
            components: list[np.ndarray] = []
            for source in self.obs_source:
                if source == "x_obs":
                    components.append(np.asarray(x_obs, dtype=np.float32).reshape(-1))
                elif source == "x_wrench":
                    components.append(self.get_x_wrench().reshape(-1))
                elif source == "obs_motor_pos":
                    components.append(
                        np.asarray(obs.motor_pos, dtype=np.float32).reshape(-1)
                    )
                else:
                    raise ValueError(f"Unsupported obs_source token: {source}")
            return np.concatenate(components, axis=0).astype(np.float32)

        x_obs_vec = np.asarray(x_obs, dtype=np.float32).reshape(-1)
        if self.lowdim_obs_dim <= 0:
            return x_obs_vec
        if x_obs_vec.size >= self.lowdim_obs_dim:
            return x_obs_vec[: self.lowdim_obs_dim]
        motor = np.asarray(obs.motor_pos, dtype=np.float32).reshape(-1)
        if motor.size >= self.lowdim_obs_dim:
            return motor[: self.lowdim_obs_dim]
        pad = np.zeros(self.lowdim_obs_dim - motor.size, dtype=np.float32)
        return np.concatenate([motor, pad])

    def _clear_queue(self, queue_obj: mp.Queue) -> None:
        while True:
            try:
                queue_obj.get_nowait()
            except queue.Empty:
                break

    def clear_queue(self, queue_obj: mp.Queue) -> None:
        self._clear_queue(queue_obj)

    def get_action_blend_alpha(self, timestamp: float, now: float) -> float:
        ramp_s = float(self.action_blend_ramp_s)
        if ramp_s <= 0.0:
            return float(self.action_blend_alpha)

        t = (timestamp - now) / ramp_s
        t = float(np.clip(t, 0.0, 1.0))
        min_alpha = min(self.action_blend_min_alpha, self.action_blend_alpha)
        return float(min_alpha + (self.action_blend_alpha - min_alpha) * t)

    def reset_diffusion_state(self) -> None:
        self.obs_deque.clear()
        self.image_deque.clear()
        self.model_action_seq = []
        self.action_seq_timestamp = 0.0
        self._clear_queue(self.inference_input_queue)
        self._clear_queue(self.inference_output_queue)
        self.action_dim_warned = False
        self.low_action_start_time = None
        self.last_model_action = None

    def _submit_inference_request(self, obs_time: float) -> None:
        payload = (list(self.obs_deque), list(self.image_deque), float(obs_time))
        put_latest(self.inference_input_queue, payload)

    def submit_inference_request(self, obs_time: float) -> None:
        self._submit_inference_request(obs_time)

    def _consume_inference_output(self, now: float) -> None:
        latest = None
        while True:
            try:
                latest = self.inference_output_queue.get_nowait()
            except queue.Empty:
                break

        if latest is None or not isinstance(latest, tuple):
            return
        if len(latest) >= 2 and latest[0] == "error":
            print(f"[ComplianceDP] inference error: {latest[1]}")
            return
        if not (len(latest) >= 5 and latest[0] == "action"):
            return

        _, action_seq, obs_time, _dur, _drop = latest
        if now - float(obs_time) > self.action_max_age_s:
            return

        base_time = float(obs_time)
        new_seq = [
            (base_time + idx * self.action_dt, np.asarray(action, dtype=np.float32))
            for idx, action in enumerate(action_seq)
        ]

        if not self.model_action_seq:
            self.model_action_seq = new_seq
        else:
            blended: List[Tuple[float, npt.NDArray[np.float32]]] = []
            prev_seq = self.model_action_seq
            prev_idx = 0
            tol = self.action_dt * 0.5
            for timestamp, action in new_seq:
                while (
                    prev_idx < len(prev_seq) and prev_seq[prev_idx][0] < timestamp - tol
                ):
                    prev_idx += 1
                if (
                    prev_idx < len(prev_seq)
                    and abs(prev_seq[prev_idx][0] - timestamp) <= tol
                ):
                    prev_action = np.asarray(prev_seq[prev_idx][1], dtype=np.float32)
                    if prev_action.shape == action.shape:
                        alpha = self.get_action_blend_alpha(timestamp, now)
                        merged = alpha * action + (1.0 - alpha) * prev_action
                        blended.append((timestamp, merged.astype(np.float32)))
                    else:
                        blended.append((timestamp, action))
                else:
                    blended.append((timestamp, action))
            self.model_action_seq = blended

        self.action_seq_timestamp = base_time
        self.dp_output_batch_id += 1
        self.append_dp_output_log(self.dp_output_batch_id)

    def consume_inference_output(self, now: float) -> None:
        self._consume_inference_output(now)

    def _action_to_pose_command(
        self, action: npt.NDArray[np.float32]
    ) -> Optional[npt.NDArray[np.float32]]:
        act = np.asarray(action, dtype=np.float32).reshape(-1)
        expected = self.num_sites * 6
        if act.size == expected:
            return act.reshape(self.num_sites, 6)
        if act.size == 6:
            if self.pose_command is None:
                pose = np.asarray(self.default_state.x_ref, dtype=np.float32).copy()
            else:
                pose = np.asarray(self.pose_command, dtype=np.float32).copy()
            pose[0] = act
            return pose
        return None

    def action_to_pose_command(
        self, action: npt.NDArray[np.float32]
    ) -> Optional[npt.NDArray[np.float32]]:
        return self._action_to_pose_command(action)

    def _interpolate_pose_action(
        self,
        action0: npt.NDArray[np.float32],
        action1: npt.NDArray[np.float32],
        alpha: float,
    ) -> npt.NDArray[np.float32]:
        a0 = np.asarray(action0, dtype=np.float32).reshape(-1)
        a1 = np.asarray(action1, dtype=np.float32).reshape(-1)
        if a0.size != a1.size:
            return a1
        if alpha <= 0.0:
            return a0
        if alpha >= 1.0:
            return a1

        p0 = self._action_to_pose_command(a0)
        p1 = self._action_to_pose_command(a1)
        if p0 is None or p1 is None:
            return (1.0 - alpha) * a0 + alpha * a1

        out = p0.copy()
        out[:, :3] = (1.0 - alpha) * p0[:, :3] + alpha * p1[:, :3]
        for idx in range(p0.shape[0]):
            key_rots = R.from_rotvec(np.stack([p0[idx, 3:6], p1[idx, 3:6]], axis=0))
            slerp = Slerp([0.0, 1.0], key_rots)
            out[idx, 3:6] = slerp([alpha]).as_rotvec()[0].astype(np.float32)
        return out.reshape(-1)

    def interpolate_pose_action(
        self,
        action0: npt.NDArray[np.float32],
        action1: npt.NDArray[np.float32],
        alpha: float,
    ) -> npt.NDArray[np.float32]:
        return self._interpolate_pose_action(action0, action1, alpha)

    def _select_action_for_time(self, now: float) -> npt.NDArray[np.float32]:
        if len(self.model_action_seq) == 1:
            return self.model_action_seq[0][1]
        if now <= self.model_action_seq[0][0]:
            return self.model_action_seq[0][1]
        if now >= self.model_action_seq[-1][0]:
            return self.model_action_seq[-1][1]

        for idx, (ts, action) in enumerate(self.model_action_seq):
            if now <= ts:
                prev_ts, prev_action = self.model_action_seq[idx - 1]
                if ts <= prev_ts:
                    return action
                if not self.interpolate_action:
                    return prev_action if (now - prev_ts) <= (ts - now) else action
                alpha = (now - prev_ts) / (ts - prev_ts)
                return self._interpolate_pose_action(prev_action, action, float(alpha))
        return self.model_action_seq[-1][1]

    def select_action_for_time(self, now: float) -> npt.NDArray[np.float32]:
        return self._select_action_for_time(now)

    def start_reset_motion(self, obs: Any) -> None:
        if self.reset_time is not None:
            return
        self.eval_reset_count += 1
        self.eval_last_reset_time = float(obs.time)
        if self.eval_start_time is not None:
            self.pending_eval_duration = float(obs.time - self.eval_start_time)
        self.eval_start_time = None
        self.reset_start_time = float(obs.time)

        start = np.asarray(obs.motor_pos, dtype=np.float32).reshape(-1)
        goal = np.asarray(self.ref_motor_pos, dtype=np.float32).reshape(-1)
        n_steps = max(2, int(round(2.0 / max(self.control_dt, 1e-3))))
        t_arr = np.linspace(0.0, 2.0, n_steps, dtype=np.float32)
        alpha = np.linspace(0.0, 1.0, n_steps, dtype=np.float32)[:, None]
        self.reset_action = (
            (1.0 - alpha) * start[None, :] + alpha * goal[None, :]
        ).astype(np.float32)
        self.reset_time = t_arr
        self.pending_eval_prompt = True
        self.reset_diffusion_state()

    def check_action_stall(self, obs: Any, action: npt.NDArray[np.float32]) -> None:
        if self.reset_time is not None or self.pending_eval_prompt:
            return
        obs_time = float(obs.time)
        action_vec = np.asarray(action, dtype=np.float32).reshape(-1)
        prep_duration = float(getattr(self, "prep_duration", 0.0))
        if self.eval_start_time is None:
            if self.defer_eval_start:
                self.eval_start_time = obs_time
                self.defer_eval_start = False
            else:
                if obs_time < prep_duration:
                    self.low_action_start_time = None
                    self.last_model_action = action_vec
                    return
                self.eval_start_time = obs_time

        if obs_time - self.eval_start_time < self.post_prep_grace_s:
            self.low_action_start_time = None
            self.last_model_action = action_vec
            return
        if obs_time - self.eval_start_time >= self.post_prep_time_out_s:
            self.start_reset_motion(obs)
            return

        if (
            self.last_model_action is None
            or self.last_model_action.shape != action_vec.shape
        ):
            self.last_model_action = action_vec
            self.low_action_start_time = None
            return

        compare_len = min(3, action_vec.size, self.last_model_action.size)
        delta = float(
            np.max(
                np.abs(action_vec[:compare_len] - self.last_model_action[:compare_len])
            )
        )
        if delta < self.action_delta_threshold:
            if self.low_action_start_time is None:
                self.low_action_start_time = obs_time
            elif obs_time - self.low_action_start_time >= self.action_stall_duration_s:
                self.start_reset_motion(obs)
                return
        else:
            self.low_action_start_time = None
        self.last_model_action = action_vec

    def prompt_eval_result(self, duration: Optional[float]) -> None:
        try:
            response = input("Is the last eval successful? [y/n]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            response = "n"
        success = response in ("y", "yes")
        self.eval_total_count += 1
        if success:
            self.eval_success_count += 1
        duration_val = float(duration) if duration is not None else float("nan")
        self.eval_durations.append(duration_val)
        self.eval_success_flags.append(success)
        self.pending_eval_duration = None

    def save_eval_results(self, exp_folder_path: str) -> None:
        if self.eval_total_count == 0 or not exp_folder_path:
            return
        os.makedirs(exp_folder_path, exist_ok=True)
        success_rate = self.eval_success_count / float(self.eval_total_count)
        valid = np.asarray(self.eval_durations, dtype=np.float32)
        valid = valid[~np.isnan(valid)]
        avg_duration = float(np.mean(valid)) if valid.size else float("nan")
        out_path = os.path.join(exp_folder_path, "eval.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(f"success_rate: {success_rate:.3f}\n")
            f.write(f"success_count: {self.eval_success_count}\n")
            f.write(f"eval_count: {self.eval_total_count}\n")
            f.write(f"avg_duration: {avg_duration:.3f}\n")
            f.write(f"reset_count: {self.eval_reset_count}\n")
            for idx, (ok, dur) in enumerate(
                zip(self.eval_success_flags, self.eval_durations, strict=False), start=1
            ):
                f.write(f"Trial {idx}: success={ok}, {float(dur):.3f} (duration)\n")

    def append_dp_output_log(self, batch_id: int) -> None:
        for timestamp, action in self.model_action_seq:
            if float(timestamp) <= float(self.dp_output_last_time):
                continue
            self.dp_output_time_buffer.append(float(timestamp))
            self.dp_output_buffer.append(
                np.asarray(action, dtype=np.float32).reshape(-1).copy()
            )
            self.dp_output_batch_buffer.append(int(batch_id))
            self.dp_output_last_time = float(timestamp)

    def save_dp_output_log(self, exp_folder_path: str) -> None:
        if not exp_folder_path:
            return
        times_full = np.asarray(self.dp_output_time_buffer, dtype=np.float64)
        if times_full.size == 0:
            return
        actions_full = np.asarray(list(self.dp_output_buffer), dtype=np.float32)
        batch_full = np.asarray(self.dp_output_batch_buffer, dtype=np.int64)
        min_len = min(times_full.size, actions_full.shape[0], batch_full.size)
        if min_len == 0:
            return
        log_data = {
            "time": times_full[-min_len:],
            "action": actions_full[-min_len:],
            "batch_id": batch_full[-min_len:],
            "action_dt": float(self.action_dt),
            "num_sites": int(self.num_sites),
        }
        os.makedirs(exp_folder_path, exist_ok=True)
        joblib.dump(
            log_data, os.path.join(exp_folder_path, "dp_output.lz4"), compress="lz4"
        )

    def start_video_logging(self) -> None:
        if self.video_logging_active:
            return
        if self.camera is None:
            return
        self.discard_video_recording()
        self.video_logging_active = True
        self.start_video_capture_thread()

    def ensure_video_writer(self, frame: np.ndarray) -> bool:
        if self.video_writer is not None:
            return True
        if self.video_temp_dir is None:
            self.video_temp_dir = tempfile.TemporaryDirectory(
                prefix="compliance_dp_video_"
            )
        self.video_path = Path(self.video_temp_dir.name) / "left_camera.mp4"
        height, width = frame.shape[:2]
        fps = max(1.0, float(1.0 / max(self.control_dt, 1e-3)))
        writer = cv2.VideoWriter(
            str(self.video_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            writer.release()
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
            name="ComplianceDPCapture",
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
        if self.camera is None:
            return
        capture_dt = max(float(self.control_dt), 1e-3)
        start_time = time.monotonic()
        next_time = start_time
        while True:
            if self.video_capture_stop is not None and self.video_capture_stop.is_set():
                break
            now = time.monotonic()
            if now < next_time:
                time.sleep(next_time - now)
            next_time += capture_dt
            try:
                frame = self.camera.get_frame()
            except Exception:
                self.last_camera_frame = None
                continue
            self.last_camera_frame = frame
            if not self.video_logging_active:
                continue
            if frame is None or not self.ensure_video_writer(frame):
                continue
            if self.video_writer is None:
                continue
            self.video_writer.write(frame)
            self.video_frame_timestamps.append(float(time.monotonic() - start_time))

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

    def export_camera_video(self, output_dir: Path) -> None:
        self.stop_video_capture_thread()
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
        if self.video_path is None or not self.video_path.exists():
            return
        output_dir.mkdir(parents=True, exist_ok=True)
        dest_path = output_dir / self.video_path.name
        if dest_path.exists():
            dest_path.unlink()
        shutil.copy2(self.video_path, dest_path)

    def _step_diffusion(self, obs: Any, x_obs: npt.NDArray[np.float32]) -> str:
        image = self.get_image_obs(obs)
        proc_img = self._prepare_image(image)
        if proc_img is None:
            self.reset_diffusion_state()
            return "no_image"

        obs_vec = self.get_obs(obs, x_obs)
        self.obs_deque.append(obs_vec)
        self.image_deque.append(proc_img)

        if len(self.obs_deque) < self.obs_deque.maxlen:
            return "warming_obs"
        if len(self.image_deque) < self.image_deque.maxlen:
            return "warming_image"

        now = float(time.monotonic() - self.trial_start_mono)
        self._consume_inference_output(now)
        self._submit_inference_request(now)

        if (
            self.model_action_seq
            and now - self.action_seq_timestamp > self.action_max_age_s
        ):
            self.model_action_seq = []

        if not self.model_action_seq:
            return "waiting_inference"

        action = self._select_action_for_time(now)
        pose_cmd = self._action_to_pose_command(action)
        if pose_cmd is None:
            if not self.action_dim_warned:
                print(
                    f"[ComplianceDP] Unexpected action dimension: {np.asarray(action).size}"
                )
                self.action_dim_warned = True
            return f"invalid_action_dim_{np.asarray(action).size}"

        self.pose_command = pose_cmd.astype(np.float32)
        self.check_action_stall(obs, np.asarray(action, dtype=np.float32))
        return "ok"

    def update_pose_command_from_obs(
        self, obs: Any, x_obs: npt.NDArray[np.float32]
    ) -> None:
        if self.reset_time is not None:
            return
        if self.pose_command is None:
            self.pose_command = np.asarray(x_obs, dtype=np.float32).copy()
        self.last_status = self._step_diffusion(obs, x_obs)

    def step(
        self,
        obs: Any,
        sim: Any,
    ) -> np.ndarray:
        qpos_obs = np.asarray(obs.qpos, dtype=np.float32)
        self.controller.sync_qpos(qpos_obs)
        x_obs = self.controller.get_x_obs()
        self.update_pose_command_from_obs(obs, x_obs)
        action = super().step(obs, sim)
        if self.reset_time is not None and self.reset_action is not None:
            elapsed = (
                float(obs.time - self.reset_start_time)
                if self.reset_start_time is not None
                else float(obs.time)
            )
            if elapsed < float(self.reset_time[-1]):
                idx = int(np.searchsorted(self.reset_time, elapsed, side="right") - 1)
                idx = int(np.clip(idx, 0, self.reset_action.shape[0] - 1))
                action = np.asarray(self.reset_action[idx], dtype=np.float32)
            else:
                action = np.asarray(self.ref_motor_pos, dtype=np.float32).copy()
                self.reset_time = None
                self.reset_action = None
                self.reset_start_time = None
                if self.pending_eval_prompt:
                    self.prompt_eval_result(self.pending_eval_duration)
                    self.pending_eval_prompt = False
                    self.eval_start_time = None
                    self.defer_eval_start = True
                    self.trial_start_mono = time.monotonic()
                    self.pose_command = None
                    self.reset_diffusion_state()
        if self.record_video:
            self.start_video_logging()
        return np.asarray(action, dtype=np.float32)

    def close(self, exp_folder_path: str = "") -> None:
        if self._closed:
            return
        self._closed = True

        if hasattr(self, "inference_stop_event"):
            self.inference_stop_event.set()
        if hasattr(self, "inference_process") and self.inference_process.is_alive():
            self.inference_process.join(timeout=1.0)
            if self.inference_process.is_alive():
                self.inference_process.terminate()
                self.inference_process.join(timeout=1.0)

        if hasattr(self, "inference_input_queue"):
            self._clear_queue(self.inference_input_queue)
            self.inference_input_queue.cancel_join_thread()
            self.inference_input_queue.close()

        if hasattr(self, "inference_output_queue"):
            self._clear_queue(self.inference_output_queue)
            self.inference_output_queue.cancel_join_thread()
            self.inference_output_queue.close()
        try:
            self.save_dp_output_log(exp_folder_path)
        except Exception as exc:
            print(f"[ComplianceDP] Failed to save DP output log: {exc}")
        try:
            self.save_eval_results(exp_folder_path)
        except Exception as exc:
            print(f"[ComplianceDP] Failed to save eval results: {exc}")
        if exp_folder_path:
            self.export_camera_video(Path(exp_folder_path))
        self.discard_video_recording()

        if self.camera is not None:
            try:
                self.camera.close()
            except Exception:
                pass

        super().close(exp_folder_path)
