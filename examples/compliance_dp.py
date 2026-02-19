from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import queue
import time
from collections import deque
from typing import Any, List, Optional, Sequence, Tuple

import cv2
import gin
import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

from diffusion_policy.dp_model import DPModel
from minimalist_compliance_control.compliance_ref import COMMAND_LAYOUT
from minimalist_compliance_control.controller import (
    ComplianceController,
    ComplianceRefConfig,
    ControllerConfig,
)
from minimalist_compliance_control.wrench_estimation import WrenchEstimateConfig


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
    parser = argparse.ArgumentParser(description="Compliance DP per-tick policy")
    parser.add_argument("--ckpt", type=str, required=True, help="DP checkpoint path")
    parser.add_argument("--num-sites", type=int, default=0, help="Override site count")
    parser.add_argument("--dt", type=float, default=0.02, help="Policy control dt")
    parser.add_argument("--image-height", type=int, default=96)
    parser.add_argument("--image-width", type=int, default=96)
    parser.add_argument("--kp-pos", type=float, default=100.0)
    parser.add_argument("--kp-rot", type=float, default=10.0)
    return parser


def _dp_config_from_model(model: DPModel) -> dict[str, Any]:
    return {
        "use_ddpm": bool(model.use_ddpm),
        "diffuse_steps": int(model.diffuse_steps),
        "action_horizon": int(model.action_horizon),
        "obs_horizon": int(model.obs_horizon),
        "image_horizon": int(model.image_horizon),
        "lowdim_obs_dim": int(model.lowdim_obs_dim),
        "input_channels": int(model.input_channels),
        "obs_source": model.obs_source,
        "action_source": model.action_source,
    }


def put_latest(queue_obj: mp.Queue, payload) -> None:
    """Put latest payload and drop stale queue element when full."""
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


def _run_inference_process(
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
        use_ddpm=use_ddpm,
        diffuse_steps=diffuse_steps,
        action_horizon=action_horizon,
    )
    put_latest(output_queue, ("config", _dp_config_from_model(dp_model)))

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


class ComplianceDPPolicy:
    """Per-tick compliance DP policy with (obs, sim) -> (control_inputs, action)."""

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

        if int(args.num_sites) > 0 and int(args.num_sites) != self.num_sites:
            raise ValueError(
                f"--num-sites={args.num_sites} does not match controller site count {self.num_sites}."
            )

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

        self._init_dp_core(
            ckpt_path=str(args.ckpt),
            num_sites=self.num_sites,
            control_dt=float(args.dt),
        )

        self.command_matrix = np.zeros(
            (self.num_sites, COMMAND_LAYOUT.width), dtype=np.float32
        )
        kp_pos = float(args.kp_pos)
        kp_rot = float(args.kp_rot)
        self.command_matrix[:, COMMAND_LAYOUT.kp_pos] = (
            np.eye(3, dtype=np.float32) * kp_pos
        ).reshape(-1)
        self.command_matrix[:, COMMAND_LAYOUT.kd_pos] = (
            np.eye(3, dtype=np.float32) * (2.0 * np.sqrt(kp_pos))
        ).reshape(-1)
        self.command_matrix[:, COMMAND_LAYOUT.kp_rot] = (
            np.eye(3, dtype=np.float32) * kp_rot
        ).reshape(-1)
        self.command_matrix[:, COMMAND_LAYOUT.kd_rot] = (
            np.eye(3, dtype=np.float32) * (2.0 * np.sqrt(kp_rot))
        ).reshape(-1)
        self._closed = False

    def _init_dp_core(
        self,
        ckpt_path: str,
        num_sites: int,
        control_dt: float = 0.02,
        use_ddpm: bool = True,
        diffuse_steps: int = 10,
        action_horizon: Optional[int] = None,
        action_drop: int = 0,
        action_dt: float = 0.1,
        interpolate_action: bool = True,
        action_blend_alpha: float = 0.9,
        action_blend_min_alpha: float = 0.1,
        action_blend_ramp_steps: int = 3,
        pos_kp: float = 100.0,
        rot_kp: float = 10.0,
    ) -> None:
        if num_sites <= 0:
            raise ValueError("num_sites must be positive")

        self.num_sites = int(num_sites)
        self.control_dt = float(control_dt)
        self.action_dt = float(action_dt)
        self.action_drop = int(action_drop)

        self.interpolate_action = bool(interpolate_action)
        self.action_blend_alpha = float(action_blend_alpha)
        self.action_blend_min_alpha = float(action_blend_min_alpha)
        self.action_blend_ramp_steps = int(action_blend_ramp_steps)
        self.action_blend_ramp_s = self.action_dt * float(self.action_blend_ramp_steps)

        self.pose_command = np.zeros((self.num_sites, 6), dtype=np.float32)
        self.wrench_command = np.zeros((self.num_sites, 6), dtype=np.float32)

        self.pos_stiffness = np.tile(
            np.diag(np.array([pos_kp, pos_kp, pos_kp], dtype=np.float32)).reshape(1, 9),
            (self.num_sites, 1),
        )
        self.rot_stiffness = np.tile(
            np.diag(np.array([rot_kp, rot_kp, rot_kp], dtype=np.float32)).reshape(1, 9),
            (self.num_sites, 1),
        )
        self.pos_damping = np.tile(
            np.diag(
                2.0 * np.sqrt(np.array([pos_kp, pos_kp, pos_kp], dtype=np.float32))
            ).reshape(1, 9),
            (self.num_sites, 1),
        )
        self.rot_damping = np.tile(
            np.diag(
                2.0 * np.sqrt(np.array([rot_kp, rot_kp, rot_kp], dtype=np.float32))
            ).reshape(1, 9),
            (self.num_sites, 1),
        )

        ctx = mp.get_context("spawn")
        self.inference_input_queue = ctx.Queue(maxsize=1)
        self.inference_output_queue = ctx.Queue(maxsize=1)
        self.inference_stop_event = ctx.Event()
        self.inference_process = ctx.Process(
            target=_run_inference_process,
            name="ComplianceDPInference",
            daemon=True,
            args=(
                ckpt_path,
                bool(use_ddpm),
                int(diffuse_steps),
                action_horizon,
                self.action_drop,
                self.inference_input_queue,
                self.inference_output_queue,
                self.inference_stop_event,
            ),
        )
        self.inference_process.start()

        dp_cfg = self._read_config_from_process()
        self.obs_source = dp_cfg.get("obs_source")
        self.action_source = dp_cfg.get("action_source")
        self.use_compliance = not (
            self.action_source is not None and "x_ref" in self.action_source
        )

        self.expected_channels = int(dp_cfg.get("input_channels", 1))
        self.lowdim_obs_dim = int(dp_cfg.get("lowdim_obs_dim", 0))
        self.dp_action_horizon = int(dp_cfg.get("action_horizon", 1))
        self.action_max_age_s = self.action_dt * float(self.dp_action_horizon + 1)

        self.obs_deque: deque = deque([], maxlen=int(dp_cfg.get("obs_horizon", 1)))
        self.image_deque: deque = deque([], maxlen=int(dp_cfg.get("image_horizon", 1)))
        self.model_action_seq: List[Tuple[float, npt.NDArray[np.float32]]] = []
        self.action_seq_timestamp = 0.0

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

            raise RuntimeError(f"Unexpected payload while reading DP config: {payload}")

    def _prepare_image(self, image: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if image is None:
            return None

        arr = np.asarray(image)
        if arr.ndim == 3 and arr.shape[2] == 1:
            arr = arr[:, :, 0]

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

    def _get_obs_vector(
        self,
        *,
        x_obs: Optional[npt.NDArray[np.float32]],
        x_wrench: Optional[npt.NDArray[np.float32]],
        motor_pos: Optional[npt.NDArray[np.float32]],
    ) -> npt.NDArray[np.float32]:
        lowdim = int(self.lowdim_obs_dim)

        if self.obs_source:
            components: List[np.ndarray] = []
            for source in self.obs_source:
                if source == "x_obs":
                    if x_obs is None:
                        raise ValueError(
                            "obs_source contains x_obs but x_obs is missing"
                        )
                    components.append(np.asarray(x_obs, dtype=np.float32).reshape(-1))
                elif source == "x_wrench":
                    if x_wrench is None:
                        raise ValueError(
                            "obs_source contains x_wrench but x_wrench is missing"
                        )
                    components.append(
                        np.asarray(x_wrench, dtype=np.float32).reshape(-1)
                    )
                elif source == "obs_motor_pos":
                    if motor_pos is None:
                        raise ValueError(
                            "obs_source contains obs_motor_pos but motor_pos is missing"
                        )
                    components.append(
                        np.asarray(motor_pos, dtype=np.float32).reshape(-1)
                    )
                else:
                    raise ValueError(f"Unsupported obs_source token: {source}")

            obs_vec = (
                np.concatenate(components, axis=0)
                if len(components) > 1
                else components[0]
            )
            return obs_vec.astype(np.float32)

        if x_obs is not None:
            x_obs_vec = np.asarray(x_obs, dtype=np.float32).reshape(-1)
            if x_obs_vec.size == lowdim:
                return x_obs_vec

        if motor_pos is not None:
            motor = np.asarray(motor_pos, dtype=np.float32).reshape(-1)
            if motor.size >= lowdim:
                return motor[:lowdim]
            pad = np.zeros(lowdim - motor.size, dtype=np.float32)
            return np.concatenate([motor, pad])

        raise ValueError("Cannot build observation vector: provide x_obs or motor_pos")

    def _clear_queue(self, queue_obj: mp.Queue) -> None:
        while True:
            try:
                queue_obj.get_nowait()
            except queue.Empty:
                break

    def _submit_inference_request(self, obs_time: float) -> None:
        payload = (list(self.obs_deque), list(self.image_deque), float(obs_time))
        put_latest(self.inference_input_queue, payload)

    def _get_action_blend_alpha(self, timestamp: float, now: float) -> float:
        ramp_s = float(self.action_blend_ramp_s)
        if ramp_s <= 0.0:
            return float(self.action_blend_alpha)

        t = (timestamp - now) / ramp_s
        if t <= 0.0:
            weight = 0.0
        elif t >= 1.0:
            weight = 1.0
        else:
            weight = t

        min_alpha = min(self.action_blend_min_alpha, self.action_blend_alpha)
        return min_alpha + (self.action_blend_alpha - min_alpha) * weight

    def _consume_inference_output(self, now: float) -> None:
        latest = None
        while True:
            try:
                latest = self.inference_output_queue.get_nowait()
            except queue.Empty:
                break

        if latest is None:
            return
        if not isinstance(latest, tuple):
            return
        if len(latest) >= 2 and latest[0] == "error":
            print(f"[compliance_dp] inference error: {latest[1]}")
            return
        if not (len(latest) >= 5 and latest[0] == "action"):
            return

        _, action_seq, obs_time, _, _ = latest
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
            prev_len = len(prev_seq)
            tol = self.action_dt * 0.5
            for timestamp, action in new_seq:
                while prev_idx < prev_len and prev_seq[prev_idx][0] < timestamp - tol:
                    prev_idx += 1
                if (
                    prev_idx < prev_len
                    and abs(prev_seq[prev_idx][0] - timestamp) <= tol
                ):
                    prev_action = np.asarray(prev_seq[prev_idx][1], dtype=np.float32)
                    if prev_action.shape == action.shape:
                        alpha = self._get_action_blend_alpha(timestamp, now)
                        merged = alpha * action + (1.0 - alpha) * prev_action
                        blended.append((timestamp, merged.astype(np.float32)))
                    else:
                        blended.append((timestamp, action))
                else:
                    blended.append((timestamp, action))
            self.model_action_seq = blended

        self.action_seq_timestamp = base_time

    def _action_to_pose_command(
        self,
        action: npt.NDArray[np.float32],
    ) -> Optional[npt.NDArray[np.float32]]:
        act = np.asarray(action, dtype=np.float32).reshape(-1)
        expected = self.num_sites * 6

        if act.size == expected:
            return act.reshape(self.num_sites, 6)

        if self.num_sites == 1 and act.size == 6:
            return act.reshape(1, 6)

        if act.size == 6:
            pose = self.pose_command.copy()
            pose[0] = act
            return pose

        return None

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

    def _build_command_matrix(
        self,
        pose_command: npt.NDArray[np.float32],
        x_wrench: Optional[npt.NDArray[np.float32]],
    ) -> npt.NDArray[np.float32]:
        matrix = np.zeros((self.num_sites, COMMAND_LAYOUT.width), dtype=np.float32)
        matrix[:, COMMAND_LAYOUT.position] = pose_command[:, :3]
        matrix[:, COMMAND_LAYOUT.orientation] = pose_command[:, 3:6]
        matrix[:, COMMAND_LAYOUT.kp_pos] = self.pos_stiffness
        matrix[:, COMMAND_LAYOUT.kp_rot] = self.rot_stiffness
        matrix[:, COMMAND_LAYOUT.kd_pos] = self.pos_damping
        matrix[:, COMMAND_LAYOUT.kd_rot] = self.rot_damping
        matrix[:, COMMAND_LAYOUT.force] = self.wrench_command[:, :3]
        matrix[:, COMMAND_LAYOUT.torque] = self.wrench_command[:, 3:6]

        if x_wrench is not None:
            wr = np.asarray(x_wrench, dtype=np.float32)
            if wr.ndim == 2 and wr.shape[1] >= 6:
                rows = min(self.num_sites, wr.shape[0])
                matrix[:rows, COMMAND_LAYOUT.measured_force] = wr[:rows, :3]
                matrix[:rows, COMMAND_LAYOUT.measured_torque] = wr[:rows, 3:6]

        return matrix

    def _step_dp_core(
        self,
        *,
        now: float,
        image_in: Optional[np.ndarray],
        x_obs: Optional[npt.NDArray[np.float32]],
        x_wrench: Optional[npt.NDArray[np.float32]],
        motor_pos: Optional[npt.NDArray[np.float32]],
    ) -> tuple[
        str,
        npt.NDArray[np.float32],
        npt.NDArray[np.float32],
        Optional[npt.NDArray[np.float32]],
    ]:
        image = self._prepare_image(image_in)
        if image is None:
            self.obs_deque.clear()
            self.image_deque.clear()
            self.model_action_seq = []
            self.action_seq_timestamp = 0.0
            self._clear_queue(self.inference_input_queue)
            self._clear_queue(self.inference_output_queue)
            cmd = self._build_command_matrix(self.pose_command, x_wrench)
            return "no_image", self.pose_command.copy(), cmd, None

        obs_vec = self._get_obs_vector(
            x_obs=x_obs,
            x_wrench=x_wrench,
            motor_pos=motor_pos,
        )
        self.obs_deque.append(obs_vec)
        self.image_deque.append(image)

        if len(self.obs_deque) < self.obs_deque.maxlen:
            cmd = self._build_command_matrix(self.pose_command, x_wrench)
            return "warming_up_obs", self.pose_command.copy(), cmd, None

        if len(self.image_deque) < self.image_deque.maxlen:
            cmd = self._build_command_matrix(self.pose_command, x_wrench)
            return "warming_up_image", self.pose_command.copy(), cmd, None

        self._consume_inference_output(now)
        self._submit_inference_request(now)

        if (
            self.model_action_seq
            and now - self.action_seq_timestamp > self.action_max_age_s
        ):
            self.model_action_seq = []

        if not self.model_action_seq:
            cmd = self._build_command_matrix(self.pose_command, x_wrench)
            return "waiting_inference", self.pose_command.copy(), cmd, None

        action = self._select_action_for_time(now)
        pose_cmd = self._action_to_pose_command(action)
        if pose_cmd is None:
            cmd = self._build_command_matrix(self.pose_command, x_wrench)
            return (
                f"invalid_action_dim_{np.asarray(action).size}",
                self.pose_command.copy(),
                cmd,
                np.asarray(action, dtype=np.float32),
            )

        self.pose_command = pose_cmd.astype(np.float32)
        cmd = self._build_command_matrix(self.pose_command, x_wrench)
        return (
            "ok",
            self.pose_command.copy(),
            cmd,
            np.asarray(action, dtype=np.float32).reshape(-1),
        )

    def _close_dp_core(self) -> None:
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
        return cls(args, robot=robot, sim=sim, vis=vis, plot=plot)

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
        qpos_obs = np.asarray(obs.qpos, dtype=np.float32)
        self.controller.wrench_sim.set_qpos(qpos_obs)
        self.controller.wrench_sim.forward()
        image = obs.image
        if image is None:
            image_arr = np.zeros(
                (int(self.args.image_height), int(self.args.image_width), 3),
                dtype=np.uint8,
            )
        else:
            image_arr = _to_hwc_u8(
                np.asarray(image),
                size_hw=(int(self.args.image_height), int(self.args.image_width)),
            )

        x_obs = self._build_x_obs()
        _status, _pose_command, cmd, _raw_action = self._step_dp_core(
            now=float(obs.time),
            image_in=image_arr,
            x_obs=x_obs,
            x_wrench=None,
            motor_pos=np.asarray(obs.motor_pos, dtype=np.float32),
        )
        cmd = np.asarray(cmd, dtype=np.float32)
        if cmd.shape != (self.num_sites, COMMAND_LAYOUT.width):
            raise ValueError(
                f"DP command_matrix shape {cmd.shape} != ({self.num_sites}, {COMMAND_LAYOUT.width})"
            )
        self.command_matrix[:] = cmd

        motor_tor_obs = np.asarray(obs.motor_tor, dtype=np.float32)
        _, state_ref = self.controller.step(
            command_matrix=self.command_matrix,
            motor_torques=motor_tor_obs,
            qpos=qpos_obs,
        )
        if state_ref is not None:
            self.target_motor_pos = np.asarray(state_ref.motor_pos, dtype=np.float32)
        return {}, self.target_motor_pos.copy()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._close_dp_core()
