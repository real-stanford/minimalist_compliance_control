"""Compliance affordance policy built on top of the base compliance controller.

This keeps old policy orchestration style while using local VLM + minimalist
compliance modules.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import joblib
import mujoco
import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation as R

from examples.compliance import CompliancePolicy
from real_world.camera import Camera
from vlm.affordance.affordance_predictor import AffordancePredictor
from vlm.affordance.plan_ee_pose import plan_end_effector_poses


class ComplianceVLMPolicy(CompliancePolicy):
    """Guides compliance references via affordance-predicted trajectories."""

    def __init__(
        self,
        name: str,
        robot: str,
        init_motor_pos: npt.ArrayLike,
        replay: str = "",
        site_names: str = "",
        object: str = "black ink. whiteboard. vase",
        record_video: bool = True,
        image_height: int = 480,
        image_width: int = 640,
        predictor_model: str = "gemini-2.5-pro",
        predictor_provider: str = "gemini",
    ) -> None:
        if robot == "leap":
            gin_config_name = "leap_vlm.gin"
        elif robot == "toddlerbot":
            gin_config_name = "toddlerbot_vlm.gin"
        else:
            raise ValueError(f"Unsupported robot: {robot}")

        super().__init__(
            name=name,
            robot=robot,
            init_motor_pos=init_motor_pos,
            config_name=gin_config_name,
            show_help=False,
        )

        model = self.controller.wrench_sim.model
        self.head_name = str(self.compliance_cfg.head_name).strip()
        self.head_site_id = -1
        self.head_body_id = -1
        if self.head_name:
            self.head_site_id = int(
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, self.head_name)
            )
        if self.head_name:
            self.head_body_id = int(
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, self.head_name)
            )

        self.image_height = int(image_height)
        self.image_width = int(image_width)

        self.target_site_names: List[str] = []
        site_names_str = str(site_names).strip()
        self.site_names_fixed = len(site_names_str) > 0
        if self.site_names_fixed:
            self.target_site_names = [
                s.strip() for s in site_names_str.split(",") if s.strip()
            ]
        if not self.target_site_names:
            if self.robot == "leap":
                self.target_site_names = ["mf_tip"]
            else:
                self.target_site_names = ["left_hand_center"]

        cfg_ref_motor_pos = np.asarray(
            self.compliance_cfg.ref_motor_pos, dtype=np.float32
        ).reshape(-1)
        if cfg_ref_motor_pos.size > 0:
            if cfg_ref_motor_pos.shape[0] != self.default_motor_pos.shape[0]:
                raise ValueError(
                    "ComplianceConfig.ref_motor_pos has wrong size: "
                    f"expected {self.default_motor_pos.shape[0]}, "
                    f"got {cfg_ref_motor_pos.shape[0]}."
                )
            self.ref_motor_pos = cfg_ref_motor_pos.copy()

        self.kp_pos_normal = float(self.compliance_cfg.kp_pos_normal)
        self.kp_pos_tangent = float(self.compliance_cfg.kp_pos_tangent)
        self.kp_rot_normal = float(self.compliance_cfg.kp_rot_normal)
        self.kp_rot_tangent = float(self.compliance_cfg.kp_rot_tangent)
        self.fixed_contact_force = float(self.compliance_cfg.fixed_contact_force)
        self.rest_pose_command = np.asarray(
            self.base_pose_command, dtype=np.float32
        ).copy()
        self._set_rest_pose_from_ref_motor_pos()

        self.status = "waiting"
        self.target_object_label = str(object)
        self.tool = "eraser"
        self.trajectory_plans: Dict[str, Tuple[np.ndarray, ...]] = {}
        self.traj_start_time: Optional[float] = None

        self.wipe_pause_duration = 2.0
        self.wipe_pause_end_time: Optional[float] = None
        self.prediction_requested = False
        self.prediction_counter = 0
        self.replay = str(replay).strip()
        self.fixed_trajectory_active = False
        self.replay_task: Optional[str] = None
        self.replay_contact_points_camera: Dict[str, np.ndarray] = {}
        self.replay_contact_normals_camera: Dict[str, np.ndarray] = {}
        self.replay_unavailable_reported = False
        self.predictor: Optional[AffordancePredictor] = None
        self._load_replay_trajectory()
        self._activate_replay_task_if_available()

        self.predictor = AffordancePredictor(
            model=str(predictor_model),
            provider=str(predictor_provider),
        )

        self.left_camera: Optional[Camera] = None
        self.right_camera: Optional[Camera] = None
        try:
            self.left_camera = Camera("left")
            self.right_camera = Camera("right")
        except Exception as exc:
            self.left_camera = None
            self.right_camera = None
            print(f"[ComplianceVLM] camera stream disabled: {exc}")

        self.teleop.set_command_bindings(
            {"w": "wiping", "d": "drawing"},
            help_labels={"w": "wipe", "d": "draw"},
            enable_default_controls=False,
        )
        self.teleop.print_help(prefix="[ComplianceVLM]")

        self.debug_output_dir = tempfile.TemporaryDirectory(prefix="compliance_vlm_")
        self.prediction_executor = ThreadPoolExecutor(max_workers=1)
        self.prediction_future: Optional[Future] = None

        self.record_video = bool(record_video)
        self.video_logging_active = False
        self.video_writer: Optional[cv2.VideoWriter] = None
        self.video_temp_dir: Optional[tempfile.TemporaryDirectory] = None
        self.video_path: Optional[Path] = None
        self.video_fps: Optional[float] = None
        self.last_left_frame: Optional[np.ndarray] = None
        self.last_right_frame: Optional[np.ndarray] = None
        self.video_frame_timestamps: List[float] = []

        self.set_stiffness(
            pos_stiffness=[400.0, 400.0, 400.0], rot_stiffness=[40.0, 40.0, 40.0]
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
        self._activate_replay_task_if_available()

    def _normalize_task_label(self, value: Any) -> Optional[str]:
        if not isinstance(value, str):
            return None
        normalized = value.strip().lower()
        if normalized in ("wipe", "wiping"):
            return "wipe"
        if normalized in ("draw", "drawing"):
            return "draw"
        return None

    def _set_rest_pose_from_ref_motor_pos(self) -> None:
        """Initialize rest pose from the compliance reference mapping."""
        comp_ref = self.controller.compliance_ref
        if comp_ref is None:
            return

        rest_pose = np.asarray(
            comp_ref.get_x_ref_from_motor_pos(
                np.asarray(self.ref_motor_pos, dtype=np.float32)
            ),
            dtype=np.float32,
        )
        self.rest_pose_command = rest_pose
        self.pose_command[:, :] = self.rest_pose_command
        self.base_pose_command[:, :] = self.rest_pose_command

    def _activate_replay_task_if_available(self) -> None:
        if not self.replay_task:
            return
        if self.replay_task == "wipe":
            self.set_mode(True)
            print("[ComplianceVLM] replay task detected: wipe (auto-selected).")
        elif self.replay_task == "draw":
            self.set_mode(False)
            print("[ComplianceVLM] replay task detected: draw (auto-selected).")

    def _replay_site_names(self) -> List[str]:
        return [str(x) for x in self.replay_contact_points_camera.keys()]

    def _default_site_names_for_mode(self, is_wiping: bool) -> List[str]:
        if self.robot == "leap":
            return ["mf_tip"] if is_wiping else ["rf_tip", "if_tip"]
        return ["left_hand_center"] if is_wiping else ["right_hand_center"]

    def _load_replay_trajectory(self) -> None:
        self.replay_task = None
        self.replay_contact_points_camera = {}
        self.replay_contact_normals_camera = {}
        path = self.get_fixed_trajectory_path()
        if path is None:
            if self.replay:
                print(
                    f"[ComplianceVLM] replay trajectory not found under: {self.replay}"
                )
            return
        try:
            payload = joblib.load(path)
            if not isinstance(payload, dict):
                raise ValueError("replay payload must be a dict.")
            task = self._normalize_task_label(payload.get("task"))
            if task is None:
                raise ValueError("replay payload missing/invalid task.")
            contact_points = payload.get("contact_pos_camera")
            contact_normals = payload.get("contact_normals_camera")
            if isinstance(contact_points, dict) and isinstance(contact_normals, dict):
                self.replay_contact_points_camera = {
                    str(k): np.asarray(v, dtype=np.float32)
                    for k, v in contact_points.items()
                }
                self.replay_contact_normals_camera = {
                    str(k): np.asarray(v, dtype=np.float32)
                    for k, v in contact_normals.items()
                }
            if (
                not self.replay_contact_points_camera
                or not self.replay_contact_normals_camera
            ):
                raise ValueError(
                    "replay payload must contain contact_pos_camera and contact_normals_camera."
                )
            self.replay_task = task
            print(
                f"[ComplianceVLM] loaded replay trajectory for task '{task}' from {path}"
            )
        except Exception as exc:
            print(f"[ComplianceVLM] failed to load replay trajectory: {exc}")
            self.replay_task = None
            self.replay_contact_points_camera = {}
            self.replay_contact_normals_camera = {}

    def can_use_replay_for_current_mode(self) -> bool:
        if self.status not in ("wiping", "drawing"):
            return False
        if self.replay_task is None:
            return False
        current_task = "wipe" if self.status == "wiping" else "draw"
        replay_sites = set(self._replay_site_names())
        if not replay_sites:
            return False
        if current_task != self.replay_task:
            return False
        return any(
            site_name in replay_sites and site_name in self.wrench_site_names
            for site_name in self.target_site_names
        )

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
        elif not self.site_names_fixed:
            self.target_site_names = self._default_site_names_for_mode(is_wiping)
        if self.predictor is not None:
            if is_wiping:
                self.predictor.default_task = f"wipe up the {self.target_object_label} on the whiteboard with an eraser."
            else:
                self.predictor.default_task = f"draw the {self.target_object_label} on the whiteboard using the pen."
        self.trajectory_plans = {}
        self.traj_start_time = None
        self.wrench_command[:, :] = 0.0
        self.prediction_requested = not self.can_use_replay_for_current_mode()
        self.fixed_trajectory_active = False

    def get_fixed_trajectory_path(self) -> Optional[Path]:
        if self.replay:
            replay_path = Path(self.replay).expanduser()
            if replay_path.suffix == ".lz4":
                return replay_path if replay_path.exists() else None
            path = replay_path / "trajectory.lz4"
            return path if path.exists() else None

        return None

    def prepare_fixed_plan(self) -> None:
        if not self.can_use_replay_for_current_mode():
            return
        try:
            valid_sites = [
                site_name
                for site_name in self.target_site_names
                if site_name in self.wrench_site_names
                and site_name in self.replay_contact_points_camera
                and site_name in self.replay_contact_normals_camera
            ]
            if not valid_sites:
                self.fixed_trajectory_active = False
                return

            # Replan from replayed camera-space contacts so execution uses current
            # head pose and current compliance parameters.
            head_pos, head_quat = self.get_head_pose()
            pose_cur = {
                site_name: np.asarray(
                    self.pose_command[self.wrench_site_names.index(site_name)],
                    dtype=np.float32,
                )
                for site_name in valid_sites
            }
            contact_points = {
                site_name: np.asarray(
                    self.replay_contact_points_camera[site_name], dtype=np.float32
                )
                for site_name in valid_sites
            }
            contact_normals = {
                site_name: np.asarray(
                    self.replay_contact_normals_camera[site_name], dtype=np.float32
                )
                for site_name in valid_sites
            }
            self.trajectory_plans = plan_end_effector_poses(
                contact_points_camera=contact_points,
                contact_normals_camera=contact_normals,
                head_position_world=np.asarray(head_pos, dtype=np.float32),
                head_quaternion_world_wxyz=np.asarray(head_quat, dtype=np.float32),
                tangent_pos_stiffness=float(self.kp_pos_tangent),
                normal_pos_stiffness=float(self.kp_pos_normal),
                tangent_rot_stiffness=float(self.kp_rot_tangent),
                normal_rot_stiffness=float(self.kp_rot_normal),
                contact_force=float(self.fixed_contact_force),
                pose_cur=pose_cur,
                output_dir=None,
                traj_dt=float(self.control_dt),
                traj_v_max_contact=0.02,
                traj_v_max_free=0.1,
                tool=self.tool,
                robot_name=self.robot,
                task=self.replay_task,
                mass=float(self.mass),
                inertia_diag=np.asarray(self.inertia_diag, dtype=np.float32),
            )

            self.traj_start_time = None
            self.fixed_trajectory_active = bool(self.trajectory_plans)
            if self.fixed_trajectory_active:
                print(
                    "[ComplianceVLM] replay trajectory prepared for sites: "
                    f"{list(self.trajectory_plans.keys())}"
                )
        except Exception as exc:
            print(f"[ComplianceVLM] failed to load fixed trajectory: {exc}")
            self.fixed_trajectory_active = False

    def get_prediction_output_dir(self, prediction_idx: int) -> Optional[str]:
        if self.debug_output_dir is None:
            return None
        base = Path(self.debug_output_dir.name)
        out_dir = base / f"prediction_{prediction_idx}"
        out_dir.mkdir(parents=True, exist_ok=True)
        return str(out_dir)

    def check_mode_command(self) -> None:
        cmd = self.teleop.poll_command()
        if cmd is None:
            return

        if cmd == "wiping":
            self.set_mode(True)
        elif cmd == "drawing":
            self.set_mode(False)

    def _to_hwc_u8(self, image: np.ndarray) -> np.ndarray:
        arr = np.asarray(image)
        if arr.ndim != 3:
            arr = np.zeros(
                (int(self.image_height), int(self.image_width), 3),
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

    def get_head_pose(self) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        """Return configured head pose as world position and scalar-first quaternion."""
        data = self.controller.wrench_sim.data
        if self.head_body_id >= 0:
            pos = np.asarray(data.xpos[self.head_body_id], dtype=np.float32)
            quat = np.asarray(data.xquat[self.head_body_id], dtype=np.float32)
            return pos.astype(np.float32), np.asarray(quat, dtype=np.float32)
        if self.head_site_id >= 0:
            pos = np.asarray(data.site_xpos[self.head_site_id], dtype=np.float32)
            quat = R.from_matrix(
                np.asarray(data.site_xmat[self.head_site_id], dtype=np.float32).reshape(
                    3, 3
                )
            ).as_quat(scalar_first=True)
            return pos.astype(np.float32), np.asarray(quat, dtype=np.float32)
        raise ValueError(
            f"Head pose source '{self.head_name}' not found as site/body. "
            "Configure ComplianceConfig.head_name in *_vlm.gin."
        )

    def _get_stereo_images(self) -> tuple[np.ndarray, np.ndarray]:
        if self.left_camera is not None:
            try:
                self.last_left_frame = self.left_camera.get_frame()
            except Exception:
                pass
        if self.right_camera is not None:
            try:
                self.last_right_frame = self.right_camera.get_frame()
            except Exception:
                pass

        left = self.last_left_frame
        right = self.last_right_frame if self.last_right_frame is not None else left

        if left is None:
            left = np.zeros(
                (int(self.image_height), int(self.image_width), 3), dtype=np.uint8
            )
            right = left
        elif right is None:
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
        site_names: List[str],
        is_wiping: bool,
        object_label: str,
    ) -> Optional[Dict[str, Tuple[np.ndarray, ...]]]:
        if self.predictor is None:
            return None

        prediction = self.predictor.predict(
            left_image=left_image,
            right_image=right_image,
            robot_name=self.robot,
            site_names=site_names,
            is_wiping=is_wiping,
            output_dir=output_dir,
            object_label=object_label,
        )
        if prediction is None:
            return None

        if not (
            isinstance(prediction, tuple)
            and len(prediction) == 2
            and isinstance(prediction[0], dict)
            and isinstance(prediction[1], dict)
        ):
            print(
                "[ComplianceVLM] prediction failed: expected (contact_points_dict, "
                f"contact_normals_dict), got {type(prediction).__name__}"
            )
            return None
        raw_points, raw_normals = prediction
        valid_sites = [
            s
            for s in site_names
            if s in raw_points and s in raw_normals and s in pose_cur_by_site
        ]
        contact_points = {
            site_name: np.asarray(raw_points[site_name], dtype=np.float32)
            for site_name in valid_sites
        }
        contact_normals = {
            site_name: np.asarray(raw_normals[site_name], dtype=np.float32)
            for site_name in valid_sites
        }

        if not contact_points:
            return None

        return plan_end_effector_poses(
            contact_points_camera=contact_points,
            contact_normals_camera=contact_normals,
            head_position_world=np.asarray(head_pos, dtype=np.float32),
            head_quaternion_world_wxyz=np.asarray(head_quat, dtype=np.float32),
            tangent_pos_stiffness=float(self.kp_pos_tangent),
            normal_pos_stiffness=float(self.kp_pos_normal),
            tangent_rot_stiffness=float(self.kp_rot_tangent),
            normal_rot_stiffness=float(self.kp_rot_normal),
            contact_force=float(self.fixed_contact_force),
            pose_cur=pose_cur_by_site,
            output_dir=output_dir,
            traj_dt=float(self.control_dt),
            traj_v_max_contact=0.02,
            traj_v_max_free=0.1,
            tool=self.tool,
            robot_name=self.robot,
            task="wipe" if is_wiping else "draw",
            mass=float(self.mass),
            inertia_diag=np.asarray(self.inertia_diag, dtype=np.float32),
        )

    def maybe_start_prediction(self, obs: Any, has_fixed_trajectory: bool) -> None:
        if self.status == "waiting":
            return
        if self.prediction_future is not None:
            return
        if self.predictor is None:
            return
        if not self.prediction_requested and self.trajectory_plans:
            return
        if has_fixed_trajectory and not self.trajectory_plans:
            return

        left_image, right_image = self._get_stereo_images()
        head_pos, head_quat = self.get_head_pose()
        output_dir = self.get_prediction_output_dir(self.prediction_counter)
        site_names = list(self.target_site_names)
        is_wiping = self.status == "wiping"
        object_label = str(self.target_object_label)

        pose_cur = {
            site_name: np.asarray(
                self.pose_command[self.wrench_site_names.index(site_name)],
                dtype=np.float32,
            )
            for site_name in site_names
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
            site_names,
            is_wiping,
            object_label,
        )
        self.prediction_counter += 1
        self.prediction_requested = False

    def request_prediction_after_completion(self) -> None:
        if self.status == "wiping":
            self.prediction_requested = True

    def _consume_prediction(self, obs_time: Optional[float] = None) -> None:
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
                now = float(obs_time) if obs_time is not None else 0.0
                self.wipe_pause_end_time = now + self.wipe_pause_duration
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

        # Base CompliancePolicy.step() rebuilds pose_command from base_pose_command
        # each cycle; keep base_pose in sync with replay references.
        self.base_pose_command = np.asarray(self.pose_command, dtype=np.float32).copy()

        if indices and all(idx >= length - 1 for idx, length in indices.values()):
            if self.status == "wiping":
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

    def start_video_logging(self) -> None:
        if self.video_logging_active or not self.record_video:
            return
        self.video_temp_dir = tempfile.TemporaryDirectory(
            prefix="compliance_vlm_video_"
        )
        self.video_path = Path(self.video_temp_dir.name) / "camera.mp4"
        self.video_fps = max(1.0, 1.0 / max(self.control_dt, 1e-3))
        self.video_logging_active = True

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

    def log_camera_frame(
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

    def export_camera_video(self, exp_dir: Path) -> None:
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

        # Let base policy initialize preparation trajectory first.
        if not bool(getattr(self, "is_prepared", False)):
            return np.asarray(action, dtype=np.float32)

        prep_duration = float(getattr(self, "prep_duration", 0.0))
        if float(obs.time) < prep_duration:
            return np.asarray(action, dtype=np.float32)

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
                self._consume_prediction(float(obs.time))
                return np.asarray(action, dtype=np.float32)

        has_fixed_trajectory = self.can_use_replay_for_current_mode()
        if self.replay_task is not None and not has_fixed_trajectory:
            if not self.replay_unavailable_reported:
                current_task = (
                    "wipe"
                    if self.status == "wiping"
                    else ("draw" if self.status == "drawing" else "none")
                )
                print(
                    "[ComplianceVLM] replay not active for current state: "
                    f"current_task={current_task}, "
                    f"replay_task={self.replay_task}, "
                    f"target_sites={self.target_site_names}, "
                    f"replay_sites={self._replay_site_names()}"
                )
                self.replay_unavailable_reported = True
        elif has_fixed_trajectory:
            self.replay_unavailable_reported = False
        if has_fixed_trajectory and not self.trajectory_plans:
            self.prepare_fixed_plan()
        self.maybe_start_prediction(obs, has_fixed_trajectory)
        self._consume_prediction(float(obs.time))
        self._apply_trajectory(float(obs.time))

        left, right = self._get_stereo_images()
        if self.record_video:
            self.start_video_logging()
            self.log_camera_frame(float(obs.time), left, right)

        return np.asarray(action, dtype=np.float32)

    def close(self, exp_folder_path: str = "") -> None:
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
                for item in src.iterdir():
                    dst = exp_dir / item.name
                    if dst.exists():
                        if dst.is_dir():
                            shutil.rmtree(dst)
                        else:
                            dst.unlink()
                    if item.is_dir():
                        shutil.copytree(item, dst)
                    else:
                        shutil.copy2(item, dst)

        if exp_folder_path:
            self.export_camera_video(Path(exp_folder_path))
        self.discard_video_recording()

        if self.debug_output_dir is not None:
            self.debug_output_dir.cleanup()
            self.debug_output_dir = None

        super().close(exp_folder_path)
