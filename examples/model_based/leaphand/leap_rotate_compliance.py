"""LEAP-hand specific compliance policy that moves fingertips to target poses."""

from __future__ import annotations

import os
import sys
import time
import argparse
from typing import Any, Dict, Optional, Tuple

import mujoco
import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
import yaml

SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
EXAMPLE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if EXAMPLE_DIR not in sys.path:
    sys.path.insert(0, EXAMPLE_DIR)

from hybrid_servo.algorithm.ochs import solve_ochs
from hybrid_servo.algorithm.solvehfvc import transform_hfvc_to_global
from hybrid_servo.demo.multi_finger_rotate_anything.ochs_helpers import (
    compute_hfvc_inputs,
    generate_constraint_jacobian,
    get_center_state,
)
from utils.zmq_control import KeyboardControlReceiver
from minimalist_compliance_control.controller import (
    ComplianceController,
    ComplianceInputs,
    ComplianceRefConfig,
    ControllerConfig,
)
from minimalist_compliance_control.reference.compliance_ref import COMMAND_LAYOUT
from minimalist_compliance_control.wrench_estimation import WrenchEstimateConfig


def symmetrize(matrix: npt.ArrayLike) -> npt.NDArray[np.float32]:
    arr = np.asarray(matrix, dtype=np.float32)
    return (0.5 * (arr + np.swapaxes(arr, -1, -2))).astype(np.float32)


def matrix_sqrt(matrix: npt.ArrayLike) -> npt.NDArray[np.float32]:
    sym = symmetrize(matrix)
    eigvals, eigvecs = np.linalg.eigh(sym)
    eigvals_clipped = np.clip(eigvals, 0.0, None)
    sqrt_vals = np.sqrt(eigvals_clipped)[..., None, :]
    scaled_vecs = eigvecs * sqrt_vals
    sqrt_matrix = np.matmul(scaled_vecs, np.swapaxes(eigvecs, -1, -2))
    return symmetrize(sqrt_matrix)


def ensure_matrix(value: float | npt.ArrayLike) -> npt.NDArray[np.float32]:
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim == 0:
        return np.eye(3, dtype=np.float32) * float(arr)
    if arr.ndim == 1:
        if arr.shape[0] != 3:
            raise ValueError("Gain vectors must have length 3.")
        return np.diag(arr.astype(np.float32))
    if arr.ndim >= 2:
        if arr.shape[-2:] != (3, 3):
            raise ValueError("Gain matrices must have trailing shape (3, 3).")
        return arr.astype(np.float32)
    raise ValueError("Unsupported gain array shape.")


def get_damping_matrix(
    stiffness: float | npt.ArrayLike,
    inertia_like: float | npt.ArrayLike,
) -> npt.NDArray[np.float32]:
    stiffness_matrix = ensure_matrix(stiffness)
    inertia_matrix = ensure_matrix(inertia_like)
    mass_sqrt = matrix_sqrt(inertia_matrix)
    stiffness_sqrt = matrix_sqrt(stiffness_matrix)
    damping = 2.0 * np.matmul(mass_sqrt, stiffness_sqrt)
    return symmetrize(damping).astype(np.float32)

PREPARE_POS = [
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    -1.57,
    0.0,
    -1.2,
    0.0,
]

# Fingertip site/geom names used for contact checks.
LEAP_FINGER_TIPS = ("if_tip", "mf_tip", "th_tip")

OBJECT_MASS_MAP = {
    "unknown": {
        "mass": 0.05,
        "init_pos": None,
        "init_quat": None,
        "min_normal_force_rotation": 6.0,
        "min_normal_force_translation": 9.0,
        "geom_size": np.array([0.04], dtype=np.float32),  # Default radius for sphere
    },
    "sphere": {
        "mass": 0.06,
        "init_pos": np.array([-0.12, -0.075, 0.16], dtype=np.float32),
        "init_quat": np.array([0.70710677, 0.70710677, 0.0, 0.0], dtype=np.float32),
        "min_normal_force_rotation": 10.0,
        "min_normal_force_translation": 10.0,
        "geom_size": np.array([0.0343], dtype=np.float32),  # Radius
    },
    "box": {
        "mass": 0.1,
        "init_pos": np.array([-0.115, -0.075, 0.164], dtype=np.float32),
        "init_quat": np.array([0.70710677, 0.70710677, 0.0, 0.0], dtype=np.float32),
        "min_normal_force_rotation": 5,
        "min_normal_force_translation": 5.0,
        "geom_size": np.array(
            [0.03, 0.03, 0.03], dtype=np.float32
        ),  # [half_x, half_y, half_z]
    },
    "cylinder_short": {
        "mass": 0.1,
        "init_pos": np.array([-0.13, -0.08, 0.145], dtype=np.float32),
        "init_quat": np.array([0.70710677, 0.70710677, 0.0, 0.0], dtype=np.float32),
        "min_normal_force_rotation": 8.0,
        "min_normal_force_translation": 5.0,
        "geom_size": np.array([0.04, 0.12], dtype=np.float32),  # [radius, half_height]
    },
    "pen": {
        "mass": 0.05,
        "init_pos": np.array([-0.13, -0.08, 0.14], dtype=np.float32),
        "init_quat": np.array([0.70710677, 0.70710677, 0.0, 0.0], dtype=np.float32),
        "min_normal_force_rotation": 6.0,
        "min_normal_force_translation": 9.0,
        "geom_size": np.array(
            [0.015, 0.08], dtype=np.float32
        ),  # [radius, half_height] for cylinder-like pen
    },
}

INIT_POSE_DATA = {
    "if_tip": {
        "pos": np.array([0.052, -0.1, 0.247], dtype=np.float32),
        "ori": np.array([-3.14, 0, 0], dtype=np.float32),
    },
    "mf_tip": {
        "pos": np.array([0.052, -0.055, 0.247], dtype=np.float32),
        "ori": np.array([-3.14, 0, 0], dtype=np.float32),
    },
    "rf_tip": {
        "pos": np.array([0.052, -0.01, 0.247], dtype=np.float32),
        "ori": np.array([-3.14, 0, 0], dtype=np.float32),
    },
    "th_tip": {
        "pos": np.array([-0.228, -0.094, 0.149], dtype=np.float32),
        "ori": np.array([-0.07, 2.42, 0.02], dtype=np.float32),
    },
}
TARGET_POSE_DATA: Dict[str, Dict[str, np.ndarray]] = {
    "if_tip": {
        "pos": np.array([-0.101, -0.099, 0.152], dtype=np.float32),
        "ori": np.array([-1.40, -0.05, 2.81], dtype=np.float32),
    },
    "mf_tip": {
        "pos": np.array([-0.101, -0.056, 0.152], dtype=np.float32),
        "ori": np.array([-1.40, 0.0, 2.81], dtype=np.float32),
    },
    "rf_tip": {
        "pos": np.array([0.042, -0.01, 0.247], dtype=np.float32),
        "ori": np.array([-3.14, 0.0, 0.0], dtype=np.float32),
    },
    "th_tip": {
        "pos": np.array([-0.145, -0.085, 0.148], dtype=np.float32),
        "ori": np.array([0.04, 1.02, 0.03], dtype=np.float32),
    },
}

###TARGET_POSE_DATA for pen rotating
# TARGET_POSE_DATA: Dict[str, Dict[str, np.ndarray]] = {
#     "if_tip": {
#         "pos": np.array([-0.121, -0.101, 0.14], dtype=np.float32),
#         "ori": np.array([1.86, -0.0, -2.53], dtype=np.float32),
#     },
#     "mf_tip": {
#         "pos": np.array([-0.121, -0.056, 0.14], dtype=np.float32),
#         "ori": np.array([1.86, 0.0, -2.53], dtype=np.float32),
#     },
#     "rf_tip": {
#         "pos": np.array([0.042, -0.01, 0.247], dtype=np.float32),
#         "ori": np.array([-3.14, -0.0, 0.0], dtype=np.float32),
#     },
#     "th_tip": {
#         "pos": np.array([-0.141, -0.077, 0.139], dtype=np.float32),
#         "ori": np.array([0.08, 1.23, 0.09], dtype=np.float32),
#     },
# }

OBJECT_INIT_POS_MAP = {
    "sphere": np.array([-0.125, -0.08, 0.145], dtype=np.float32),
    "box": np.array([-0.125, -0.08, 0.15], dtype=np.float32),
    "cylinder_short": np.array([-0.125, -0.075, 0.16], dtype=np.float32),
}

OBJECT_TYPE = "cylinder_short"
# "box"
# "sphere"
# "cylinder_short"


class LeapRotateCompliancePolicy:
    """Toggle fingertip pose commands between default and a fixed grasp pose."""

    def __init__(
        self,
        wrench_sim: Any,
        wrench_site_names: Tuple[str, ...] = LEAP_FINGER_TIPS,
        control_dt: float = 0.02,
        prep_duration: float = 0.0,
        auto_switch_target_enabled: bool = True,
        control_port: int = 5592,
    ):
        self.wrench_sim = wrench_sim
        self.wrench_site_names = list(wrench_site_names)
        self.num_sites = len(self.wrench_site_names)
        self.control_dt = float(control_dt)
        self.prep_duration = float(prep_duration)
        self.wrenches_by_site: Dict[str, np.ndarray] = {}
        self.wrench_command = np.zeros((self.num_sites, 6), dtype=np.float32)
        self.pos_stiffness = np.zeros((self.num_sites, 9), dtype=np.float32)
        self.rot_stiffness = np.zeros((self.num_sites, 9), dtype=np.float32)
        self.pos_damping = np.zeros((self.num_sites, 9), dtype=np.float32)
        self.rot_damping = np.zeros((self.num_sites, 9), dtype=np.float32)

        self.object_type = OBJECT_TYPE
        self.object_type_detected = False
        self.object_mass = 0.05
        self.object_geom_size = np.array([0.04], dtype=np.float32)  # Default size

        self.use_compliance = True
        self.log_ik = True
        self.pd_updated = False
        self.desired_kp = 1500  # 450
        self.desired_kd = 0

        self.contact_force = 0.0
        self.normal_pos_stiffness = 10.0
        self.tangent_pos_stiffness = 100.0
        self.normal_rot_stiffness = 10.0
        self.tangent_rot_stiffness = 20.0

        self.ref_motor_pos = np.array(PREPARE_POS, dtype=np.float32)
        self.initial_pose_command = self.build_pose_command(INIT_POSE_DATA)
        self.target_pose_command = self.build_pose_command(TARGET_POSE_DATA)
        self.pose_command = self.initial_pose_command.copy()
        self.integrated_angle_thumb = np.zeros(3, dtype=np.float32)
        self.pose_interp_pos_speed = 0.1
        self.pose_interp_rot_speed = 1.0
        self.pose_interp_min_duration = 0.2
        self.pose_interp_max_duration = 2.0
        mass_matrix = ensure_matrix(1.0)
        inertia_matrix = ensure_matrix([1.0, 1.0, 1.0])

        open_pos_stiff = ensure_matrix([400.0, 400.0, 400.0])
        open_rot_stiff = ensure_matrix([20.0, 20.0, 20.0])
        open_pos_stiff_arr = np.broadcast_to(
            open_pos_stiff, (self.num_sites, 3, 3)
        ).astype(np.float32)
        open_rot_stiff_arr = np.broadcast_to(
            open_rot_stiff, (self.num_sites, 3, 3)
        ).astype(np.float32)
        open_pos_damp = np.stack(
            [
                get_damping_matrix(open_pos_stiff, mass_matrix)
                for _ in range(self.num_sites)
            ],
            axis=0,
        ).astype(np.float32)
        open_rot_damp = np.stack(
            [
                get_damping_matrix(open_rot_stiff, inertia_matrix)
                for _ in range(self.num_sites)
            ],
            axis=0,
        ).astype(np.float32)
        open_wrench = np.zeros((self.num_sites, 6), dtype=np.float32)
        self.open_gains = {
            "pos_stiff": open_pos_stiff_arr,
            "rot_stiff": open_rot_stiff_arr,
            "pos_damp": open_pos_damp,
            "rot_damp": open_rot_damp,
            "wrench": open_wrench,
        }
        self.close_gains = self.compute_force_and_stiffness(self.target_pose_command)

        self.forward_traj = self.build_command_trajectory(
            self.initial_pose_command,
            self.target_pose_command,
            self.open_gains,
            self.close_gains,
        )
        self.backward_traj = self.build_command_trajectory(
            self.target_pose_command,
            self.initial_pose_command,
            self.close_gains,
            self.open_gains,
        )

        self.active_traj: Optional[Dict[str, np.ndarray]] = None
        self.traj_start_time = 0.0

        # Initialize stiffness/wrench targets from first trajectory sample.
        self.apply_traj_sample(self.forward_traj, 0)

        self.phase = "close"
        self.traj_set = False
        self.object_body_name: str = "manip_object"
        self.object_qpos_adr: Optional[int] = None
        self.object_qvel_adr: Optional[int] = None
        self.close_stage: str = "to_init"
        self.jacobian_constraint = generate_constraint_jacobian()
        self.target_rotation_angvel = np.array([0.0, 0.0, 0.0])
        self.target_rotation_linvel = np.array([0.03, 0.0, 0.0])
        self.last_angvel_flip_time: Optional[float] = None
        self.pos_kp = 300  # High stiffness for anisotropic fingers.
        self.force_kp = 200  # Low stiffness for anisotropic fingers.
        self.rot_kp = 20
        self.baseline_tip_rot: Dict[str, R] = {}
        self.interval = 1.5
        # Store last contact position for each fingertip (used during close phase)
        self.last_contact_pos: Dict[str, Optional[np.ndarray]] = {
            tip: None for tip in self.wrench_site_names
        }

        # Control mode: "rotation" or "translation"
        self.control_mode = "translation"  # Default mode
        self.rotation_angvel_magnitude = 0.5  # ±0.5 rad/s for rotation mode
        self.translation_linvel_magnitude = 0.03  # ±0.03 m/s for translation mode

        # HFVC parameters for different modes
        default_force = OBJECT_MASS_MAP.get(
            self.object_type, OBJECT_MASS_MAP["unknown"]
        )
        self.min_normal_force_rotation = float(
            default_force["min_normal_force_rotation"]
        )
        self.min_normal_force_translation = float(
            default_force["min_normal_force_translation"]
        )

        # Centripetal force: additional force toward object center for each end effector
        self.centripetal_force_magnitude = (
            1.0  # N, force magnitude toward object center
        )

        # Contact detection via external wrench
        self.contact_force_threshold = 0.1

        # Threshold mechanism (relative to initial position)
        self.threshold_angle = np.pi / 4  # 45 degrees for rotation mode
        self.threshold_position = 0.05  # 5cm for translation mode
        self.threshold_angle_reverse = -self.threshold_angle
        self.threshold_position_reverse = -0.02
        self.auto_switch_target_enabled = auto_switch_target_enabled
        self.auto_switch_counter = 0  # 0=reverse, 1=mode_switch, 2=reverse, ...
        self.limit_reached_flag = False
        self.integrated_angle = 0.0  # Integrated angle from angvel
        self.integrated_position = 0.0  # Integrated position from linvel
        self.last_integration_time: Optional[float] = None

        # Mode switching state
        self.mode_switch_pending = False
        self.target_mode: Optional[str] = None
        self.return_to_zero_tolerance = 0.003

        # ZMQ receiver for keyboard control.
        self.control_port = int(control_port)
        self.control_receiver: Optional[KeyboardControlReceiver] = None
        try:
            self.control_receiver = KeyboardControlReceiver(port=self.control_port)
            if self.control_receiver is not None and self.control_receiver.enabled:
                print(
                    f"[LeapRotateCompliance] Control receiver listening on port {self.control_port} (c=reverse, r=switch mode)."
                )
        except Exception as exc:
            self.control_receiver = None
            print(f"[LeapRotateCompliance] Warning: control receiver disabled: {exc}")

    def build_pose_command(
        self, pose_data: Dict[str, Dict[str, np.ndarray]]
    ) -> npt.NDArray[np.float32]:
        pose_cmd = np.zeros((self.num_sites, 6), dtype=np.float32)
        for idx, site in enumerate(self.wrench_site_names):
            site_data = pose_data.get(site)
            if site_data is None:
                raise ValueError(f"No pose data provided for site '{site}'.")
            pose_cmd[idx, :3] = site_data["pos"]
            pose_cmd[idx, 3:6] = site_data["ori"]
        return pose_cmd

    def compute_force_and_stiffness(
        self, pose: npt.NDArray[np.float32]
    ) -> Dict[str, np.ndarray]:
        rot_mats = R.from_rotvec(pose[:, 3:6]).as_matrix().astype(np.float32)
        normals = []
        for idx, site in enumerate(self.wrench_site_names):
            local_normal = (
                np.array([1.0, 0.0, 0.0], dtype=np.float32)
                if site == "th_tip"
                else np.array([0.0, 0.0, 1.0], dtype=np.float32)
            )
            normal = rot_mats[idx] @ local_normal
            normals.append(normal / (np.linalg.norm(normal) + 1e-9))

        normals_arr = np.asarray(normals, dtype=np.float32)
        wrench = np.zeros((self.num_sites, 6), dtype=np.float32)
        wrench[:, :3] = normals_arr * self.contact_force

        eye = np.eye(3, dtype=np.float32)
        pos_stiff = []
        rot_stiff = []
        for normal in normals_arr:
            outer = np.outer(normal, normal)
            pos_stiff.append(
                eye * self.tangent_pos_stiffness
                + (self.normal_pos_stiffness - self.tangent_pos_stiffness) * outer
            )
            rot_stiff.append(
                eye * self.tangent_rot_stiffness
                + (self.normal_rot_stiffness - self.tangent_rot_stiffness) * outer
            )
        pos_stiff_arr = np.stack(pos_stiff, axis=0).astype(np.float32)
        rot_stiff_arr = np.stack(rot_stiff, axis=0).astype(np.float32)

        mass_matrix = ensure_matrix(1.0)
        inertia_matrix = ensure_matrix([1.0, 1.0, 1.0])
        pos_damp = np.stack(
            [get_damping_matrix(mat, mass_matrix) for mat in pos_stiff_arr], axis=0
        ).astype(np.float32)
        rot_damp = np.stack(
            [get_damping_matrix(mat, inertia_matrix) for mat in rot_stiff_arr], axis=0
        ).astype(np.float32)

        return {
            "pos_stiff": pos_stiff_arr,
            "rot_stiff": rot_stiff_arr,
            "pos_damp": pos_damp,
            "rot_damp": rot_damp,
            "wrench": wrench,
        }

    def build_command_trajectory(
        self,
        pose_start: npt.NDArray[np.float32],
        pose_target: npt.NDArray[np.float32],
        gains_start: Dict[str, np.ndarray],
        gains_target: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        pose_start = np.asarray(pose_start, dtype=np.float32)
        pose_target = np.asarray(pose_target, dtype=np.float32)

        pos_delta_max = float(
            np.linalg.norm(pose_target[:, :3] - pose_start[:, :3], axis=1).max()
        )
        rot_delta_max = float(
            np.linalg.norm(
                (
                    R.from_rotvec(pose_target[:, 3:6])
                    * R.from_rotvec(pose_start[:, 3:6]).inv()
                ).as_rotvec(),
                axis=1,
            ).max()
        )
        duration = max(
            pos_delta_max / max(self.pose_interp_pos_speed, 1e-6),
            rot_delta_max / max(self.pose_interp_rot_speed, 1e-6),
        )
        duration = float(
            np.clip(
                duration,
                self.pose_interp_min_duration,
                self.pose_interp_max_duration,
            )
        )

        t_samples = np.arange(
            0.0, duration + self.control_dt, self.control_dt, dtype=np.float32
        )
        if t_samples.size == 0 or t_samples[-1] < duration:
            t_samples = np.append(t_samples, np.float32(duration))
        u = np.clip(t_samples / max(duration, 1e-6), 0.0, 1.0)
        weights = u

        pos_interp = (
            pose_start[None, :, :3]
            + (pose_target[None, :, :3] - pose_start[None, :, :3])
            * weights[:, None, None]
        ).astype(np.float32)

        ori_interp = np.zeros((weights.size, self.num_sites, 3), dtype=np.float32)
        for idx in range(self.num_sites):
            rot_start = pose_start[idx, 3:6]
            rot_target = pose_target[idx, 3:6]
            if np.allclose(rot_start, rot_target, atol=1e-6):
                ori_interp[:, idx] = rot_target
                continue
            key_rots = R.from_rotvec(np.stack([rot_start, rot_target], axis=0))
            slerp = Slerp([0.0, 1.0], key_rots)
            interp_rots = slerp(weights)
            ori_interp[:, idx] = interp_rots.as_rotvec().astype(np.float32)

        def blend(a: np.ndarray, b: np.ndarray) -> np.ndarray:
            w = weights.reshape((-1,) + (1,) * a.ndim)
            return (a[None] + (b - a)[None] * w).astype(np.float32)

        traj = {
            "time": t_samples,
            "pos": pos_interp,
            "ori": ori_interp,
            "pos_stiff": blend(gains_start["pos_stiff"], gains_target["pos_stiff"]),
            "rot_stiff": blend(gains_start["rot_stiff"], gains_target["rot_stiff"]),
            "pos_damp": blend(gains_start["pos_damp"], gains_target["pos_damp"]),
            "rot_damp": blend(gains_start["rot_damp"], gains_target["rot_damp"]),
            "wrench": blend(gains_start["wrench"], gains_target["wrench"]),
        }
        return traj

    def apply_traj_sample(self, traj: Dict[str, np.ndarray], idx: int) -> None:
        idx = int(np.clip(idx, 0, traj["time"].shape[0] - 1))
        self.pose_command[:, :3] = traj["pos"][idx]
        self.pose_command[:, 3:6] = traj["ori"][idx]
        self.wrench_command = traj["wrench"][idx].copy()

        # Directly set stiffness arrays (already in correct shape: num_sites x 9)
        self.pos_stiffness = np.asarray(
            traj["pos_stiff"][idx], dtype=np.float32
        ).reshape(self.num_sites, 9)
        self.rot_stiffness = np.asarray(
            traj["rot_stiff"][idx], dtype=np.float32
        ).reshape(self.num_sites, 9)
        self.pos_damping = np.asarray(traj["pos_damp"][idx], dtype=np.float32).reshape(
            self.num_sites, 9
        )
        self.rot_damping = np.asarray(traj["rot_damp"][idx], dtype=np.float32).reshape(
            self.num_sites, 9
        )

    def start_command_trajectory(
        self, traj: Dict[str, np.ndarray], time_curr: Optional[float]
    ) -> None:
        self.active_traj = traj
        self.traj_start_time = float(time_curr if time_curr is not None else 0.0)
        self.apply_traj_sample(traj, 0)

    def advance_command_trajectory(self, time_curr: float) -> None:
        if self.active_traj is None:
            return
        times = self.active_traj["time"]
        elapsed = time_curr - self.traj_start_time
        idx = int(np.searchsorted(times, elapsed, side="right") - 1)
        self.apply_traj_sample(self.active_traj, idx)
        if elapsed >= float(times[-1]):
            self.active_traj = None

    def check_control_command(self) -> str | None:
        """Check for keyboard commands via ZMQ receiver."""
        if self.control_receiver is None:
            return None

        msg = self.control_receiver.poll_command()
        if msg is None or msg.command is None:
            return None

        cmd = str(msg.command).strip().lower()
        return cmd if cmd in ("c", "r") else None

    def update_goal(self, time_curr: float) -> None:
        """Update target velocities based on keyboard commands and threshold.

        Commands:
        - 'c': Reverse current target (flip sign of angvel or linvel)
        - 'r': Switch between rotation mode and translation mode

        Threshold logic:
        - Integrates angvel/linvel to track relative position from initial state
        - When reaching threshold, sets velocity to 0
        - Pressing 'c' reverses direction to move toward reverse threshold
        """
        # Initialize integration timer on first call during rotate phase
        if self.last_integration_time is None:
            self.last_integration_time = time_curr

        # Calculate dt for integration
        dt = time_curr - self.last_integration_time
        self.last_integration_time = time_curr

        # Integrate velocities to track relative position
        if self.control_mode == "rotation":
            # Integrate angular velocity (z-axis component)
            angvel_z = self.target_rotation_angvel[2]
            self.integrated_angle += angvel_z * dt
            current_metric = self.integrated_angle
            threshold = self.threshold_angle
            reverse_threshold = self.threshold_angle_reverse
        else:
            # Integrate linear velocity (x-axis component)
            linvel_x = self.target_rotation_linvel[0]
            self.integrated_position += linvel_x * dt
            current_metric = self.integrated_position
            threshold = self.threshold_position
            reverse_threshold = self.threshold_position_reverse

        # Check for keyboard commands
        cmd = self.check_control_command()

        if cmd == "c":
            self.apply_reverse_command()

        elif cmd == "r":
            self.request_mode_switch()

        if self.mode_switch_pending:
            at_zero = False
            if self.control_mode == "rotation":
                at_zero = abs(self.integrated_angle) < self.return_to_zero_tolerance
                if not at_zero:
                    direction = -np.sign(self.integrated_angle)
                    if direction == 0:
                        direction = 1.0
                    self.target_rotation_angvel = np.array(
                        [0.0, 0.0, direction * self.rotation_angvel_magnitude]
                    )
            else:
                at_zero = abs(self.integrated_position) < self.return_to_zero_tolerance
                if not at_zero:
                    direction = -np.sign(self.integrated_position)
                    if direction == 0:
                        direction = 1.0
                    self.target_rotation_linvel = np.array(
                        [direction * self.translation_linvel_magnitude, 0.0, 0.0]
                    )

            if at_zero:
                self.mode_switch_pending = False
                # Reset pose_command to target pose to avoid drift
                self.pose_command = self.target_pose_command.copy()
                print(
                    "[LeapRotateCompliance] Reset pose_command to target_pose_command to prevent drift"
                )

                if self.target_mode == "translation":
                    self.control_mode = "translation"
                    self.integrated_position = 0.0
                    self.target_rotation_angvel = np.array([0.0, 0.0, 0.0])
                    self.target_rotation_linvel = np.array(
                        [self.translation_linvel_magnitude, 0.0, 0.0]
                    )
                    print(
                        f"[LeapRotateCompliance] Switched to TRANSLATION mode: linvel = {self.target_rotation_linvel}"
                    )
                else:
                    self.control_mode = "rotation"
                    self.integrated_angle = 0.0
                    self.target_rotation_linvel = np.array([0.0, 0.0, 0.0])
                    self.target_rotation_angvel = np.array(
                        [0.0, 0.0, self.rotation_angvel_magnitude]
                    )
                    print(
                        f"[LeapRotateCompliance] Switched to ROTATION mode: angvel = {self.target_rotation_angvel}"
                    )
                self.target_mode = None
                # Skip threshold check this frame to ensure velocity is applied
                return
            return

        if self.control_mode == "rotation":
            active_vel = self.target_rotation_angvel[2]
        else:
            active_vel = self.target_rotation_linvel[0]
        active_threshold = abs(reverse_threshold) if active_vel < 0.0 else threshold
        moving_outward = False
        just_reached_limit = False
        if abs(current_metric) >= abs(active_threshold):
            # Check if moving outward (away from origin)
            if self.control_mode == "rotation":
                moving_outward = np.sign(self.target_rotation_angvel[2]) == np.sign(
                    current_metric
                )
            else:
                moving_outward = np.sign(self.target_rotation_linvel[0]) == np.sign(
                    current_metric
                )

            if moving_outward:
                just_reached_limit = not self.limit_reached_flag
                if just_reached_limit:
                    self.limit_reached_flag = True
                # Stop at threshold
                if self.control_mode == "rotation":
                    self.target_rotation_angvel = np.array([0.0, 0.0, 0.0])
                    self.integrated_angle = float(
                        np.clip(
                            self.integrated_angle, -active_threshold, active_threshold
                        )
                    )
                    print(
                        f"[LeapRotateCompliance] Reached threshold: angle = {self.integrated_angle:.3f}, stopped"
                    )
                else:
                    self.target_rotation_linvel = np.array([0.0, 0.0, 0.0])
                    self.integrated_position = float(
                        np.clip(
                            self.integrated_position,
                            -active_threshold,
                            active_threshold,
                        )
                    )
                    print(
                        f"[LeapRotateCompliance] Reached threshold: position = {self.integrated_position:.3f}, stopped"
                    )
                self.auto_switch_target(just_reached_limit)

        if (not moving_outward) or (abs(current_metric) < abs(active_threshold) * 0.98):
            self.limit_reached_flag = False

    def apply_reverse_command(self) -> None:
        """Reverse direction as if pressing 'c'."""
        if self.control_mode == "rotation":
            if np.linalg.norm(self.target_rotation_angvel) < 1e-6:
                direction = -np.sign(self.integrated_angle)
                if direction == 0:
                    direction = 1.0
            else:
                direction = -np.sign(self.target_rotation_angvel[2])
            self.target_rotation_angvel = np.array(
                [0.0, 0.0, direction * self.rotation_angvel_magnitude]
            )
            print(
                f"[LeapRotateCompliance] Reversed rotation: angvel = {self.target_rotation_angvel}, integrated_angle = {self.integrated_angle:.3f}"
            )
        else:
            if np.linalg.norm(self.target_rotation_linvel) < 1e-6:
                direction = -np.sign(self.integrated_position)
                if direction == 0:
                    direction = 1.0
            else:
                direction = -np.sign(self.target_rotation_linvel[0])
            self.target_rotation_linvel = np.array(
                [direction * self.translation_linvel_magnitude, 0.0, 0.0]
            )
            print(
                f"[LeapRotateCompliance] Reversed translation: linvel = {self.target_rotation_linvel}, integrated_position = {self.integrated_position:.3f}"
            )

    def request_mode_switch(self) -> None:
        """Request a mode switch as if pressing 'r'."""
        if self.mode_switch_pending:
            return
        self.mode_switch_pending = True
        if self.control_mode == "rotation":
            self.target_mode = "translation"
            print(
                f"[LeapRotateCompliance] Mode switch requested: rotation -> translation, returning to zero first (angle={self.integrated_angle:.3f})"
            )
        else:
            self.target_mode = "rotation"
            print(
                f"[LeapRotateCompliance] Mode switch requested: translation -> rotation, returning to zero first (position={self.integrated_position:.3f})"
            )

    def auto_switch_target(self, just_reached_limit: bool) -> None:
        """Auto-switch target when reaching limit: reverse -> mode_switch -> reverse -> ..."""
        if not self.auto_switch_target_enabled:
            return
        if not just_reached_limit:
            return

        action_type = self.auto_switch_counter % 2  # 0=reverse, 1=mode_switch
        if action_type == 0:
            self.apply_reverse_command()
        else:
            self.request_mode_switch()
        self.auto_switch_counter += 1

    def _ensure_object_detected(self) -> None:
        if self.object_type_detected:
            return
        self.object_type = OBJECT_TYPE
        object_info = OBJECT_MASS_MAP.get(self.object_type, OBJECT_MASS_MAP["unknown"])
        self.object_mass = float(object_info.get("mass", 0.05))
        self.object_init_pos = object_info.get("init_pos")
        self.object_init_quat = object_info.get("init_quat")
        self.object_geom_size = object_info.get("geom_size")
        self.min_normal_force_rotation = float(
            object_info.get(
                "min_normal_force_rotation",
                self.min_normal_force_rotation,
            )
        )
        self.min_normal_force_translation = float(
            object_info.get(
                "min_normal_force_translation",
                self.min_normal_force_translation,
            )
        )
        if self.object_init_pos is not None:
            self.object_init_pos = self.object_init_pos.copy()
        if self.object_init_quat is not None:
            self.object_init_quat = self.object_init_quat.copy()
        self.object_type_detected = True
        print(
            f"[LeapRotateCompliance] Detected object type: {self.object_type}, mass: {self.object_mass}kg"
        )

    def forward_object_to_init(self, sim_name: str = "sim") -> None:
        """Immediately place object at policy init pose and forward once."""
        self._ensure_object_detected()
        self.capture_object_init()
        self.fix_object(self.wrench_sim, sim_name=sim_name)
        mujoco.mj_forward(self.wrench_sim.model, self.wrench_sim.data)

    def step(
        self,
        time_curr: float,
        wrenches_by_site: Optional[Dict[str, np.ndarray]] = None,
        system_state: Optional[Dict[str, np.ndarray]] = None,
        *,
        sim_name: str = "sim",
        is_real_world: bool = False,
    ) -> Dict[str, np.ndarray | str]:
        if wrenches_by_site is not None:
            self.wrenches_by_site = {
                key: np.asarray(val, dtype=np.float32) for key, val in wrenches_by_site.items()
            }

        self._ensure_object_detected()

        if time_curr < self.prep_duration:
            self.capture_object_init()
            if not is_real_world:
                self.fix_object(self.wrench_sim, sim_name=sim_name)
            return self.get_outputs()

        if self.phase == "close":
            self.capture_object_init()
            if not is_real_world:
                self.fix_object(self.wrench_sim, sim_name=sim_name)
            if self.close_stage == "to_init":
                if self.active_traj is None:
                    traj = self.build_command_trajectory(
                        self.pose_command.copy(),
                        self.initial_pose_command,
                        self.open_gains,
                        self.open_gains,
                    )
                    self.start_command_trajectory(traj, time_curr)
            elif self.close_stage == "to_target":
                if self.active_traj is None and not self.traj_set:
                    self.start_command_trajectory(self.forward_traj, time_curr)
                    self.traj_set = True
                elif self.active_traj is None:
                    self.check_switch_phase()
            self.advance_command_trajectory(time_curr)
        elif self.phase == "rotate":
            # Update goal with keyboard commands and threshold checking
            self.update_goal(time_curr)

            # Handle rotation action
            self.handle_rotate_action(system_state)
            self.check_switch_phase()

        if (
            self.phase == "close"
            and self.close_stage == "to_init"
            and self.active_traj is None
        ):
            self.close_stage = "to_target"
            self.traj_set = False
        return self.get_outputs()

    def get_outputs(self) -> Dict[str, np.ndarray | str]:
        return {
            "phase": self.phase,
            "control_mode": self.control_mode,
            "pose_command": self.pose_command.copy(),
            "wrench_command": self.wrench_command.copy(),
            "pos_stiffness": self.pos_stiffness.copy(),
            "rot_stiffness": self.rot_stiffness.copy(),
            "pos_damping": self.pos_damping.copy(),
            "rot_damping": self.rot_damping.copy(),
        }

    def assign_stiffness(
        self,
        left_vel: np.ndarray,
        right_vel: np.ndarray,
    ) -> None:
        """Set anisotropic stiffness for index/middle; others very stiff."""

        def build_diag(vel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            dir_vec = np.asarray(vel, dtype=np.float32).reshape(-1)
            norm = np.linalg.norm(dir_vec)
            pos_high = float(self.pos_kp)
            pos_low = float(self.force_kp)
            eye = np.eye(3, dtype=np.float32)
            if norm < 1e-6:
                diag = np.full(3, pos_low, dtype=np.float32)
            else:
                dir_unit = dir_vec / norm
                proj = np.outer(dir_unit, dir_unit)
                mat = eye * pos_low + (pos_high - pos_low) * proj
                diag = np.diag(mat)
            damp = 2.0 * np.sqrt(diag)
            return diag, damp

        # Map finger tips to velocities
        vel_map = {
            "if_tip": left_vel,
            "mf_tip": right_vel,
        }

        # High stiffness for non-anisotropic fingers
        high_stiff_diag = np.full(3, float(self.pos_kp), dtype=np.float32)
        high_damp_diag = 2.0 * np.sqrt(high_stiff_diag)

        # Rotation stiffness/damping (same for all fingers)
        rot_stiff_diag = np.full(3, float(self.rot_kp), dtype=np.float32)
        rot_damp_diag = 2.0 * np.sqrt(rot_stiff_diag)

        # Set stiffness for each finger
        for idx, tip in enumerate(self.wrench_site_names):
            if tip in vel_map:
                pos_diag, pos_damp_diag = build_diag(vel_map[tip])
            else:
                pos_diag = high_stiff_diag
                pos_damp_diag = high_damp_diag

            self.pos_stiffness[idx] = np.diag(pos_diag).flatten()
            self.pos_damping[idx] = np.diag(pos_damp_diag).flatten()
            self.rot_stiffness[idx] = np.diag(rot_stiff_diag).flatten()
            self.rot_damping[idx] = np.diag(rot_damp_diag).flatten()

    def set_phase(self, phase: str) -> None:
        """Update phase and reset trajectory flag whenever phase changes."""
        if self.phase != phase:
            self.phase = phase
            self.traj_set = False

    def check_switch_phase(self) -> None:
        """Switch from close to rotate once all fingertips have sufficient contact force."""
        if self.phase == "close":
            has_contact = self.check_all_fingertips_contact()
            if has_contact:
                self.freeze_pose_to_current()
                self.capture_baseline_tip_rot()
                self.set_phase("rotate")
        else:
            return

    def check_all_fingertips_contact(self) -> bool:
        """Check if index or middle fingertip has contact based on external wrench."""
        if not hasattr(self, "wrenches_by_site") or not self.wrenches_by_site:
            return False

        for tip in ("if_tip", "mf_tip"):
            wrench = self.wrenches_by_site.get(tip)
            if wrench is None:
                continue
            force_magnitude = np.linalg.norm(wrench[:3])
            if force_magnitude >= self.contact_force_threshold:
                return True
        return False

    def capture_baseline_tip_rot(self) -> None:
        """Cache current fingertip orientations as baseline for relative quats."""
        self.baseline_tip_rot.clear()
        model = self.wrench_sim.model
        data = self.wrench_sim.data
        for tip in ("th_tip", "if_tip", "mf_tip"):
            sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, tip)
            if sid < 0:
                continue
            mat = data.site_xmat[sid].reshape(3, 3)
            self.baseline_tip_rot[tip] = R.from_matrix(mat)

    def freeze_pose_to_current(self) -> None:
        """Set pose_command to current site poses to avoid jumps when switching phase."""
        model = self.wrench_sim.model
        data = self.wrench_sim.data
        for idx, site in enumerate(self.wrench_site_names):
            sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site)
            if sid < 0:
                continue
            self.pose_command[idx, :3] = data.site_xpos[sid]
            rotvec = R.from_matrix(data.site_xmat[sid].reshape(3, 3)).as_rotvec()
            self.pose_command[idx, 3:6] = rotvec.astype(np.float32)

    def capture_object_init(self) -> None:
        """Store object's initial pose once."""
        if self.object_init_pos is None:
            init_pos = OBJECT_INIT_POS_MAP.get(self.object_type)
            if init_pos is None:
                init_pos = np.zeros(3, dtype=np.float32)
            self.object_init_pos = init_pos.copy()

        if self.object_init_quat is None:
            object_info = OBJECT_MASS_MAP.get(
                self.object_type, OBJECT_MASS_MAP["unknown"]
            )
            init_quat = object_info.get("init_quat")
            if init_quat is None:
                init_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
            self.object_init_quat = init_quat.copy()

    def fix_object(self, sim: Any, sim_name: str = "sim") -> None:
        """Keep object fixed at the captured pose during close phase."""
        if "real" in str(sim_name).lower():
            return
        if self.object_init_pos is None or self.object_init_quat is None:
            return

        body_id = mujoco.mj_name2id(
            sim.model, mujoco.mjtObj.mjOBJ_BODY, self.object_body_name
        )
        if body_id < 0:
            if self.object_init_pos is None or self.object_init_quat is None:
                self.object_init_pos = np.zeros(3, dtype=np.float32)
                self.object_init_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
            return

        # Resolve the first joint attached to this body (free joint expected).
        jnt_adr = sim.model.body_jntadr[body_id]
        self.object_qpos_adr = int(sim.model.jnt_qposadr[jnt_adr])
        self.object_qvel_adr = int(sim.model.jnt_dofadr[jnt_adr])
        if (
            self.object_qpos_adr is not None
            and self.object_qpos_adr + 7 <= sim.model.nq
        ):
            qpos_slice = slice(self.object_qpos_adr, self.object_qpos_adr + 7)
            sim.data.qpos[qpos_slice][0:3] = self.object_init_pos
            sim.data.qpos[qpos_slice][3:7] = self.object_init_quat
        if (
            self.object_qvel_adr is not None
            and self.object_qvel_adr + 6 <= sim.model.nv
        ):
            qvel_slice = slice(self.object_qvel_adr, self.object_qvel_adr + 6)
            sim.data.qvel[qvel_slice] = 0.0

    def apply_pd(self, sim: Any) -> None:
        # Kept for API parity with toddlerbot policy; no-op in standalone mode.
        return

    def get_system_state(self) -> Dict[str, np.ndarray]:
        """Return object and fingertip state using site positions.

        Thumb contact -> fix_*, index contact -> left_*, middle contact -> right_*.
        Positions come from site poses (not contact points).
        Orientation/velocities come from the fingertip sites.
        """
        model: mujoco.MjModel = self.wrench_sim.model
        data: mujoco.MjData = self.wrench_sim.data

        def get_sensor_data(sensor_name: str) -> Optional[np.ndarray]:
            try:
                sensor_id = model.sensor(sensor_name).id
            except Exception:
                return None
            sensor_adr = model.sensor_adr[sensor_id]
            sensor_dim = model.sensor_dim[sensor_id]
            return data.sensordata[sensor_adr : sensor_adr + sensor_dim].copy()

        def tip_state(tip_name: str) -> Dict[str, np.ndarray]:
            pos = get_sensor_data(f"{tip_name}_framepos")
            quat = get_sensor_data(f"{tip_name}_framequat")
            linvel = get_sensor_data(f"{tip_name}_framelinvel")
            angvel = get_sensor_data(f"{tip_name}_frameangvel")
            if pos is None:
                pos = np.zeros(3, dtype=np.float32)
            if quat is None:
                quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
            if linvel is None:
                linvel = np.zeros(3, dtype=np.float32)
            if angvel is None:
                angvel = np.zeros(3, dtype=np.float32)
            return {
                "pos": np.asarray(pos, dtype=np.float32),
                "quat": np.asarray(quat, dtype=np.float32),
                "linvel": np.asarray(linvel, dtype=np.float32),
                "angvel": np.asarray(angvel, dtype=np.float32),
                "force": np.zeros(3, dtype=np.float32),
                "torque": np.zeros(3, dtype=np.float32),
            }

        thumb = tip_state("th_tip")
        index = tip_state("if_tip")
        middle = tip_state("mf_tip")

        def relative_quat(tip_name: str, quat_wxyz: np.ndarray) -> np.ndarray:
            if self.phase != "rotate":
                return quat_wxyz
            base = self.baseline_tip_rot.get(tip_name)
            if base is None:
                return quat_wxyz
            curr = R.from_quat(
                np.array(
                    [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]],
                    dtype=np.float32,
                )
            )
            rel = curr * base.inv()
            rel_xyzw = rel.as_quat()
            return np.array(
                [rel_xyzw[3], rel_xyzw[0], rel_xyzw[1], rel_xyzw[2]], dtype=np.float32
            )

        thumb["quat"] = relative_quat("th_tip", thumb["quat"])
        index["quat"] = relative_quat("if_tip", index["quat"])
        middle["quat"] = relative_quat("mf_tip", middle["quat"])

        # Object state from integrated target (no sensor fallback).
        if self.object_init_pos is None:
            obj_pos = np.zeros(3, dtype=np.float32)
        else:
            obj_pos = self.object_init_pos.copy()
        obj_pos += np.array([self.integrated_position, 0.0, 0.0], dtype=np.float32)

        if self.object_init_quat is None:
            base_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        else:
            base_quat = self.object_init_quat
        rot_delta = R.from_rotvec(
            np.array([0.0, 0.0, self.integrated_angle], dtype=np.float32)
        )
        base_rot = R.from_quat(base_quat, scalar_first=True)
        obj_quat = (rot_delta * base_rot).as_quat(scalar_first=True).astype(np.float32)
        obj_linvel = np.zeros(3, dtype=np.float32)
        obj_angvel = np.zeros(3, dtype=np.float32)

        # print(
        #     f"[Expected] pos=[{obj_pos[0]:+.3f}, {obj_pos[1]:+.3f}, {obj_pos[2]:+.3f}] m, rotvec: f{R.from_quat(obj_quat, scalar_first=True).as_rotvec()}, integrated_pos={self.integrated_position:+.3f}, integrated_angle={self.integrated_angle:+.3f}"
        # )

        return {
            "sliding_cube_pos": obj_pos,
            "sliding_cube_quat": obj_quat,
            "sliding_cube_linvel": obj_linvel,
            "sliding_cube_angvel": obj_angvel,
            "fix_traj_pos": thumb["pos"],
            "fix_traj_quat": thumb["quat"],
            "fix_traj_linvel": thumb["linvel"],
            "fix_traj_angvel": thumb["angvel"],
            "fix_traj_force": thumb["force"],
            "fix_traj_torque": thumb["torque"],
            "control_left_pos": index["pos"],
            "control_left_quat": index["quat"],
            "control_left_linvel": index["linvel"],
            "control_left_angvel": index["angvel"],
            "control_left_force": index["force"],
            "control_left_torque": index["torque"],
            "control_right_pos": middle["pos"],
            "control_right_quat": middle["quat"],
            "control_right_linvel": middle["linvel"],
            "control_right_angvel": middle["angvel"],
            "control_right_force": middle["force"],
            "control_right_torque": middle["torque"],
        }

    def get_target_vel(self, state):
        p_thumb_obj = state["fix_traj_pos"] - state["sliding_cube_pos"]
        thumb_linvel = self.target_rotation_linvel + np.cross(
            self.target_rotation_angvel, p_thumb_obj
        )
        thumb_angvel = np.zeros(3)

        v_obj_goal = np.cross(self.target_rotation_angvel - thumb_angvel, -p_thumb_obj)
        omega_obj_goal = self.target_rotation_angvel - thumb_angvel

        return v_obj_goal, omega_obj_goal, thumb_linvel, thumb_angvel

    def handle_rotate_action(self, state: Optional[Dict[str, np.ndarray]] = None):
        if state is None:
            state = self.get_system_state()
        target_linvel, target_angvel, thumb_linvel, thumb_angvel = self.get_target_vel(
            state
        )

        min_force = (
            self.min_normal_force_rotation
            if self.control_mode == "rotation"
            else self.min_normal_force_translation
        )
        # print(target_linvel, target_angvel)
        hfvc_inputs = compute_hfvc_inputs(
            state,
            goal_velocity=target_linvel.reshape(-1, 1),
            goal_angvel=target_angvel.reshape(-1, 1),
            friction_coeff_hand=0.8,
            min_normal_force=min_force,
            jac_phi_q_cube_rotating=self.jacobian_constraint,
            object_mass=self.object_mass,
            object_type=self.object_type,
            geom_size=self.object_geom_size,
        )
        hfvc_solution = solve_ochs(*hfvc_inputs, kNumSeeds=1, kPrintLevel=0)
        if hfvc_solution is None:
            return

        self.distribute_action(hfvc_solution, thumb_linvel, thumb_angvel, state)

    def ensure_rotvec_continuity(
        self, old_rotvec: np.ndarray, new_rotvec: np.ndarray
    ) -> np.ndarray:
        """Ensure rotvec sign consistency to avoid jumps at pi boundary.

        When a rotation vector crosses the pi boundary, scipy can flip its sign
        (since rotvec and -rotvec represent rotations differing by 2*pi).
        This function ensures the new rotvec maintains the same sign as the old one.

        Args:
            old_rotvec: Previous rotation vector (3,)
            new_rotvec: New rotation vector that might have flipped sign (3,)

        Returns:
            Corrected rotation vector with consistent sign
        """
        # Check if the dot product is negative (opposite directions)
        if np.dot(old_rotvec, new_rotvec) < 0:
            # Flip the sign to maintain continuity
            return -new_rotvec
        return new_rotvec

    def distribute_action(
        self, hfvc_solution, thumb_linvel, thumb_angvel, state
    ) -> None:
        """Distribute HFVC center commands to index/middle fingertips."""
        # Convert HFVC (center) wrench/velocity to contact-level targets.
        global_vel, global_frc = transform_hfvc_to_global(hfvc_solution)
        # print(f"global_vel:{global_vel.ravel()}, global_force:{global_frc.ravel()}")

        p_H, _, _, _ = get_center_state(state)
        p_fix = state["fix_traj_pos"].reshape(3)
        r = p_H.reshape(3) - p_fix
        coriolis_term = np.cross(thumb_angvel, r)
        global_vel[:3] += (thumb_linvel + coriolis_term).reshape(3, 1)
        global_vel[3:6] += thumb_angvel.reshape(3, 1)

        v_center = global_vel[:3].reshape(-1)
        omega = global_vel[3:6].reshape(-1)
        F_center = global_frc[:3].reshape(-1, 1)
        M_center = global_frc[3:6].reshape(-1, 1)

        # Use live site poses (not cached state) for distribution.
        model = self.wrench_sim.model
        data = self.wrench_sim.data
        idx_if = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "if_tip")
        idx_mf = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "mf_tip")
        if idx_if < 0 or idx_mf < 0:
            return
        p_left = data.site_xpos[idx_if].reshape(3, 1)
        p_right = data.site_xpos[idx_mf].reshape(3, 1)
        center_pos = 0.5 * (p_left + p_right)
        r_left = p_left - center_pos
        r_right = p_right - center_pos

        def cross3(a: np.ndarray, b: np.ndarray) -> np.ndarray:
            return np.array(
                [
                    a[1] * b[2] - a[2] * b[1],
                    a[2] * b[0] - a[0] * b[2],
                    a[0] * b[1] - a[1] * b[0],
                ],
                dtype=np.float32,
            ).reshape(-1, 1)

        v_left = v_center.reshape(3, 1) + cross3(omega, r_left.flatten())
        v_right = v_center.reshape(3, 1) + cross3(omega, r_right.flatten())

        # Set stiffness: index/middle follow their velocities; others stiff.
        self.assign_stiffness(v_left, v_right)

        def skew(v: np.ndarray) -> np.ndarray:
            return np.array(
                [[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]],
                dtype=np.float32,
            )

        A = np.zeros((6, 6), dtype=np.float32)
        A[0:3, 0:3] = np.eye(3, dtype=np.float32)
        A[0:3, 3:6] = np.eye(3, dtype=np.float32)
        A[3:6, 0:3] = skew(r_left.flatten())
        A[3:6, 3:6] = skew(r_right.flatten())
        b = np.vstack([F_center.astype(np.float32), M_center.astype(np.float32)])
        forces, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        f_left = forces[0:3].reshape(-1)
        f_right = forces[3:6].reshape(-1)

        # Add centripetal force: force toward object center for each end effector
        object_center = p_H.reshape(3)  # Object center position

        # Calculate centripetal force for left finger (if_tip)
        dir_left_to_center = object_center - p_left.flatten()
        dist_left = np.linalg.norm(dir_left_to_center)
        if dist_left > 1e-6:
            centripetal_left = (
                dir_left_to_center / dist_left
            ) * self.centripetal_force_magnitude
            f_left = f_left + centripetal_left

        # Calculate centripetal force for right finger (mf_tip)
        dir_right_to_center = object_center - p_right.flatten()
        dist_right = np.linalg.norm(dir_right_to_center)
        if dist_right > 1e-6:
            centripetal_right = (
                dir_right_to_center / dist_right
            ) * self.centripetal_force_magnitude
            f_right = f_right + centripetal_right

        # Write distributed targets into wrench_command and optional velocity targets.
        # We only set forces here; torques kept zero for fingertips.
        for tip_name, force in zip(("if_tip", "mf_tip"), (f_left, f_right)):
            if tip_name not in self.wrench_site_names:
                continue
            tip_idx = self.wrench_site_names.index(tip_name)
            self.wrench_command[tip_idx, :3] = force.astype(np.float32)
            self.wrench_command[tip_idx, 3:] = 0.0
            force_mag = float(np.linalg.norm(self.wrench_command[tip_idx, :3]))
            print(
                f"[Force] {tip_name} |wrench|={force_mag:.3f} N, force={self.wrench_command[tip_idx, :3]}"
            )
        if "th_tip" in self.wrench_site_names:
            thumb_idx = self.wrench_site_names.index("th_tip")
            self.wrench_command[thumb_idx, :3] = np.array(
                [-float(global_frc[0]), 0.0, 0.0], dtype=np.float32
            )
            self.wrench_command[thumb_idx, 3:] = 0.0
            thumb_force_mag = float(np.linalg.norm(self.wrench_command[thumb_idx, :3]))
            print(
                f"[Force] th_tip |wrench|={thumb_force_mag:.3f} N, force={self.wrench_command[thumb_idx, :3]}"
            )

        # Optionally, update pose_command velocities via a small feed-forward step.
        dt = self.control_dt
        angvel = self.target_rotation_angvel.copy()
        # angvel[2] = 0.0
        if_mf_rot_increment = R.from_rotvec(angvel * dt)
        for tip_name, vel in zip(("if_tip", "mf_tip"), (v_left, v_right)):
            if tip_name not in self.wrench_site_names:
                continue
            tip_idx = self.wrench_site_names.index(tip_name)
            self.pose_command[tip_idx, :3] += (vel.reshape(-1) * dt).astype(np.float32)
            # Compose rotations correctly: R_new = R_inc * R_curr
            old_rotvec = self.pose_command[tip_idx, 3:6].copy()
            curr_rot = R.from_rotvec(old_rotvec)
            new_rot = if_mf_rot_increment * curr_rot
            new_rotvec = new_rot.as_rotvec().astype(np.float32)
            # Ensure sign continuity to avoid jumps
            # new_rotvec = self.ensure_rotvec_continuity(old_rotvec, new_rotvec)
            self.pose_command[tip_idx, 3:6] = new_rotvec

        # set the thumb linvel and angvel
        thumb_idx = self.wrench_site_names.index("th_tip")
        self.pose_command[thumb_idx, :3] += (thumb_linvel.reshape(-1) * dt).astype(
            np.float32
        )

        old_rotvec_thumb = self.pose_command[thumb_idx, 3:6].copy()
        curr_rot = R.from_rotvec(old_rotvec_thumb)
        rot_increment = R.from_rotvec(self.target_rotation_angvel * dt)
        # print(self.target_rotation_angvel)
        new_rot = rot_increment * curr_rot
        new_rotvec_thumb = new_rot.as_rotvec().astype(np.float32)
        # Ensure sign continuity to avoid jumps
        # new_rotvec_thumb = self.ensure_rotvec_continuity(
        #     old_rotvec_thumb, new_rotvec_thumb
        # )
        # print(new_rotvec_thumb)

        # integrated_angle = R.from_rotvec(self.integrated_angle_thumb)
        # self.integrated_angle_thumb = (rot_increment * integrated_angle).as_rotvec()
        # print(self.integrated_angle_thumb)
        self.pose_command[thumb_idx, 3:6] = new_rotvec_thumb
        # print(self.pose_command[thumb_idx, 0:3])


def _find_repo_root(start_dir: str) -> str:
    cur = os.path.abspath(start_dir)
    while True:
        if os.path.isfile(os.path.join(cur, "pyproject.toml")):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            raise FileNotFoundError("Could not locate repository root (pyproject.toml).")
        cur = parent


def _deep_update(dst: dict, src: dict) -> dict:
    for key, value in src.items():
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            _deep_update(dst[key], value)
        else:
            dst[key] = value
    return dst


def _load_motor_params(
    repo_root: str,
    robot_desc_dir: str,
    model: mujoco.MjModel,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
]:
    default_path = os.path.join(repo_root, "examples", "descriptions", "default.yml")
    robot_path = os.path.join(robot_desc_dir, "robot.yml")
    motors_path = os.path.join(robot_desc_dir, "motors.yml")

    with open(default_path, "r") as f:
        config = yaml.safe_load(f)
    with open(robot_path, "r") as f:
        robot_cfg = yaml.safe_load(f)
    if robot_cfg is not None:
        _deep_update(config, robot_cfg)
    with open(motors_path, "r") as f:
        motor_cfg = yaml.safe_load(f)
    if motor_cfg is not None:
        _deep_update(config, motor_cfg)

    kp_ratio = float(config["actuators"]["kp_ratio"])
    kd_ratio = float(config["actuators"]["kd_ratio"])
    passive_active_ratio = float(config["actuators"]["passive_active_ratio"])

    names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        for i in range(model.nu)
    ]
    kp = []
    kd = []
    tau_max = []
    q_dot_max = []
    tau_q_dot_max = []
    q_dot_tau_max = []
    tau_brake_max = []
    kd_min = []

    for name in names:
        motor_key = name
        if motor_key not in config["motors"] and motor_key.endswith("_act"):
            base_key = motor_key[: -len("_act")]
            if base_key in config["motors"]:
                motor_key = base_key
        if motor_key not in config["motors"]:
            raise ValueError(f"Missing motor config for actuator '{name}'")

        motor_entry = config["motors"][motor_key]
        motor_type = motor_entry["motor"]
        act_cfg = config["actuators"][motor_type]

        kp.append(float(motor_entry.get("kp", 0.0)) / kp_ratio)
        kd.append(float(motor_entry.get("kd", 0.0)) / kd_ratio)
        tau_max.append(float(act_cfg["tau_max"]))
        q_dot_max.append(float(act_cfg["q_dot_max"]))
        tau_q_dot_max.append(float(act_cfg["tau_q_dot_max"]))
        q_dot_tau_max.append(float(act_cfg["q_dot_tau_max"]))
        tau_brake_max.append(float(act_cfg["tau_brake_max"]))
        kd_min.append(float(act_cfg["kd_min"]))

    return (
        np.asarray(kp, dtype=np.float32),
        np.asarray(kd, dtype=np.float32),
        np.asarray(tau_max, dtype=np.float32),
        np.asarray(q_dot_max, dtype=np.float32),
        np.asarray(tau_q_dot_max, dtype=np.float32),
        np.asarray(q_dot_tau_max, dtype=np.float32),
        np.asarray(tau_brake_max, dtype=np.float32),
        np.asarray(kd_min, dtype=np.float32),
        passive_active_ratio,
    )


def _build_controller(scene_xml_path: str, control_dt: float) -> ComplianceController:
    scene_xml_path = os.path.abspath(scene_xml_path)
    scene_dir = os.path.dirname(scene_xml_path)
    fixed_model_xml = os.path.join(scene_dir, "left_hand_fixed.xml")

    site_names = ("if_tip", "mf_tip", "rf_tip", "th_tip")

    controller_cfg = ControllerConfig(
        xml_path=scene_xml_path,
        site_names=site_names,
        fixed_base=True,
        base_body_name="",
        joint_indices_by_site={
            "if_tip": np.array([0, 1, 2, 3], dtype=np.int32),
            "mf_tip": np.array([4, 5, 6, 7], dtype=np.int32),
            "rf_tip": np.array([8, 9, 10, 11], dtype=np.int32),
            "th_tip": np.array([12, 13, 14, 15], dtype=np.int32),
        },
        motor_indices_by_site={
            "if_tip": np.array([0, 1, 2, 3], dtype=np.int32),
            "mf_tip": np.array([4, 5, 6, 7], dtype=np.int32),
            "rf_tip": np.array([8, 9, 10, 11], dtype=np.int32),
            "th_tip": np.array([12, 13, 14, 15], dtype=np.int32),
        },
        gear_ratios_by_site={
            "if_tip": np.ones(4, dtype=np.float32),
            "mf_tip": np.ones(4, dtype=np.float32),
            "rf_tip": np.ones(4, dtype=np.float32),
            "th_tip": np.ones(4, dtype=np.float32),
        },
    )

    estimate_cfg = WrenchEstimateConfig(
        force_reg=1e-3,
        torque_reg=1e-2,
        force_only=False,
        estimate_full_wrench=True,
        normal_axis="+z",
    )

    ref_cfg = ComplianceRefConfig(
        dt=float(control_dt),
        q_start_idx=0,
        qd_start_idx=0,
        ik_position_only=False,
        fixed_model_xml_path=fixed_model_xml,
        mass=1.0,
        inertia_diag=(1.0, 1.0, 1.0),
        mink_num_iter=5,
        mink_damping=1e-2,
        actuator_indices=tuple(range(16)),
        joint_to_actuator_scale=tuple([1.0] * 16),
        joint_to_actuator_bias=tuple([0.0] * 16),
    )

    return ComplianceController(
        config=controller_cfg,
        estimate_config=estimate_cfg,
        ref_config=ref_cfg,
    )


def _build_command_matrix(
    site_names: tuple[str, ...],
    policy_out: Dict[str, np.ndarray | str],
    measured_wrenches: Dict[str, np.ndarray],
) -> np.ndarray:
    pose_command = np.asarray(policy_out["pose_command"], dtype=np.float32)
    wrench_command = np.asarray(policy_out["wrench_command"], dtype=np.float32)
    pos_stiffness = np.asarray(policy_out["pos_stiffness"], dtype=np.float32)
    rot_stiffness = np.asarray(policy_out["rot_stiffness"], dtype=np.float32)
    pos_damping = np.asarray(policy_out["pos_damping"], dtype=np.float32)
    rot_damping = np.asarray(policy_out["rot_damping"], dtype=np.float32)

    num_sites = len(site_names)
    if pose_command.shape[0] != num_sites:
        raise ValueError(
            f"pose_command rows ({pose_command.shape[0]}) != num_sites ({num_sites})"
        )

    command_matrix = np.zeros((num_sites, COMMAND_LAYOUT.width), dtype=np.float32)
    command_matrix[:, COMMAND_LAYOUT.position] = pose_command[:, :3]
    command_matrix[:, COMMAND_LAYOUT.orientation] = pose_command[:, 3:6]
    command_matrix[:, COMMAND_LAYOUT.kp_pos] = pos_stiffness
    command_matrix[:, COMMAND_LAYOUT.kp_rot] = rot_stiffness
    command_matrix[:, COMMAND_LAYOUT.kd_pos] = pos_damping
    command_matrix[:, COMMAND_LAYOUT.kd_rot] = rot_damping
    command_matrix[:, COMMAND_LAYOUT.force] = wrench_command[:, :3]
    command_matrix[:, COMMAND_LAYOUT.torque] = wrench_command[:, 3:]

    for idx, site_name in enumerate(site_names):
        wrench = measured_wrenches.get(site_name)
        if wrench is None:
            continue
        wrench = np.asarray(wrench, dtype=np.float32).reshape(-1)
        if wrench.shape[0] < 6:
            continue
        command_matrix[idx, COMMAND_LAYOUT.measured_force] = wrench[:3]
        command_matrix[idx, COMMAND_LAYOUT.measured_torque] = wrench[3:6]

    return command_matrix


def _get_ground_truth_wrenches(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    site_names: tuple[str, ...],
) -> Dict[str, np.ndarray]:
    """Match toddlerbot sim behavior: use body cfrc_ext for each fingertip site."""
    mujoco.mj_rnePostConstraint(model, data)
    wrenches: Dict[str, np.ndarray] = {}
    for site_name in site_names:
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if site_id < 0:
            continue
        body_id = int(model.site_bodyid[site_id])
        wrenches[site_name] = np.asarray(data.cfrc_ext[body_id], dtype=np.float32).copy()
    return wrenches


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Self-contained LEAP rotate compliance demo."
    )
    parser.add_argument(
        "--scene-xml",
        type=str,
        default="",
        help="Path to MuJoCo scene xml (default: leap_hand_rotation/scene_fixed.xml).",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=120.0,
        help="Simulation duration in seconds. <=0 means run forever.",
    )
    parser.add_argument(
        "--control-dt",
        type=float,
        default=0.02,
        help="Control update period in seconds.",
    )
    parser.add_argument(
        "--prep-duration",
        type=float,
        default=7.0,
        help="Prep stage duration in seconds (toddlerbot default: 7.0).",
    )
    parser.add_argument(
        "--keyboard-port",
        type=int,
        default=5592,
        help="ZMQ keyboard receiver port for c/r commands.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without MuJoCo viewer.",
    )
    parser.add_argument(
        "--realtime",
        dest="realtime",
        action="store_true",
        help="Sleep to approximately match real-time playback.",
    )
    parser.add_argument(
        "--no-realtime",
        dest="realtime",
        action="store_false",
        help="Run as fast as possible without real-time pacing.",
    )
    parser.set_defaults(realtime=True)
    parser.add_argument(
        "--status-interval",
        type=float,
        default=1.0,
        help="Seconds between status prints.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    repo_root = _find_repo_root(SCRIPT_DIR)

    if args.scene_xml:
        scene_xml_path = os.path.abspath(args.scene_xml)
    else:
        scene_xml_path = os.path.join(
            repo_root,
            "examples",
            "descriptions",
            "leap_hand_rotation",
            "scene_fixed.xml",
        )

    controller = _build_controller(scene_xml_path, args.control_dt)
    model = controller.wrench_sim.model
    data = controller.wrench_sim.data
    site_names = tuple(controller.config.site_names)

    if controller.compliance_ref is None:
        raise RuntimeError("Controller compliance_ref is not initialized.")

    # Reset to model default posture from reference model.
    data.qpos[:] = controller.compliance_ref.default_qpos.copy()
    mujoco.mj_forward(model, data)

    policy = LeapRotateCompliancePolicy(
        wrench_sim=controller.wrench_sim,
        wrench_site_names=site_names,
        control_dt=float(controller.ref_config.dt),
        prep_duration=max(float(args.prep_duration), 0.0),
        auto_switch_target_enabled=True,
        control_port=int(args.keyboard_port),
    )
    # Align with toddlerbot behavior: force object to policy init pose at startup.
    policy.forward_object_to_init(sim_name="sim")

    target_motor_pos = np.asarray(
        controller.compliance_ref.default_motor_pos, dtype=np.float32
    )
    measured_wrenches: Dict[str, np.ndarray] = {}
    next_control_time = 0.0
    control_dt = float(controller.ref_config.dt)

    trnid = np.asarray(model.actuator_trnid[:, 0], dtype=np.int32)
    qpos_adr = model.jnt_qposadr[trnid]
    qvel_adr = model.jnt_dofadr[trnid]

    prep_start_motor_pos = np.asarray(data.qpos[qpos_adr], dtype=np.float32).copy()
    prep_target_motor_pos = np.asarray(PREPARE_POS, dtype=np.float32)
    if prep_target_motor_pos.shape != prep_start_motor_pos.shape:
        prep_target_motor_pos = prep_start_motor_pos.copy()
    prep_duration = float(policy.prep_duration)
    prep_hold_duration = min(5.0, prep_duration)
    prep_ramp_duration = max(prep_duration - prep_hold_duration, 1e-6)

    robot_desc_dir = os.path.dirname(scene_xml_path)
    (
        kp,
        kd,
        tau_max,
        q_dot_max,
        tau_q_dot_max,
        q_dot_tau_max,
        tau_brake_max,
        kd_min,
        passive_active_ratio,
    ) = _load_motor_params(repo_root, robot_desc_dir, model)

    status_interval = max(float(args.status_interval), 1e-3)
    next_status_time = 0.0
    wall_t0 = time.time()

    if not args.headless:
        controller.wrench_sim._ensure_viewer()

    def _fix_object_high_frequency_if_needed() -> None:
        """Clamp object pose every physics step during prep/close to avoid jitter."""
        sim_time_local = float(data.time)
        if sim_time_local < prep_duration or policy.phase == "close":
            policy.capture_object_init()
            policy.fix_object(controller.wrench_sim, sim_name="sim")
            mujoco.mj_forward(model, data)

    try:
        while True:
            sim_time = float(data.time)

            # Apply object fixation at physics-step frequency (not only control_dt).
            _fix_object_high_frequency_if_needed()

            if args.duration > 0.0 and sim_time >= float(args.duration):
                print("[leaphand] Reached duration limit, exiting.")
                break

            if not args.headless and controller.wrench_sim.viewer is not None:
                if not controller.wrench_sim.viewer.is_running():
                    print("[leaphand] Viewer closed, exiting.")
                    break

            if sim_time + 1e-9 >= next_control_time:
                # Match toddlerbot: use ground-truth contact wrench from cfrc_ext.
                measured_wrenches = _get_ground_truth_wrenches(model, data, site_names)

                # During prep, mimic toddlerbot's base interpolation:
                # duration=7s with end_time=5s => 2s ramp then hold.
                if sim_time < prep_duration:
                    policy.step(
                        time_curr=sim_time,
                        wrenches_by_site=measured_wrenches,
                        sim_name="sim",
                        is_real_world=False,
                    )
                    if sim_time < prep_ramp_duration:
                        alpha = float(
                            np.clip(sim_time / max(prep_ramp_duration, 1e-6), 0.0, 1.0)
                        )
                        target_motor_pos = (
                            prep_start_motor_pos
                            + (prep_target_motor_pos - prep_start_motor_pos) * alpha
                        ).astype(np.float32)
                    else:
                        target_motor_pos = prep_target_motor_pos.copy()
                else:
                    policy_out = policy.step(
                        time_curr=sim_time,
                        wrenches_by_site=measured_wrenches,
                        sim_name="sim",
                        is_real_world=False,
                    )
                    command_matrix = _build_command_matrix(
                        site_names=site_names,
                        policy_out=policy_out,
                        measured_wrenches=measured_wrenches,
                    )

                    inputs = ComplianceInputs(
                        motor_torques=np.asarray(data.actuator_force, dtype=np.float32),
                        qpos=np.asarray(data.qpos, dtype=np.float32),
                        time=sim_time,
                        command_matrix=command_matrix,
                    )
                    out = controller.step(inputs, use_estimated_wrench=False)
                    if "state_ref" in out:
                        target_motor_pos = np.asarray(
                            out["state_ref"]["motor_pos"], dtype=np.float32
                        )
                next_control_time += control_dt

            q = data.qpos[qpos_adr]
            q_dot = data.qvel[qvel_adr]
            q_dot_dot = data.qacc[qvel_adr]
            error = target_motor_pos - q

            real_kp = np.where(q_dot_dot * error < 0, kp * passive_active_ratio, kp)
            tau_m = real_kp * error - (kd_min + kd) * q_dot
            abs_q_dot = np.abs(q_dot)
            slope = (tau_q_dot_max - tau_max) / (q_dot_max - q_dot_tau_max)
            taper_limit = tau_max + slope * (abs_q_dot - q_dot_tau_max)
            tau_acc_limit = np.where(abs_q_dot <= q_dot_tau_max, tau_max, taper_limit)
            tau_m_clamped = np.where(
                np.logical_and(abs_q_dot > q_dot_max, q_dot * target_motor_pos > 0),
                np.where(
                    q_dot > 0,
                    np.ones_like(tau_m) * -tau_brake_max,
                    np.ones_like(tau_m) * tau_brake_max,
                ),
                np.where(
                    q_dot > 0,
                    np.clip(tau_m, -tau_brake_max, tau_acc_limit),
                    np.clip(tau_m, -tau_acc_limit, tau_brake_max),
                ),
            )
            data.ctrl[:] = tau_m_clamped

            mujoco.mj_step(model, data)
            _fix_object_high_frequency_if_needed()

            if not args.headless and controller.wrench_sim.viewer is not None:
                controller.wrench_sim.viewer.sync()

            if sim_time >= next_status_time:
                print(
                    f"[leaphand] t={sim_time:.2f}s phase={policy.phase} "
                    f"mode={policy.control_mode}"
                )
                next_status_time = sim_time + status_interval

            if args.realtime:
                target_wall = wall_t0 + sim_time
                sleep_time = target_wall - time.time()
                if sleep_time > 0:
                    time.sleep(min(sleep_time, 0.01))
    finally:
        controller.wrench_sim.close()
        if policy.control_receiver is not None:
            policy.control_receiver.close()


if __name__ == "__main__":
    main()
