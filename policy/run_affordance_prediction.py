#!/usr/bin/env python3
"""Run affordance prediction and trajectory planning from static asset images."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import yaml
from scipy.spatial.transform import Rotation as R

from minimalist_compliance_control.visualization import ComplianceVLMPlotter
from vlm.affordance.affordance_predictor import AffordancePredictor
from vlm.affordance.plan_ee_pose import plan_end_effector_poses

ASSETS_DIR = Path("assets")
DEFAULT_OUTPUT_ROOT = Path("results")
ROBOT_CHOICES = ("toddlerbot", "leap")
TASK_CHOICES = ("wipe", "draw")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run affordance prediction on stereo test images in assets/."
    )
    parser.add_argument("--robot", type=str, choices=ROBOT_CHOICES, required=True)
    parser.add_argument("--task", type=str, choices=TASK_CHOICES, required=True)
    parser.add_argument(
        "--object",
        type=str,
        default="",
        help="Semantic object label. Defaults per task if omitted.",
    )
    parser.add_argument(
        "--site",
        type=str,
        nargs="+",
        default=None,
        help="Optional site names; defaults are selected from robot+task.",
    )
    parser.add_argument("--provider", type=str, default="gemini")
    parser.add_argument("--model", type=str, default="gemini-2.5-pro")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--zmax", type=float, default=0.5)
    return parser.parse_args()


def default_object_label(task: str) -> str:
    if task == "wipe":
        return "black ink"
    return "star"


def default_site_names(robot: str, task: str) -> List[str]:
    if robot == "leap":
        return ["mf_tip"] if task == "wipe" else ["rf_tip", "if_tip"]
    return ["left_hand_center"] if task == "wipe" else ["right_hand_center"]


def planning_gains(
    robot: str, task: str
) -> Tuple[float, float, float, float, float, str]:
    if robot == "leap":
        tangent_pos_stiffness = 200.0
        normal_pos_stiffness = 20.0
        contact_force = 0.2 if task == "wipe" else 0.7
    else:
        tangent_pos_stiffness = 400.0
        normal_pos_stiffness = 80.0
        contact_force = 5.0
    tangent_rot_stiffness = 20.0
    normal_rot_stiffness = 5.0
    tool = "eraser" if task == "wipe" else "pen"
    return (
        tangent_pos_stiffness,
        normal_pos_stiffness,
        tangent_rot_stiffness,
        normal_rot_stiffness,
        contact_force,
        tool,
    )


def default_head_pose(robot: str) -> Tuple[np.ndarray, np.ndarray]:
    if robot == "leap":
        head_position_world = np.array([-0.12, -0.117, 0.167], dtype=np.float32)
        head_quaternion_world_wxyz = R.from_euler(
            "xyz", [0.0, -np.pi / 3, 0.0]
        ).as_quat(scalar_first=True)
    else:
        head_position_world = np.array([0.0, 0.0, 0.15], dtype=np.float32)
        head_quaternion_world_wxyz = R.from_euler("xyz", [0.0, np.pi / 4, 0.0]).as_quat(
            scalar_first=True
        )
    return (
        head_position_world.astype(np.float32),
        head_quaternion_world_wxyz.astype(np.float32),
    )


def task_description(task: str, object_label: str) -> str:
    if task == "wipe":
        return f"wipe up the {object_label} on the whiteboard with an eraser."
    return f"draw the {object_label} on the whiteboard using the pen."


def resolve_stereo_pairs(robot: str, task: str) -> List[Tuple[Path, Path]]:
    exact_left = ASSETS_DIR / f"{robot}_{task}_left.png"
    exact_right = ASSETS_DIR / f"{robot}_{task}_right.png"
    if exact_left.exists() and exact_right.exists():
        return [(exact_left, exact_right)]

    pairs: List[Tuple[Path, Path]] = []
    for left_path in sorted(ASSETS_DIR.glob(f"{robot}_{task}_left*.png")):
        right_name = left_path.name.replace("_left", "_right", 1)
        right_path = left_path.with_name(right_name)
        if right_path.exists():
            pairs.append((left_path, right_path))
    if not pairs:
        raise FileNotFoundError(
            f"No stereo pair found for robot='{robot}' task='{task}' under {ASSETS_DIR.resolve()}"
        )
    return pairs


def load_stereo_pair(
    left_path: Path, right_path: Path
) -> Tuple[np.ndarray, np.ndarray]:
    left = cv2.imread(str(left_path))
    right = cv2.imread(str(right_path))
    if left is None or right is None:
        raise FileNotFoundError(
            f"Failed to load stereo pair: {left_path.resolve()} / {right_path.resolve()}"
        )
    return left, right


def build_depth_config(
    robot: str, image_hw: Tuple[int, int], zmax: float
) -> Dict[str, object]:
    camera_yaml_path = ASSETS_DIR / f"{robot}_camera.yml"
    if not camera_yaml_path.exists():
        raise FileNotFoundError(f"Missing camera config: {camera_yaml_path.resolve()}")

    data = yaml.safe_load(camera_yaml_path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Camera YAML must be a mapping: {camera_yaml_path}")

    calibration = data.get("calibration")
    rectification = data.get("rectification")
    if not isinstance(calibration, dict) or not isinstance(rectification, dict):
        raise ValueError(
            f"Camera YAML must include mapping keys 'calibration' and 'rectification': {camera_yaml_path}"
        )

    def to_numpy_fields(mapping: Dict[str, object]) -> Dict[str, object]:
        converted: Dict[str, object] = {}
        for key, value in mapping.items():
            if isinstance(value, list):
                converted[key] = np.asarray(value, dtype=np.float64)
            else:
                converted[key] = value
        return converted

    calibration = to_numpy_fields(calibration)
    rectification = to_numpy_fields(rectification)

    height, width = image_hw
    return {
        "calib_params": calibration,
        "rec_params": rectification,
        "calib_width": int(width),
        "calib_height": int(height),
        "zmax": float(zmax),
    }


def save_run_args(
    out_path: Path,
    *,
    robot: str,
    task: str,
    object_label: str,
    site_names: List[str],
    task_text: str,
    left_image: Path,
    right_image: Path,
    head_pos: np.ndarray,
    head_quat: np.ndarray,
) -> None:
    payload = {
        "robot": robot,
        "task": task,
        "object": object_label,
        "site": site_names,
        "task_description": task_text,
        "left_image": str(left_image),
        "right_image": str(right_image),
        "head_position": head_pos.tolist(),
        "head_orientation": head_quat.tolist(),
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def plot_prediction_results(prediction_dir: Path) -> None:
    ComplianceVLMPlotter.plot_prediction_results(prediction_dir)


def main() -> None:
    args = parse_args()
    robot = args.robot
    task = args.task
    object_label = (
        args.object.strip() if args.object.strip() else default_object_label(task)
    )
    site_names = args.site if args.site else default_site_names(robot, task)
    is_wiping = task == "wipe"
    task_text = task_description(task, object_label)

    stereo_pairs = resolve_stereo_pairs(robot, task)
    first_left, first_right = stereo_pairs[0]
    first_left_img, _ = load_stereo_pair(first_left, first_right)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    base_output_dir = args.output_root / f"{robot}_{task}_affordance_{timestamp}"
    base_output_dir.mkdir(parents=True, exist_ok=True)

    head_pos, head_quat = default_head_pose(robot)
    (
        tangent_pos_stiffness,
        normal_pos_stiffness,
        tangent_rot_stiffness,
        normal_rot_stiffness,
        contact_force,
        tool,
    ) = planning_gains(robot, task)

    depth_config = build_depth_config(
        robot=robot,
        image_hw=first_left_img.shape[:2],
        zmax=args.zmax,
    )
    predictor = AffordancePredictor(
        provider=args.provider,
        model=args.model,
        default_task=task_text,
        depth_config=depth_config,
    )
    try:
        for idx, (left_path, right_path) in enumerate(stereo_pairs):
            output_dir = base_output_dir / f"prediction_{idx}"
            output_dir.mkdir(parents=True, exist_ok=True)
            save_run_args(
                output_dir / "args.json",
                robot=robot,
                task=task,
                object_label=object_label,
                site_names=site_names,
                task_text=task_text,
                left_image=left_path,
                right_image=right_path,
                head_pos=head_pos,
                head_quat=head_quat,
            )

            print(
                f"[AffordanceRun] Processing pair {idx}: "
                f"{left_path.name} / {right_path.name}"
            )
            left_image, right_image = load_stereo_pair(left_path, right_path)
            try:
                prediction = predictor.predict(
                    left_image=left_image,
                    right_image=right_image,
                    robot_name=robot,
                    site_names=site_names,
                    is_wiping=is_wiping,
                    output_dir=str(output_dir),
                    object_label=object_label,
                )
            except TimeoutError as exc:
                print(f"[AffordanceRun] Prediction timed out: {exc}")
                continue
            except Exception as exc:
                print(f"[AffordanceRun] Prediction failed: {exc}")
                continue

            if prediction is None:
                if is_wiping and predictor.last_wiping_done:
                    print(
                        "[AffordanceRun] Wiping appears complete; no trajectory planned."
                    )
                else:
                    print("[AffordanceRun] Predictor returned no contact points.")
                continue

            contact_points_3d, contact_normals = prediction
            planned_sites = list(contact_points_3d.keys())
            pose_cur = {
                site_name: np.zeros(6, dtype=np.float32) for site_name in planned_sites
            }
            trajectory = plan_end_effector_poses(
                contact_points_camera=contact_points_3d,
                contact_normals_camera=contact_normals,
                head_position_world=head_pos,
                head_quaternion_world_wxyz=head_quat,
                tangent_pos_stiffness=tangent_pos_stiffness,
                normal_pos_stiffness=normal_pos_stiffness,
                tangent_rot_stiffness=tangent_rot_stiffness,
                normal_rot_stiffness=normal_rot_stiffness,
                contact_force=np.asarray(contact_force, dtype=np.float32),
                pose_cur=pose_cur,
                output_dir=str(output_dir),
                tool=tool,
                robot_name=robot,
                task=task,
            )
            print(
                f"[AffordanceRun] Planned trajectory for sites: {list(trajectory.keys())}"
            )
            plot_prediction_results(output_dir)
    finally:
        predictor.close()


if __name__ == "__main__":
    main()
