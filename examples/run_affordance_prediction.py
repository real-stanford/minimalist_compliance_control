#!/usr/bin/env python3
"""Run affordance prediction and trajectory planning from static asset images."""

from __future__ import annotations

import argparse
import json
import pickle
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import yaml
from scipy.spatial.transform import Rotation as R

from vlm.affordance.affordance_predictor import AffordancePredictor
from vlm.affordance.plan_ee_pose import (
    plan_end_effector_poses,
    transform_normals,
    transform_points,
)

ASSETS_DIR = Path("assets")
DEFAULT_OUTPUT_ROOT = Path("results")
ROBOT_CHOICES = ("toddlerbot", "leap")
TASK_CHOICES = ("wipe", "draw")
ROBOT_VARIANT_MAP = {
    "toddlerbot": "toddlerbot_2xm",
    "leap": "leap_hand",
}
TORSO_HALF_SIZE = np.array([0.045, 0.065, 0.06], dtype=np.float32)


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
    parser.add_argument("--plot", action="store_true")
    return parser.parse_args()


def default_object_label(task: str) -> str:
    if task == "wipe":
        return "black ink. vase"
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
        head_position_world = np.array([-0.1752, 0.117, 0.0625], dtype=np.float32)
        head_quaternion_world_wxyz = R.from_euler("xyz", [np.pi, 0.0, 0.0]).as_quat(
            scalar_first=True
        )
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
        return f"wipe up the {object_label} on the vase with an eraser."
    return f"draw the {object_label} on the vase using the pen."


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
    robot: str, image_hw: Tuple[int, int], zmax: float, temp_dir: Path
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

    calib_path = temp_dir / "calibration.pkl"
    rect_path = temp_dir / "rectification.npz"
    with open(calib_path, "wb") as f:
        pickle.dump(calibration, f)
    with open(rect_path, "wb") as f:
        pickle.dump(rectification, f)

    height, width = image_hw
    return {
        "calib_params": str(calib_path),
        "rec_params": str(rect_path),
        "calib_width": int(width),
        "calib_height": int(height),
        "zmax": float(zmax),
    }


def save_run_args(
    out_path: Path,
    *,
    robot: str,
    robot_variant: str,
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
        "robot_variant": robot_variant,
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


def plot_frame(
    ax, origin: np.ndarray, rotation: np.ndarray, label: Optional[str] = None
) -> None:
    origin = np.asarray(origin, dtype=np.float32)
    rotation = np.asarray(rotation, dtype=np.float32)
    axis_len = 0.05
    axis_colors = ("r", "g", "b")
    for i, color in enumerate(axis_colors):
        axis = rotation[:, i]
        ax.quiver(
            origin[0],
            origin[1],
            origin[2],
            axis[0],
            axis[1],
            axis[2],
            color=color,
            length=axis_len,
        )
    if label:
        ax.text(origin[0], origin[1], origin[2], label, fontsize=8)


def plot_torso(ax, z_bottom: float = 0.309161) -> np.ndarray:
    hx, hy, hz = TORSO_HALF_SIZE
    z_top = z_bottom + 2.0 * hz
    corners = np.array(
        [
            [-hx, -hy, z_bottom],
            [hx, -hy, z_bottom],
            [hx, hy, z_bottom],
            [-hx, hy, z_bottom],
            [-hx, -hy, z_top],
            [hx, -hy, z_top],
            [hx, hy, z_top],
            [-hx, hy, z_top],
        ],
        dtype=np.float32,
    )
    edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ]
    for start, end in edges:
        pts = corners[[start, end]]
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color="dimgray", linewidth=1.0)
    return corners


def load_pose_data(
    data_path: Path,
) -> Dict[str, Union[np.ndarray, Dict[str, Tuple[np.ndarray, ...]]]]:
    try:
        import joblib
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError(
            "plotting requires joblib. Please install it in your environment."
        ) from exc

    if not data_path.exists():
        raise FileNotFoundError(f"Pose data file not found: {data_path}")
    payload = joblib.load(data_path)
    required_keys = {
        "world_T_left_camera",
        "world_T_right_camera",
        "trajectory_by_site",
    }
    missing = required_keys - set(payload.keys())
    if missing:
        raise KeyError(f"Pose data missing keys: {', '.join(sorted(missing))}")

    result: Dict[str, Union[np.ndarray, Dict[str, Tuple[np.ndarray, ...]]]] = {
        "world_T_left_camera": np.asarray(
            payload["world_T_left_camera"], dtype=np.float32
        ),
        "world_T_right_camera": np.asarray(
            payload["world_T_right_camera"], dtype=np.float32
        ),
    }

    traj_by_site_raw = payload.get("trajectory_by_site", {})
    trajectories: Dict[str, Tuple[np.ndarray, ...]] = {}
    if isinstance(traj_by_site_raw, dict):
        for site_name, traj in traj_by_site_raw.items():
            if isinstance(traj, tuple):
                trajectories[site_name] = tuple(np.asarray(comp) for comp in traj)
    result["trajectory_by_site"] = trajectories

    contact_points_camera = payload.get("contact_pos_camera")
    contact_normals_camera = payload.get("contact_normals_camera")
    if isinstance(contact_points_camera, dict):
        result["contact_pos_camera"] = {
            k: np.asarray(v, dtype=np.float32) for k, v in contact_points_camera.items()
        }
    if isinstance(contact_normals_camera, dict):
        result["contact_normals_camera"] = {
            k: np.asarray(v, dtype=np.float32)
            for k, v in contact_normals_camera.items()
        }

    return result


def plot_trajectory_profiles(
    plt, trajectories_by_site: Dict[str, Tuple[np.ndarray, ...]]
):
    if not trajectories_by_site:
        return None

    fig, axes = plt.subplots(4, 1, figsize=(9, 12))
    fig.subplots_adjust(hspace=0.3)
    palette = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf"]

    for idx, (site_name, traj) in enumerate(trajectories_by_site.items()):
        if len(traj) < 3:
            continue
        color = palette[idx % len(palette)]
        t = np.asarray(traj[0], dtype=np.float64)
        pos = np.asarray(traj[2], dtype=np.float64)
        if pos.ndim != 2 or pos.shape[1] != 3 or pos.shape[0] != t.size or t.size == 0:
            continue

        vel = np.gradient(pos, t, axis=0) if t.size > 1 else np.zeros_like(pos)
        speed = np.linalg.norm(vel, axis=1)

        axes[0].plot(pos[:, 0], pos[:, 1], color=color, linewidth=1.3, label=site_name)
        axes[1].plot(t, pos[:, 0], color=color, linestyle="-", label=f"{site_name} px")
        axes[1].plot(t, pos[:, 1], color=color, linestyle="--", label=f"{site_name} py")
        axes[1].plot(t, pos[:, 2], color=color, linestyle=":", label=f"{site_name} pz")
        axes[2].plot(t, vel[:, 0], color=color, linestyle="-", label=f"{site_name} vx")
        axes[2].plot(t, vel[:, 1], color=color, linestyle="--", label=f"{site_name} vy")
        axes[2].plot(t, vel[:, 2], color=color, linestyle=":", label=f"{site_name} vz")
        axes[3].plot(t, speed, color=color, label=f"{site_name} speed")

    axes[0].set_title("Planned Trajectory (XY)")
    axes[0].set_xlabel("X [m]")
    axes[0].set_ylabel("Y [m]")
    axes[0].axis("equal")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="best")

    axes[1].set_title("Position vs Time")
    axes[1].set_ylabel("Position [m]")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="best")

    axes[2].set_title("Velocity vs Time")
    axes[2].set_ylabel("Velocity [m/s]")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(loc="best")

    axes[3].set_title("Speed vs Time")
    axes[3].set_xlabel("Time [s]")
    axes[3].set_ylabel("Speed [m/s]")
    axes[3].grid(True, alpha=0.3)
    axes[3].legend(loc="best")
    return fig


def plot_prediction_results(prediction_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        print("[AffordanceRun] --plot requested but matplotlib is not installed.")
        return

    try:
        pose_data = load_pose_data(prediction_dir / "trajectory.lz4")
    except FileNotFoundError:
        print(f"[AffordanceRun] No trajectory.lz4 to plot in {prediction_dir}.")
        return
    except Exception as exc:
        print(f"[AffordanceRun] Failed to load plot data from {prediction_dir}: {exc}")
        return

    world_T_left_camera = np.asarray(pose_data["world_T_left_camera"], dtype=np.float32)
    world_T_right_camera = np.asarray(
        pose_data["world_T_right_camera"], dtype=np.float32
    )
    trajectories_by_site = pose_data.get("trajectory_by_site", {})
    trajectories_by_site = (
        trajectories_by_site if isinstance(trajectories_by_site, dict) else {}
    )
    contact_points_camera = pose_data.get("contact_pos_camera")
    contact_normals_camera = pose_data.get("contact_normals_camera")
    if not isinstance(contact_points_camera, dict):
        contact_points_camera = {}
    if not isinstance(contact_normals_camera, dict):
        contact_normals_camera = {}

    robot_variant = "toddlerbot_2xm"
    args_path = prediction_dir / "args.json"
    if args_path.exists():
        try:
            payload = json.loads(args_path.read_text(encoding="utf-8"))
            if isinstance(payload, dict) and "robot_variant" in payload:
                robot_variant = str(payload["robot_variant"])
        except Exception:
            pass

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    if robot_variant == "toddlerbot_2xm":
        plot_torso(ax)
    plot_frame(ax, np.zeros(3, dtype=np.float32), np.eye(3), "World")
    plot_frame(
        ax,
        world_T_left_camera[:3, 3],
        world_T_left_camera[:3, :3],
        "Left Camera",
    )
    plot_frame(
        ax,
        world_T_right_camera[:3, 3],
        world_T_right_camera[:3, :3],
        "Right Camera",
    )

    palette = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf"]
    for idx, (site_name, traj) in enumerate(trajectories_by_site.items()):
        if not isinstance(traj, tuple) or len(traj) < 4:
            continue
        positions = np.asarray(traj[2], dtype=np.float32)
        if positions.size == 0:
            continue
        color = palette[idx % len(palette)]
        ax.plot(
            positions[:, 0],
            positions[:, 1],
            positions[:, 2],
            color=color,
            linewidth=1.5,
            label=f"{site_name} path",
        )
        ax.scatter(
            positions[-1, 0],
            positions[-1, 1],
            positions[-1, 2],
            color=color,
            s=70,
            marker="X",
            label=f"{site_name} end",
        )

    for idx, (site_name, points_cam) in enumerate(contact_points_camera.items()):
        points_world = transform_points(points_cam, world_T_left_camera)
        if points_world.size == 0:
            continue
        color = palette[idx % len(palette)]
        ax.scatter(
            points_world[:, 0],
            points_world[:, 1],
            points_world[:, 2],
            color=color,
            s=18,
            marker="o",
            label=f"{site_name} contact",
        )
        if site_name in contact_normals_camera:
            normals_world = transform_normals(
                contact_normals_camera[site_name], world_T_left_camera
            )
            if normals_world.shape[0] == points_world.shape[0]:
                ax.quiver(
                    points_world[:, 0],
                    points_world[:, 1],
                    points_world[:, 2],
                    normals_world[:, 0],
                    normals_world[:, 1],
                    normals_world[:, 2],
                    color=color,
                    length=0.03,
                    normalize=True,
                    linewidth=0.6,
                )

    ax.set_title("Affordance Pose Planning")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.grid(True)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, loc="upper right")
    fig.tight_layout(rect=[0, 0, 1, 0.98])

    plot_trajectory_profiles(plt, trajectories_by_site)

    desired_order = [
        "left_raw.png",
        "right_raw.png",
        "segmentation_mask.png",
        "depth_map_vis.png",
        "left_rectified.png",
        "right_rectified.png",
        "candidate_points.png",
        "contact_points_overlay.png",
    ]
    image_paths = []
    for filename in desired_order:
        path = prediction_dir / filename
        image_paths.append(path if path.exists() else None)

    if any(path is not None for path in image_paths):
        rows, cols = 2, 4
        fig_images, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
        axes = axes.reshape(rows, cols)
        for idx, ax_img in enumerate(axes.flat):
            if idx < len(image_paths) and image_paths[idx] is not None:
                img = plt.imread(str(image_paths[idx]))
                ax_img.imshow(img)
                ax_img.set_title(desired_order[idx], fontsize=9)
            else:
                ax_img.set_facecolor("#f0f0f0")
                ax_img.text(
                    0.5,
                    0.5,
                    "Missing",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="gray",
                )
            ax_img.axis("off")
        fig_images.suptitle("Affordance Debug Images")

    plt.show()


def main() -> None:
    args = parse_args()
    robot = args.robot
    task = args.task
    robot_variant = ROBOT_VARIANT_MAP[robot]
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
    base_output_dir = (
        args.output_root / f"affordance_prediction_{robot}_{task}_{timestamp}"
    )
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

    with tempfile.TemporaryDirectory(prefix=f"{robot}_affordance_depth_") as tmp_dir:
        depth_config = build_depth_config(
            robot=robot,
            image_hw=first_left_img.shape[:2],
            zmax=args.zmax,
            temp_dir=Path(tmp_dir),
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
                    robot_variant=robot_variant,
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
                        robot_name=robot_variant,
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
                    site_name: np.zeros(6, dtype=np.float32)
                    for site_name in planned_sites
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
                    robot_name=robot_variant,
                )
                print(
                    f"[AffordanceRun] Planned trajectory for sites: {list(trajectory.keys())}"
                )
                if args.plot:
                    plot_prediction_results(output_dir)
        finally:
            predictor.close()


if __name__ == "__main__":
    main()
