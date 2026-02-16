"""Replay/dummy runner for standalone compliance_vlm."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np

from vlm.compliance_vlm import ComplianceVLMInput, StandaloneComplianceVLM


def _find_key(npz_obj, candidates: list[str]) -> Optional[str]:
    for key in candidates:
        if key in npz_obj:
            return key
    return None


def _to_hwc_u8(frame: np.ndarray) -> np.ndarray:
    arr = np.asarray(frame)
    if arr.ndim != 3:
        raise ValueError(f"Expected frame with 3 dims, got {arr.shape}")

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


def _reshape_site_matrix(
    arr: np.ndarray,
    num_sites: int,
    width: int,
) -> np.ndarray:
    out = np.asarray(arr, dtype=np.float32)
    if out.ndim == 1 and out.size == num_sites * width:
        return out.reshape(num_sites, width)
    if out.ndim == 2 and out.shape == (num_sites, width):
        return out
    raise ValueError(
        f"Invalid shape for site matrix, got {out.shape}, expected ({num_sites},{width})"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone compliance_vlm runner")
    parser.add_argument("--robot-name", type=str, default="toddlerbot_2xm")
    parser.add_argument(
        "--site-names", type=str, default="", help="Comma-separated site names"
    )
    parser.add_argument("--dt", type=float, default=0.02, help="Control dt")
    parser.add_argument(
        "--mode", type=str, default="waiting", choices=["waiting", "wiping", "drawing"]
    )
    parser.add_argument(
        "--object", type=str, default="black ink. vase", help="Target object label"
    )

    parser.add_argument("--replay-npz", type=str, default="", help="Replay npz path")
    parser.add_argument(
        "--left-key", type=str, default="", help="left image key in npz"
    )
    parser.add_argument(
        "--right-key", type=str, default="", help="right image key in npz"
    )
    parser.add_argument("--x-obs-key", type=str, default="", help="x_obs key in npz")
    parser.add_argument(
        "--x-wrench-key", type=str, default="", help="x_wrench key in npz"
    )
    parser.add_argument(
        "--head-pos-key", type=str, default="", help="head position key in npz"
    )
    parser.add_argument(
        "--head-quat-key", type=str, default="", help="head quaternion key in npz"
    )

    parser.add_argument(
        "--steps", type=int, default=300, help="Dummy steps if no replay"
    )
    parser.add_argument(
        "--save", type=str, default="", help="Output npz for pose/command"
    )
    parser.add_argument("--mode-control-port", type=int, default=5591)
    parser.add_argument("--disable-zmq", action="store_true")
    parser.add_argument("--use-camera-stream", action="store_true")
    parser.add_argument("--disable-video", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    site_names = None
    if len(args.site_names.strip()) > 0:
        site_names = [s.strip() for s in args.site_names.split(",") if s.strip()]

    policy = StandaloneComplianceVLM(
        site_names=site_names,
        robot_name=args.robot_name,
        control_dt=float(args.dt),
        mode_control_port=int(args.mode_control_port),
        enable_mode_control=not bool(args.disable_zmq),
        use_camera_stream=bool(args.use_camera_stream),
        record_video=not bool(args.disable_video),
    )

    if args.mode == "wiping":
        policy.set_mode(True, object_label=None, site_names=None)
    elif args.mode == "drawing":
        policy.set_mode(False, object_label=str(args.object), site_names=None)

    pose_list = []
    cmd_list = []
    status_list = []

    try:
        if len(args.replay_npz) > 0:
            replay = np.load(args.replay_npz)

            left_key = (
                args.left_key
                if len(args.left_key) > 0
                else _find_key(
                    replay, ["left_image", "image", "images", "rgb", "camera"]
                )
            )
            right_key = (
                args.right_key
                if len(args.right_key) > 0
                else _find_key(replay, ["right_image", "image_right", "right"])
            )
            x_obs_key = (
                args.x_obs_key
                if len(args.x_obs_key) > 0
                else _find_key(replay, ["x_obs", "pose", "ee_pose"])
            )
            x_wrench_key = (
                args.x_wrench_key
                if len(args.x_wrench_key) > 0
                else _find_key(replay, ["x_wrench", "wrench", "wrenches"])
            )
            head_pos_key = (
                args.head_pos_key
                if len(args.head_pos_key) > 0
                else _find_key(
                    replay, ["head_pos", "head_position", "head_position_world"]
                )
            )
            head_quat_key = (
                args.head_quat_key
                if len(args.head_quat_key) > 0
                else _find_key(
                    replay,
                    ["head_quat", "head_quaternion", "head_quaternion_world_wxyz"],
                )
            )

            if left_key is None:
                raise KeyError("Replay file missing left image key.")
            if x_obs_key is None:
                raise KeyError("Replay file missing x_obs key.")

            left_images = replay[left_key]
            right_images = replay[right_key] if right_key is not None else None
            x_obs_arr = replay[x_obs_key]
            x_wrench_arr = replay[x_wrench_key] if x_wrench_key is not None else None
            head_pos_arr = replay[head_pos_key] if head_pos_key is not None else None
            head_quat_arr = replay[head_quat_key] if head_quat_key is not None else None

            T = left_images.shape[0]
            T = min(T, x_obs_arr.shape[0])
            if right_images is not None:
                T = min(T, right_images.shape[0])
            if x_wrench_arr is not None:
                T = min(T, x_wrench_arr.shape[0])
            if head_pos_arr is not None:
                T = min(T, head_pos_arr.shape[0])
            if head_quat_arr is not None:
                T = min(T, head_quat_arr.shape[0])

            for i in range(T):
                left_image = _to_hwc_u8(left_images[i])
                right_image = (
                    _to_hwc_u8(right_images[i])
                    if right_images is not None
                    else left_image
                )

                x_obs = _reshape_site_matrix(
                    np.asarray(x_obs_arr[i]), policy.num_sites, 6
                )

                x_wrench = None
                if x_wrench_arr is not None:
                    x_wrench = _reshape_site_matrix(
                        np.asarray(x_wrench_arr[i]), policy.num_sites, 6
                    )

                head_pos = None
                if head_pos_arr is not None:
                    head_pos = np.asarray(head_pos_arr[i], dtype=np.float32).reshape(3)

                head_quat = None
                if head_quat_arr is not None:
                    head_quat = np.asarray(head_quat_arr[i], dtype=np.float32).reshape(
                        4
                    )

                out = policy.step(
                    ComplianceVLMInput(
                        time=float(i) * float(args.dt),
                        x_obs=x_obs,
                        x_wrench=x_wrench,
                        head_pos_world=head_pos,
                        head_quat_world_wxyz=head_quat,
                        left_image=left_image,
                        right_image=right_image,
                    )
                )

                pose_list.append(out.pose_command)
                cmd_list.append(out.command_matrix)
                status_list.append(out.status)

                if i % 20 == 0:
                    print(
                        f"[compliance_vlm] step={i} status={out.status} "
                        f"pose_norm={np.linalg.norm(out.pose_command):.4f}"
                    )
        else:
            for i in range(int(args.steps)):
                left_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                right_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                x_obs = np.zeros((policy.num_sites, 6), dtype=np.float32)
                x_wrench = np.zeros((policy.num_sites, 6), dtype=np.float32)

                out = policy.step(
                    ComplianceVLMInput(
                        time=float(i) * float(args.dt),
                        x_obs=x_obs,
                        x_wrench=x_wrench,
                        left_image=left_image,
                        right_image=right_image,
                    )
                )

                pose_list.append(out.pose_command)
                cmd_list.append(out.command_matrix)
                status_list.append(out.status)

                if i % 20 == 0:
                    print(
                        f"[compliance_vlm] dummy step={i} status={out.status} "
                        f"pose_norm={np.linalg.norm(out.pose_command):.4f}"
                    )
    finally:
        policy.close()

    if len(args.save) > 0:
        out_path = Path(args.save)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out_path,
            pose_command=np.asarray(pose_list, dtype=np.float32),
            command_matrix=np.asarray(cmd_list, dtype=np.float32),
            status=np.asarray(status_list),
        )
        print(f"[compliance_vlm] saved: {out_path}")


if __name__ == "__main__":
    main()
