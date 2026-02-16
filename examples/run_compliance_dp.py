"""Replay/dummy runner for standalone compliance_dp."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from diffusion_policy.compliance_dp import ComplianceDPInput, StandaloneComplianceDP


def _find_key(npz_obj, candidates: list[str]) -> str:
    for key in candidates:
        if key in npz_obj:
            return key
    raise KeyError(f"None of keys found in replay file: {candidates}")


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone compliance_dp runner")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to DP checkpoint")
    parser.add_argument("--num-sites", type=int, default=2, help="Number of EE sites")
    parser.add_argument("--dt", type=float, default=0.02, help="Control dt")
    parser.add_argument("--replay-npz", type=str, default="", help="Replay npz path")
    parser.add_argument(
        "--steps", type=int, default=300, help="Dummy steps if no replay"
    )
    parser.add_argument(
        "--save", type=str, default="", help="Output npz for pose/command"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    policy = StandaloneComplianceDP(
        ckpt_path=args.ckpt,
        num_sites=int(args.num_sites),
        control_dt=float(args.dt),
    )

    pose_list = []
    cmd_list = []
    status_list = []

    try:
        if len(args.replay_npz) > 0:
            replay = np.load(args.replay_npz)
            image_key = _find_key(replay, ["image", "images", "rgb", "camera"])
            x_obs_key = _find_key(replay, ["x_obs", "pose", "ee_pose"])
            x_wrench_key = None
            for k in ["x_wrench", "wrench", "wrenches"]:
                if k in replay:
                    x_wrench_key = k
                    break
            motor_key = None
            for k in ["motor_pos", "obs_motor_pos", "qpos"]:
                if k in replay:
                    motor_key = k
                    break

            images = replay[image_key]
            x_obs_arr = replay[x_obs_key]
            x_wrench_arr = replay[x_wrench_key] if x_wrench_key else None
            motor_arr = replay[motor_key] if motor_key else None

            T = images.shape[0]
            if x_obs_arr.shape[0] < T:
                T = x_obs_arr.shape[0]
            if x_wrench_arr is not None and x_wrench_arr.shape[0] < T:
                T = x_wrench_arr.shape[0]
            if motor_arr is not None and motor_arr.shape[0] < T:
                T = motor_arr.shape[0]

            for i in range(T):
                image = _to_hwc_u8(images[i])
                x_obs = np.asarray(x_obs_arr[i], dtype=np.float32)
                if x_obs.ndim == 1 and x_obs.size == policy.num_sites * 6:
                    x_obs = x_obs.reshape(policy.num_sites, 6)

                x_wrench = None
                if x_wrench_arr is not None:
                    x_wrench = np.asarray(x_wrench_arr[i], dtype=np.float32)
                    if x_wrench.ndim == 1 and x_wrench.size == policy.num_sites * 6:
                        x_wrench = x_wrench.reshape(policy.num_sites, 6)

                motor_pos = None
                if motor_arr is not None:
                    motor_pos = np.asarray(motor_arr[i], dtype=np.float32)

                out = policy.step(
                    ComplianceDPInput(
                        time=float(i) * float(args.dt),
                        image=image,
                        x_obs=x_obs,
                        x_wrench=x_wrench,
                        motor_pos=motor_pos,
                    )
                )
                pose_list.append(out.pose_command)
                cmd_list.append(out.command_matrix)
                status_list.append(out.status)

                if i % 20 == 0:
                    print(
                        f"[compliance_dp] step={i} status={out.status} "
                        f"pose_norm={np.linalg.norm(out.pose_command):.4f}"
                    )
        else:
            for i in range(int(args.steps)):
                image = np.random.randint(0, 255, (96, 96, 3), dtype=np.uint8)
                x_obs = np.zeros((policy.num_sites, 6), dtype=np.float32)
                out = policy.step(
                    ComplianceDPInput(
                        time=float(i) * float(args.dt),
                        image=image,
                        x_obs=x_obs,
                    )
                )
                pose_list.append(out.pose_command)
                cmd_list.append(out.command_matrix)
                status_list.append(out.status)

                if i % 20 == 0:
                    print(
                        f"[compliance_dp] dummy step={i} status={out.status} "
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
        print(f"[compliance_dp] saved: {out_path}")


if __name__ == "__main__":
    main()
