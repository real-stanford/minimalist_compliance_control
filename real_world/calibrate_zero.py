"""Tool for calibrating toddlerbot motor zero positions."""

from __future__ import annotations

import argparse
import os
import platform
import xml.etree.ElementTree as ET
from types import ModuleType
from typing import Any

import numpy as np
import yaml

from minimalist_compliance_control.utils import load_merged_motor_config


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _resolve_repo_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(_repo_root(), path)


def _load_yaml_dict(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


def _select_robot_xml_path(robot: str) -> str:
    candidates = [
        _resolve_repo_path(f"descriptions/{robot}/{robot}_fixed.xml"),
        _resolve_repo_path(f"descriptions/{robot}/{robot}.xml"),
        _resolve_repo_path(f"descriptions/{robot}/scene_fixed.xml"),
        _resolve_repo_path(f"descriptions/{robot}/scene.xml"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"No XML found for robot {robot}. Checked: {candidates}")


def _import_dynamixel_cpp() -> ModuleType:
    try:
        import dynamixel_cpp  # type: ignore

        return dynamixel_cpp  # type: ignore[return-value]
    except ImportError as exc:
        print(f"Failed to import dynamixel_cpp: {exc}")
        raise


def _build_motor_ordering(
    merged_config: dict[str, Any],
    xml_path: str,
) -> tuple[list[str], list[float], list[float], list[float]]:
    motors_cfg = merged_config.get("motors", {})
    if not isinstance(motors_cfg, dict) or not motors_cfg:
        raise ValueError("Merged motor config is empty.")

    robot_root = ET.parse(xml_path).getroot()
    motor_ordering: list[str] = []
    for motor_name in motors_cfg:
        if robot_root.find(f".//joint[@name='{motor_name}']") is not None:
            motor_ordering.append(motor_name)
    if not motor_ordering:
        raise RuntimeError(f"No motors found in XML joints: {xml_path}")

    motor_kp = [float(motors_cfg[name].get("kp", 0.0)) for name in motor_ordering]
    motor_kd = [float(motors_cfg[name].get("kd", 0.0)) for name in motor_ordering]
    motor_ki = [float(motors_cfg[name].get("ki", 0.0)) for name in motor_ordering]
    return motor_ordering, motor_kp, motor_kd, motor_ki


def _parse_parts(parts: str) -> list[str]:
    cleaned = parts.replace(",", " ").strip()
    if not cleaned or cleaned == "all":
        return ["all"]
    return [token for token in cleaned.split() if token]


def _build_motor_mask(parts: list[str], all_motor_ids: list[int]) -> set[int]:
    all_parts: dict[str, list[int]] = {
        "left_arm": [16, 17, 18, 19, 20, 21, 22],
        "right_arm": [23, 24, 25, 26, 27, 28, 29],
        "gripper": [30, 31],
        "waist": [2, 3],
        "hip": [2, 3, 4, 5, 6, 10, 11, 12],
        "knee": [7, 13],
        "left_ankle": [8, 9],
        "right_ankle": [14, 15],
        "neck": [0, 1],
    }
    if "all" in parts:
        return set(all_motor_ids)

    mask: set[int] = set()
    for part in parts:
        if part not in all_parts:
            valid = ", ".join(sorted(all_parts))
            raise ValueError(f"Invalid part: {part}. Valid parts: {valid}, all")
        mask.update(all_parts[part])
    return mask


def calibrate_zero(robot: str, parts: list[str]) -> None:
    if "toddlerbot" not in robot.lower():
        raise ValueError("This calibration script is toddlerbot-specific.")

    default_cfg = _resolve_repo_path("descriptions/default.yml")
    robot_cfg = _resolve_repo_path(f"descriptions/{robot}/robot.yml")
    motors_cfg_path = _resolve_repo_path(f"descriptions/{robot}/motors.yml")
    if not os.path.exists(robot_cfg):
        raise FileNotFoundError(f"Robot config not found: {robot_cfg}")
    merged_config = load_merged_motor_config(
        default_path=default_cfg,
        robot_path=robot_cfg,
        motors_path=motors_cfg_path if os.path.exists(motors_cfg_path) else None,
    )
    xml_path = _select_robot_xml_path(robot)
    motor_ordering, motor_kp, motor_kd, motor_ki = _build_motor_ordering(
        merged_config, xml_path
    )

    while True:
        response = input("Have you installed the calibration parts? (y/n) > ").strip()
        if not response:
            continue
        if response[0].lower() == "y":
            break
        if response[0].lower() == "n":
            return
        print("Please answer 'yes' or 'no'.")

    dynamixel_cpp = _import_dynamixel_cpp()
    port_pattern = (
        r"cu\.usbserial-.*"
        if platform.system() == "Darwin"
        else r"tty(?:CH9344)?USB[0-9]+"
    )
    n_motor = len(motor_ordering)
    controllers = dynamixel_cpp.create_controllers(
        port_pattern,
        motor_kp,
        motor_kd,
        motor_ki,
        np.zeros(n_motor, dtype=np.float32),
        ["extended_position"] * n_motor,
        2_000_000,
        1,
    )

    try:
        dynamixel_cpp.initialize(controllers)
        motor_ids_dict = dynamixel_cpp.get_motor_ids(controllers)
        motor_state = dynamixel_cpp.get_motor_states(controllers, -1)
        if len(motor_state) == 0:
            raise RuntimeError("No motors found. Please check the connections.")

        motor_ids_flat: list[int] = []
        motor_pos_flat: list[float] = []
        for key in sorted(motor_ids_dict.keys()):
            motor_ids_flat.extend(motor_ids_dict[key])
            motor_pos_flat.extend(motor_state[key]["pos"])

        motor_ids_arr = np.asarray(motor_ids_flat, dtype=np.int32)
        motor_pos_arr = np.asarray(motor_pos_flat, dtype=np.float32)
        if motor_ids_arr.size != n_motor or motor_pos_arr.size != n_motor:
            raise RuntimeError(
                f"Motor count mismatch. expected {n_motor}, got ids={motor_ids_arr.size}, pos={motor_pos_arr.size}"
            )

        if np.array_equal(np.sort(motor_ids_arr), np.arange(n_motor, dtype=np.int32)):
            order_idx = np.argsort(motor_ids_arr)
            motor_ids_ordered = motor_ids_arr[order_idx].tolist()
            zero_pos = motor_pos_arr[order_idx]
        else:
            print(
                "[calibrate_zero] Motor IDs are not a 0..N-1 permutation; using controller order."
            )
            motor_ids_ordered = motor_ids_arr.tolist()
            zero_pos = motor_pos_arr
    finally:
        dynamixel_cpp.close(controllers)

    motor_mask = _build_motor_mask(parts, motor_ids_ordered)

    if os.path.exists(motors_cfg_path):
        motor_config = _load_yaml_dict(motors_cfg_path)
        if "motors" not in motor_config or not isinstance(motor_config["motors"], dict):
            motor_config["motors"] = {}
    else:
        motor_config = {"motors": {}}

    selected_names = [
        name
        for motor_id, name in zip(motor_ids_ordered, motor_ordering)
        if motor_id in motor_mask
    ]
    if not selected_names:
        raise RuntimeError(f"No motors matched parts: {parts}")
    max_name_len = max(len(name) for name in selected_names)

    print(f"Updating motor zero point at {motors_cfg_path}...")
    for motor_id, name, pos in zip(motor_ids_ordered, motor_ordering, zero_pos):
        if motor_id not in motor_mask:
            continue
        if name not in motor_config["motors"]:
            motor_config["motors"][name] = {}
            motor_config["motors"][name]["zero_pos"] = 0.0

        prev = motor_config["motors"][name].get("zero_pos", 0.0)
        new_pos = round(float(pos), 6)
        print(f"{name:<{max_name_len}} : {prev:<10} -> {new_pos}")
        motor_config["motors"][name]["zero_pos"] = new_pos

    with open(motors_cfg_path, "w", encoding="utf-8") as f:
        yaml.dump(motor_config, f, indent=4, default_flow_style=False, sort_keys=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run toddlerbot zero-point calibration."
    )
    parser.add_argument(
        "--robot",
        type=str,
        default="toddlerbot_2xm",
        help="Toddlerbot description folder under descriptions/.",
    )
    parser.add_argument(
        "--parts",
        type=str,
        default="all",
        help="Use 'all' or a subset of [left_arm, right_arm, gripper, waist, hip, knee, left_ankle, right_ankle, neck].",
    )
    args = parser.parse_args()

    calibrate_zero(robot=args.robot, parts=_parse_parts(args.parts))


if __name__ == "__main__":
    main()
