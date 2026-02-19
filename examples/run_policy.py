"""Single shared runner for all example policies."""

from __future__ import annotations

import argparse
import datetime as dt
import importlib
import json
import os
import pickle
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

import mujoco
import numpy as np

from examples.real_world import RealWorld
from examples.sim import (
    MuJoCoSim,
    build_clamped_torque_substep_control,
    build_site_force_applier,
)
from minimalist_compliance_control.utils import load_motor_params


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _as_f32_array(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    return np.asarray(value, dtype=np.float32).copy()


@dataclass
class Obs:
    """Typed observation object shared across all example policies."""

    time: float = 0.0
    motor_pos: np.ndarray | None = None
    motor_vel: np.ndarray | None = None
    motor_acc: np.ndarray | None = None
    motor_tor: np.ndarray | None = None
    qpos: np.ndarray | None = None
    qvel: np.ndarray | None = None
    motor_cur: np.ndarray | None = None
    motor_pwm: np.ndarray | None = None
    motor_vin: np.ndarray | None = None
    image: np.ndarray | None = None
    left_image: np.ndarray | None = None
    right_image: np.ndarray | None = None
    imu: Any = None
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any]) -> "Obs":
        known_keys = {
            "time",
            "motor_pos",
            "motor_vel",
            "motor_acc",
            "motor_tor",
            "qpos",
            "qvel",
            "motor_cur",
            "motor_pwm",
            "motor_vin",
            "image",
            "left_image",
            "right_image",
            "imu",
        }
        image = mapping.get("image")
        left_image = mapping.get("left_image")
        right_image = mapping.get("right_image")
        return cls(
            time=float(mapping.get("time", 0.0)),
            motor_pos=_as_f32_array(mapping.get("motor_pos")),
            motor_vel=_as_f32_array(mapping.get("motor_vel")),
            motor_acc=_as_f32_array(mapping.get("motor_acc")),
            motor_tor=_as_f32_array(mapping.get("motor_tor")),
            qpos=_as_f32_array(mapping.get("qpos")),
            qvel=_as_f32_array(mapping.get("qvel")),
            motor_cur=_as_f32_array(mapping.get("motor_cur")),
            motor_pwm=_as_f32_array(mapping.get("motor_pwm")),
            motor_vin=_as_f32_array(mapping.get("motor_vin")),
            image=(
                np.asarray(image, dtype=np.uint8).copy() if image is not None else None
            ),
            left_image=(
                np.asarray(left_image, dtype=np.uint8).copy()
                if left_image is not None
                else None
            ),
            right_image=(
                np.asarray(right_image, dtype=np.uint8).copy()
                if right_image is not None
                else None
            ),
            imu=mapping.get("imu"),
            extra={k: v for k, v in mapping.items() if k not in known_keys},
        )


def _import_module_any(module_name: str):
    last_exc: Exception | None = None
    for name in (f"examples.{module_name}", module_name):
        try:
            return importlib.import_module(name)
        except ModuleNotFoundError as exc:
            last_exc = exc
            continue
    if last_exc is not None:
        raise last_exc
    raise ModuleNotFoundError(module_name)


def _parse_args(argv: Sequence[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="Unified policy runner.")
    parser.add_argument(
        "--policy",
        choices=[
            "compliance",
            "compliance_model_based",
            "compliance_dp",
            "compliance_vlm",
        ],
        required=True,
        help="Policy to run.",
    )
    parser.add_argument(
        "--sim",
        choices=["mujoco", "real"],
        default="mujoco",
        help="Backend to run.",
    )
    if hasattr(argparse, "BooleanOptionalAction"):
        parser.add_argument(
            "--vis",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Enable MuJoCo interactive viewer (mujoco backend only).",
        )
    else:
        parser.add_argument(
            "--vis",
            dest="vis",
            action="store_true",
            default=True,
            help="Enable MuJoCo interactive viewer (mujoco backend only).",
        )
        parser.add_argument(
            "--no-vis",
            dest="vis",
            action="store_false",
            help="Disable MuJoCo interactive viewer (mujoco backend only).",
        )
    parser.add_argument(
        "--robot",
        choices=["toddlerbot", "leap"],
        required=True,
        help="Robot target.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        default=False,
        help="Enable compliance/wrench plotting where supported.",
    )
    raw = list(argv)
    if "--" in raw:
        split_idx = raw.index("--")
        args = parser.parse_args(raw[:split_idx])
        return args, raw[split_idx + 1 :]
    return parser.parse_known_args(raw)


def _run_compliance(args: argparse.Namespace) -> None:
    mod = _import_module_any("compliance")
    policy = mod.CompliancePolicy(
        robot=str(args.robot),
        sim=str(args.sim),
        vis=bool(args.vis),
        plot=bool(args.plot),
    )
    sim = _build_sim(policy, args)
    _run_tick_loop(policy, sim, args)


def _run_compliance_model_based(
    args: argparse.Namespace, remainder: Sequence[str]
) -> None:
    mod = _import_module_any("compliance_model_based")
    policy = mod.ModelBasedPolicy.from_argv(
        remainder,
        robot=str(args.robot),
        sim=str(args.sim),
        vis=bool(args.vis),
        plot=bool(args.plot),
    )
    sim = _build_sim(policy, args)
    _run_tick_loop(policy, sim, args)


def _run_compliance_dp(args: argparse.Namespace, remainder: Sequence[str]) -> None:
    mod = _import_module_any("compliance_dp")
    policy = mod.ComplianceDPPolicy.from_argv(
        remainder,
        robot=str(args.robot),
        sim=str(args.sim),
        vis=bool(args.vis),
        plot=bool(args.plot),
    )
    sim = _build_sim(policy, args)
    _run_tick_loop(policy, sim, args)


def _run_compliance_vlm(args: argparse.Namespace, remainder: Sequence[str]) -> None:
    mod = _import_module_any("compliance_vlm")
    policy = mod.ComplianceVLMPolicy.from_argv(
        remainder,
        robot=str(args.robot),
        sim=str(args.sim),
        vis=bool(args.vis),
        plot=bool(args.plot),
    )
    sim = _build_sim(policy, args)
    _run_tick_loop(policy, sim, args)


def _get_motor_config_paths(policy: Any) -> tuple[str, str, str | None] | None:
    if hasattr(policy, "motor_cfg_paths"):
        cfg = policy.motor_cfg_paths
        return (
            str(cfg.default_config_path),
            str(cfg.robot_config_path),
            str(cfg.motors_config_path) if cfg.motors_config_path is not None else None,
        )
    if hasattr(policy, "default_config_path") and hasattr(policy, "robot_config_path"):
        return (
            str(policy.default_config_path),
            str(policy.robot_config_path),
            str(policy.motors_config_path)
            if getattr(policy, "motors_config_path", None) is not None
            else None,
        )
    return None


def _build_sim(policy: Any, args: argparse.Namespace) -> Any:
    if str(args.sim) == "mujoco":
        impl = getattr(policy, "impl", policy)
        if hasattr(policy, "impl"):
            substep_control = impl.substep_control
            model = impl.model
            data = impl.data
        else:
            cfg_paths = _get_motor_config_paths(policy)
            if cfg_paths is None:
                raise ValueError("Missing motor config paths for simulation setup.")
            default_config_path, robot_config_path, motors_config_path = cfg_paths
            motor_params = load_motor_params(
                model=policy.model,
                default_config_path=default_config_path,
                robot_config_path=robot_config_path,
                motors_config_path=motors_config_path,
            )
            substep_control = build_clamped_torque_substep_control(
                qpos_adr=policy.qpos_adr,
                qvel_adr=policy.qvel_adr,
                motor_params=motor_params,
                target_motor_pos_getter=lambda: policy.target_motor_pos,
            )
            if hasattr(policy, "force_site_ids"):
                policy.site_force_applier = build_site_force_applier(
                    model=policy.model,
                    site_ids=policy.force_site_ids,
                )
            model = policy.model
            data = policy.data

        sim = MuJoCoSim(
            model=model,
            data=data,
            control_dt=policy.control_dt,
            sim_dt=float(model.opt.timestep),
            vis=bool(args.vis),
            substep_control=substep_control,
        )
        compliance_ref = getattr(impl.controller, "compliance_ref", None)
        if (
            compliance_ref is not None
            and getattr(compliance_ref, "default_qpos", None) is not None
        ):
            data.qpos[:] = np.asarray(compliance_ref.default_qpos, dtype=np.float32)
            mujoco.mj_forward(model, data)
        return sim

    if hasattr(policy, "impl"):
        raise ValueError("compliance_model_based currently supports only --sim mujoco")
    cfg_paths = _get_motor_config_paths(policy)
    if cfg_paths is None:
        raise ValueError("Missing motor config paths for real backend setup.")
    default_config_path, robot_config_path, motors_config_path = cfg_paths
    motor_ordering: list[str] | None = None
    if hasattr(policy, "model"):
        model = policy.model
        names = [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            for i in range(int(model.nu))
        ]
        if all(name is not None for name in names):
            motor_ordering = [str(name) for name in names]
    return RealWorld(
        robot=str(args.robot),
        control_dt=policy.control_dt,
        default_config_path=default_config_path,
        robot_config_path=robot_config_path,
        motors_config_path=motors_config_path,
        motor_ordering=motor_ordering,
    )


class _ResultRecorder:
    def __init__(self, *, enabled: bool, robot: str, policy: str, sim: str) -> None:
        self.enabled = bool(enabled)
        self.records: list[dict[str, Any]] = []
        self.root_dir: str | None = None
        self._metadata = {"robot": robot, "policy": policy, "sim": sim}
        if self.enabled:
            stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.root_dir = os.path.join(
                _repo_root(), "results", f"{robot}_{policy}_{stamp}"
            )
            os.makedirs(self.root_dir, exist_ok=True)
            print(f"[plot] runner dump path: {self.root_dir}")

    def append(
        self,
        *,
        obs: Obs,
        control_inputs: dict[str, float],
        action: np.ndarray,
    ) -> None:
        if not self.enabled:
            return
        clean_ctrl = {k: float(v) for k, v in control_inputs.items()}
        self.records.append(
            {
                "obs": asdict(obs),
                "control_inputs": clean_ctrl,
                "action": np.asarray(action, dtype=np.float32).copy(),
            }
        )

    def close(self) -> None:
        if not self.enabled or self.root_dir is None:
            return
        dump_path = os.path.join(self.root_dir, "runner_dump.pkl")
        meta_path = os.path.join(self.root_dir, "runner_meta.json")
        with open(dump_path, "wb") as f:
            pickle.dump(self.records, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    **self._metadata,
                    "num_steps": len(self.records),
                    "dump_file": os.path.basename(dump_path),
                },
                f,
                indent=2,
            )
        print(f"[plot] runner dump written: {dump_path}")


def _run_tick_loop(policy: Any, sim: Any, args: argparse.Namespace) -> None:
    next_tick = time.monotonic()
    recorder = _ResultRecorder(
        enabled=bool(args.plot),
        robot=str(args.robot),
        policy=str(args.policy),
        sim=str(args.sim),
    )
    try:
        while True:
            if bool(getattr(policy, "done", False)):
                break
            obs_raw = sim.get_observation()
            obs = Obs.from_mapping(obs_raw)
            _control_inputs, action = policy.step(obs, sim)
            control_inputs = dict(_control_inputs)
            recorder.append(
                obs=obs,
                control_inputs=control_inputs,
                action=np.asarray(action, dtype=np.float32),
            )
            if bool(getattr(policy, "done", False)):
                break
            sim.set_motor_target(action)
            sim.step()
            if not sim.sync():
                break
            next_tick += float(policy.control_dt)
            sleep_s = next_tick - time.monotonic()
            if sleep_s > 0.0:
                time.sleep(sleep_s)
            else:
                next_tick = time.monotonic()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            policy.close()
        finally:
            recorder.close()
            sim.close()


def main(argv: Sequence[str] | None = None) -> None:
    args, remainder = _parse_args(sys.argv[1:] if argv is None else argv)

    if args.policy == "compliance":
        _run_compliance(args)
    elif args.policy == "compliance_model_based":
        _run_compliance_model_based(args, remainder)
    elif args.policy == "compliance_dp":
        _run_compliance_dp(args, remainder)
    elif args.policy == "compliance_vlm":
        _run_compliance_vlm(args, remainder)
    else:
        raise ValueError(f"Unsupported policy: {args.policy}")


if __name__ == "__main__":
    main()
