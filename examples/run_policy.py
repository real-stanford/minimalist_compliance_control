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
from typing import Any, Sequence

import numpy as np

from minimalist_compliance_control.real_world import RealWorld
from minimalist_compliance_control.sim import (
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


class Obs:
    """Typed observation object shared across all example policies."""

    def __init__(
        self,
        *,
        time_s: float = 0.0,
        motor_pos: np.ndarray | None = None,
        motor_vel: np.ndarray | None = None,
        motor_acc: np.ndarray | None = None,
        motor_tor: np.ndarray | None = None,
        qpos: np.ndarray | None = None,
        qvel: np.ndarray | None = None,
        motor_cur: np.ndarray | None = None,
        motor_pwm: np.ndarray | None = None,
        motor_vin: np.ndarray | None = None,
        image: np.ndarray | None = None,
        left_image: np.ndarray | None = None,
        right_image: np.ndarray | None = None,
        imu: Any = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        self.time = float(time_s)
        self.motor_pos = motor_pos
        self.motor_vel = motor_vel
        self.motor_acc = motor_acc
        self.motor_tor = motor_tor
        self.qpos = qpos
        self.qvel = qvel
        self.motor_cur = motor_cur
        self.motor_pwm = motor_pwm
        self.motor_vin = motor_vin
        self.image = image
        self.left_image = left_image
        self.right_image = right_image
        self.imu = imu
        self.extra = extra if extra is not None else {}

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
            time_s=float(mapping.get("time", 0.0)),
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

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    def __contains__(self, key: str) -> bool:
        try:
            self[key]
            return True
        except KeyError:
            return False

    def __getitem__(self, key: str) -> Any:
        if key == "time":
            return self.time
        if key == "motor_pos" and self.motor_pos is not None:
            return self.motor_pos
        if key == "motor_vel" and self.motor_vel is not None:
            return self.motor_vel
        if key == "motor_acc" and self.motor_acc is not None:
            return self.motor_acc
        if key == "motor_tor" and self.motor_tor is not None:
            return self.motor_tor
        if key == "qpos" and self.qpos is not None:
            return self.qpos
        if key == "qvel" and self.qvel is not None:
            return self.qvel
        if key == "motor_cur" and self.motor_cur is not None:
            return self.motor_cur
        if key == "motor_pwm" and self.motor_pwm is not None:
            return self.motor_pwm
        if key == "motor_vin" and self.motor_vin is not None:
            return self.motor_vin
        if key == "image" and self.image is not None:
            return self.image
        if key == "left_image" and self.left_image is not None:
            return self.left_image
        if key == "right_image" and self.right_image is not None:
            return self.right_image
        if key == "imu" and self.imu is not None:
            return self.imu
        if key in self.extra:
            return self.extra[key]
        raise KeyError(key)

    def to_dump_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "time": float(self.time),
            "motor_pos": self.motor_pos,
            "motor_vel": self.motor_vel,
            "motor_acc": self.motor_acc,
            "motor_tor": self.motor_tor,
            "qpos": self.qpos,
            "qvel": self.qvel,
            "motor_cur": self.motor_cur,
            "motor_pwm": self.motor_pwm,
            "motor_vin": self.motor_vin,
            "imu": self.imu,
        }
        out.update(self.extra)
        return out


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
    sim = _build_compliance_sim(policy, args)
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
    sim = _build_model_based_sim(policy, args)
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
    sim = _build_dp_or_vlm_sim(policy, args)
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
    sim = _build_dp_or_vlm_sim(policy, args)
    _run_tick_loop(policy, sim, args)


def _build_compliance_sim(policy: Any, args: argparse.Namespace) -> Any:
    if str(args.sim) == "mujoco":
        motor_params = load_motor_params(
            model=policy.model,
            default_config_path=policy.motor_cfg_paths.default_config_path,
            robot_config_path=policy.motor_cfg_paths.robot_config_path,
            motors_config_path=policy.motor_cfg_paths.motors_config_path,
        )
        torque_substep_control = build_clamped_torque_substep_control(
            qpos_adr=policy.qpos_adr,
            qvel_adr=policy.qvel_adr,
            motor_params=motor_params,
            target_motor_pos_getter=lambda: policy.target_motor_pos,
        )
        policy.site_force_applier = build_site_force_applier(
            model=policy.model,
            site_ids=policy.force_site_ids,
        )
        sim = MuJoCoSim(
            policy.controller.wrench_sim,
            control_dt=policy.control_dt,
            sim_dt=float(policy.model.opt.timestep),
            vis=bool(args.vis),
            substep_control=torque_substep_control,
        )
        policy.data.qpos[:] = policy.controller.compliance_ref.default_qpos.copy()
        policy.controller.wrench_sim.forward()
        return sim
    return RealWorld(
        policy.controller.wrench_sim,
        control_dt=policy.control_dt,
        default_config_path=policy.motor_cfg_paths.default_config_path,
        robot_config_path=policy.motor_cfg_paths.robot_config_path,
        motors_config_path=policy.motor_cfg_paths.motors_config_path,
        vis=bool(args.vis),
    )


def _build_dp_or_vlm_sim(policy: Any, args: argparse.Namespace) -> Any:
    if str(args.sim) == "mujoco":
        motor_params = load_motor_params(
            model=policy.model,
            default_config_path=policy.default_config_path,
            robot_config_path=policy.robot_config_path,
            motors_config_path=policy.motors_config_path,
        )
        torque_substep_control = build_clamped_torque_substep_control(
            qpos_adr=policy.qpos_adr,
            qvel_adr=policy.qvel_adr,
            motor_params=motor_params,
            target_motor_pos_getter=lambda: policy.target_motor_pos,
        )
        sim = MuJoCoSim(
            policy.controller.wrench_sim,
            control_dt=policy.control_dt,
            sim_dt=float(policy.model.opt.timestep),
            vis=bool(args.vis),
            substep_control=torque_substep_control,
        )
        policy.data.qpos[:] = policy.controller.compliance_ref.default_qpos.copy()
        policy.controller.wrench_sim.forward()
        return sim
    return RealWorld(
        policy.controller.wrench_sim,
        control_dt=policy.control_dt,
        default_config_path=policy.default_config_path,
        robot_config_path=policy.robot_config_path,
        motors_config_path=policy.motors_config_path,
        vis=bool(args.vis),
    )


def _build_model_based_sim(policy: Any, args: argparse.Namespace) -> Any:
    if str(args.sim) != "mujoco":
        raise ValueError("compliance_model_based currently supports only --sim mujoco")
    impl = policy.impl
    return MuJoCoSim(
        impl.controller.wrench_sim,
        control_dt=policy.control_dt,
        sim_dt=float(impl.model.opt.timestep),
        vis=bool(args.vis),
        substep_control=impl.substep_control,
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
                "obs": obs.to_dump_dict(),
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
