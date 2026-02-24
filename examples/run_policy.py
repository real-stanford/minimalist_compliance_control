"""Policy execution framework with visualization and logging capabilities.

Refactored to match the old run-policy structure while using local minimalist
compliance policies and backends.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import gin
import mujoco
import numpy as np
import yaml
from tqdm import tqdm

from examples.base_sim import Obs
from examples.sim import MuJoCoSim, build_site_force_applier
from minimalist_compliance_control.controller import ControllerConfig


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


def _deep_update(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _merge_motor_config(
    default_config_path: str,
    robot_config_path: str,
    motor_config_path: str | None,
) -> dict[str, Any]:
    config = _load_yaml_dict(default_config_path)
    _deep_update(config, _load_yaml_dict(robot_config_path))
    if motor_config_path:
        _deep_update(config, _load_yaml_dict(motor_config_path))
    return config


@gin.configurable
@dataclass
class MotorConfigPaths:
    default_config_path: Optional[str] = None
    robot_config_path: Optional[str] = None
    motor_config_path: Optional[str] = None


def _build_sim(
    args: argparse.Namespace,
    control_dt: float,
    xml_path: str,
    merged_config: dict[str, Any],
) -> Any:
    if args.sim == "mujoco":
        model = mujoco.MjModel.from_xml_path(xml_path)
        data = mujoco.MjData(model)
        if args.robot in {"arx", "g1"}:
            custom_pd = False
        else:
            custom_pd = True

        return MuJoCoSim(
            model=model,
            data=data,
            control_dt=control_dt,
            sim_dt=float(model.opt.timestep),
            vis=args.vis != "none",
            custom_pd=custom_pd,
            merged_config=merged_config,
        )
    else:
        from examples.real_world import RealWorld

        return RealWorld(
            robot=str(args.robot),
            control_dt=control_dt,
            xml_path=str(xml_path),
            merged_config=merged_config,
        )


class ResultRecorder:
    def __init__(self, *, enabled: bool, robot: str, policy: str, sim: str) -> None:
        self.enabled = bool(enabled)
        self.num_steps = 0
        self.root_dir: str | None = None
        self._metadata = {"robot": robot, "policy": policy, "sim": sim}
        if self.enabled:
            stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.root_dir = os.path.join(
                _repo_root(), "results", f"{robot}_{policy}_{sim}_{stamp}"
            )
            os.makedirs(self.root_dir, exist_ok=True)
            print(f"[run_policy] dump path: {self.root_dir}")

    def append(self, *, obs: Obs, action: np.ndarray) -> None:
        del obs, action
        if not self.enabled:
            return
        self.num_steps += 1

    def close(self) -> None:
        if not self.enabled or self.root_dir is None:
            return
        meta_path = os.path.join(self.root_dir, "runner_meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    **self._metadata,
                    "num_steps": int(self.num_steps),
                },
                f,
                indent=2,
            )
        print(f"[run_policy] meta written: {meta_path}")


def run_policy(sim: Any, robot: str, policy: Any) -> None:
    control_dt = float(getattr(sim, "control_dt", 0.02))
    next_tick = time.monotonic()
    start_time: float | None = None
    step_idx = 0
    p_bar = tqdm(total=float("inf"), desc="Running policy", unit="step")

    recorder = ResultRecorder(
        enabled=True,
        robot=str(robot),
        policy=str(getattr(policy, "name", type(policy).__name__)),
        sim=str(getattr(sim, "name", "unknown")),
    )

    try:
        while True:
            if bool(getattr(policy, "done", False)):
                break

            obs = sim.get_observation()
            if start_time is None:
                start_time = float(obs.time)
            obs.time -= start_time
            action = policy.step(obs, sim)
            action_arr = np.asarray(action, dtype=np.float32)

            recorder.append(obs=obs, action=action_arr)

            step_idx += 1
            p_bar_steps = int(1 / policy.control_dt)
            if step_idx % p_bar_steps == 0:
                p_bar.update(p_bar_steps)

            if bool(getattr(policy, "done", False)):
                break

            sim.set_motor_target(action_arr)
            sim.step()
            if not sim.sync():
                break

            next_tick += control_dt
            sleep_s = next_tick - time.monotonic()
            if sleep_s > 0.0:
                time.sleep(sleep_s)
            else:
                next_tick = time.monotonic()
    except KeyboardInterrupt:
        pass
    finally:
        p_bar.close()
        exp_dir = recorder.root_dir or ""
        try:
            try:
                policy.close(exp_folder_path=exp_dir)
            except TypeError:
                policy.close()
        finally:
            recorder.close()
            sim.close()


def _parse_args(args: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified policy runner")
    parser.add_argument(
        "--robot", type=str, required=True, choices=["toddlerbot", "leap", "arx", "g1"]
    )
    parser.add_argument("--sim", type=str, default="mujoco", choices=["mujoco", "real"])
    parser.add_argument(
        "--vis", type=str, default="view", choices=["render", "view", "none"]
    )
    parser.add_argument(
        "--policy",
        type=str,
        default="compliance",
        choices=[
            "compliance",
            "compliance_model_based",
            "compliance_dp",
            "compliance_vlm",
        ],
    )
    parser.add_argument("--ip", type=str, default="")
    parser.add_argument("--ckpt", type=str, default="")
    parser.add_argument("--object", type=str, default="black ink. whiteboard. vase")
    parser.add_argument("--site-names", type=str, default="")
    parser.add_argument(
        "--replay",
        type=str,
        default="",
        help="Path to replay trajectory folder (or trajectory .lz4 file) for compliance_vlm.",
    )

    return parser.parse_args(args=args)


def main(args: Sequence[str] | None = None) -> None:
    parsed = _parse_args(args)
    if str(parsed.policy) == "compliance_vlm" and str(parsed.robot) in {
        "toddlerbot",
        "leap",
    }:
        gin_file = f"{parsed.robot}_vlm.gin"
    elif str(parsed.policy) == "compliance_model_based":
        if str(parsed.robot) == "leap":
            gin_file = "leap_model_based.gin"
        elif str(parsed.robot) == "toddlerbot":
            gin_file = "toddlerbot_model_based.gin"
        else:
            gin_file = f"{parsed.robot}.gin"
    elif str(parsed.policy) == "compliance_dp" and str(parsed.robot) == "toddlerbot":
        gin_file = "toddlerbot_dp.gin"
    else:
        gin_file = f"{parsed.robot}.gin"
    gin_path = os.path.join(_repo_root(), "config", gin_file)
    gin.parse_config_file(gin_path, skip_unknown=True)
    motor_cfg_paths = MotorConfigPaths()
    if (
        motor_cfg_paths.default_config_path is None
        or motor_cfg_paths.robot_config_path is None
    ):
        raise ValueError(f"MotorConfigPaths missing in gin file: {gin_path}")

    default_cfg = _resolve_repo_path(str(motor_cfg_paths.default_config_path))
    robot_cfg = _resolve_repo_path(str(motor_cfg_paths.robot_config_path))
    motors_cfg = (
        _resolve_repo_path(str(motor_cfg_paths.motor_config_path))
        if motor_cfg_paths.motor_config_path is not None
        else None
    )
    merged_config = _merge_motor_config(default_cfg, robot_cfg, motors_cfg)
    controller_cfg = ControllerConfig()
    if controller_cfg.xml_path is None:
        raise ValueError(f"ControllerConfig.xml_path missing in gin file: {gin_path}")
    xml_path_raw = str(controller_cfg.xml_path)
    xml_path = _resolve_repo_path(xml_path_raw)

    sim = _build_sim(
        parsed,
        control_dt=float(getattr(parsed, "control_dt", 0.02)),
        xml_path=xml_path,
        merged_config=merged_config,
    )

    if parsed.sim == "mujoco":
        init_motor_pos = sim.get_observation().motor_pos
    elif parsed.sim == "real":
        init_motor_pos = sim.get_observation(retries=-1).motor_pos

    init_motor_pos = np.asarray(sim.get_observation().motor_pos, dtype=np.float32)

    policy_name = str(parsed.policy)
    policy_kwargs: dict[str, Any] = {
        "name": str(parsed.policy),
        "robot": str(parsed.robot),
        "init_motor_pos": init_motor_pos,
    }
    vis_enabled = bool(parsed.vis != "none")
    if policy_name == "compliance":
        from examples.compliance import CompliancePolicy

        policy = CompliancePolicy(**policy_kwargs)
    elif policy_name == "compliance_model_based":
        from examples.compliance_model_based import ModelBasedPolicy

        policy = ModelBasedPolicy(
            robot=str(parsed.robot),
            sim=str(parsed.sim),
            vis=vis_enabled,
        )
    elif policy_name == "compliance_dp":
        from examples.compliance_dp import ComplianceDPPolicy

        policy = ComplianceDPPolicy(
            **policy_kwargs,
            ckpt=str(parsed.ckpt),
        )
    elif policy_name == "compliance_vlm":
        from examples.compliance_vlm import ComplianceVLMPolicy

        replay_path = str(parsed.replay).strip()
        if replay_path:
            replay_path = _resolve_repo_path(replay_path)
        policy = ComplianceVLMPolicy(
            **policy_kwargs,
            object=str(parsed.object),
            site_names=str(parsed.site_names),
            replay=replay_path,
        )
    else:
        raise ValueError(f"Unsupported policy: {parsed.policy}")

    if parsed.sim == "mujoco" and hasattr(policy, "force_site_ids"):
        policy.site_force_applier = build_site_force_applier(
            model=sim.model,
            site_ids=np.asarray(policy.force_site_ids, dtype=np.int32),
        )

    run_policy(sim=sim, robot=str(parsed.robot), policy=policy)


if __name__ == "__main__":
    main()
