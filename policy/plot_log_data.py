"""Plot motor logs from run_policy log_data.lz4.

Usage:
    python plot_log_data.py --log results/.../log_data.lz4
"""

from __future__ import annotations

import argparse
import math
import os
from typing import Any

import joblib
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _to_2d(arr: Any) -> np.ndarray | None:
    if arr is None:
        return None
    out = np.asarray(arr)
    if out.ndim != 2:
        return None
    if out.shape[0] == 0 or out.shape[1] == 0:
        return None
    return out.astype(np.float32, copy=False)


def _stack_from_obs_list(payload: dict[str, Any]) -> dict[str, Any]:
    """Fallback parser for legacy payloads that store obs objects directly."""
    if "obs_list" not in payload:
        return payload
    obs_list = payload.get("obs_list", [])
    action_list = payload.get("action_list", [])
    if not isinstance(obs_list, list):
        return payload

    fields = (
        "motor_pos",
        "motor_vel",
        "motor_acc",
        "motor_tor",
        "motor_cur",
        "motor_drive",
        "motor_pwm",
        "motor_vin",
        "motor_temp",
    )
    obs: dict[str, list[np.ndarray]] = {k: [] for k in fields}
    time = []
    for item in obs_list:
        t = getattr(item, "time", None)
        if t is not None:
            time.append(float(t))
        for key in fields:
            value = getattr(item, key, None)
            if value is None:
                continue
            obs[key].append(np.asarray(value, dtype=np.float32).reshape(-1))

    obs_out: dict[str, np.ndarray] = {}
    for key, seq in obs.items():
        if not seq:
            continue
        try:
            obs_out[key] = np.stack(seq).astype(np.float32)
        except ValueError:
            continue

    action = None
    if isinstance(action_list, list) and len(action_list) > 0:
        try:
            action = np.stack([np.asarray(a, dtype=np.float32) for a in action_list])
        except ValueError:
            action = None

    return {
        "time": np.asarray(time, dtype=np.float64),
        "action": action if action is not None else np.zeros((0, 0), dtype=np.float32),
        "obs": obs_out,
        "motor_names": payload.get("motor_names", []),
    }


def _plot_field(
    x: np.ndarray,
    y: np.ndarray,
    motor_names: list[str],
    title: str,
    save_path: str,
) -> None:
    n_motor = int(y.shape[1])
    n_cols = min(4, max(1, n_motor))
    n_rows = int(math.ceil(n_motor / n_cols))
    fig_w = max(12, 3.0 * n_cols)
    fig_h = max(3, 2.0 * n_rows)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h), sharex=True)
    axes_arr = np.atleast_1d(axes).reshape(-1)

    for idx in range(n_motor):
        ax = axes_arr[idx]
        ax.plot(x, y[:, idx], linewidth=1.0)
        if idx < len(motor_names):
            ax.set_title(motor_names[idx], fontsize=8)
        else:
            ax.set_title(f"motor_{idx:02d}", fontsize=8)
        ax.grid(True, alpha=0.25)

    for idx in range(n_motor, axes_arr.shape[0]):
        axes_arr[idx].axis("off")

    fig.suptitle(title)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.98))
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def _plot_action_vs_pos(
    x: np.ndarray,
    action: np.ndarray,
    motor_pos: np.ndarray,
    motor_names: list[str],
    save_path: str,
) -> None:
    n_motor = int(min(action.shape[1], motor_pos.shape[1]))
    if n_motor <= 0:
        return

    n_cols = min(4, max(1, n_motor))
    n_rows = int(math.ceil(n_motor / n_cols))
    fig_w = max(12, 3.0 * n_cols)
    fig_h = max(3, 2.0 * n_rows)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h), sharex=True)
    axes_arr = np.atleast_1d(axes).reshape(-1)

    for idx in range(n_motor):
        ax = axes_arr[idx]
        ax.plot(x, action[:, idx], linewidth=1.0, label="action")
        ax.plot(x, motor_pos[:, idx], linewidth=1.0, label="motor_pos")
        if idx < len(motor_names):
            ax.set_title(motor_names[idx], fontsize=8)
        else:
            ax.set_title(f"motor_{idx:02d}", fontsize=8)
        if idx == 0:
            ax.legend(fontsize=8)
        ax.grid(True, alpha=0.25)

    for idx in range(n_motor, axes_arr.shape[0]):
        axes_arr[idx].axis("off")

    fig.suptitle("action_vs_pos")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.98))
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def _plot_loop_dt(time: np.ndarray, save_path: str) -> None:
    if time.shape[0] < 2:
        return
    x = time[1:]
    y = np.diff(time)
    fig, ax = plt.subplots(1, 1, figsize=(10, 4), sharex=True)
    ax.plot(x, y, linewidth=1.0)
    ax.set_title("loop_dt")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("dt (s)")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot motor data from log_data.lz4")
    parser.add_argument("--log", type=str, required=True, help="Path to log_data.lz4")
    args = parser.parse_args()

    log_path = os.path.abspath(args.log)
    if not os.path.isfile(log_path):
        raise FileNotFoundError(f"log file not found: {log_path}")

    payload = joblib.load(log_path)
    if not isinstance(payload, dict):
        raise ValueError("Unsupported log file format: expected dict payload.")
    payload = _stack_from_obs_list(payload)

    obs = payload.get("obs", {})
    if not isinstance(obs, dict):
        raise ValueError("Unsupported log file format: payload['obs'] must be a dict.")

    time = np.asarray(payload.get("time", []), dtype=np.float64).reshape(-1)
    motor_names_raw = payload.get("motor_names", [])
    motor_names = [str(name) for name in motor_names_raw] if motor_names_raw else []
    output_dir = os.path.dirname(log_path)

    field_map = [
        ("action", payload.get("action")),
        ("motor_pos", obs.get("motor_pos")),
        ("motor_vel", obs.get("motor_vel")),
        ("motor_acc", obs.get("motor_acc")),
        ("motor_tor", obs.get("motor_tor")),
        ("motor_cur", obs.get("motor_cur")),
        ("motor_drive", obs.get("motor_drive")),
        ("motor_pwm", obs.get("motor_pwm")),
        ("motor_vin", obs.get("motor_vin")),
        ("motor_temp", obs.get("motor_temp")),
    ]

    num_plots = 0
    for field_name, raw in field_map:
        y = _to_2d(raw)
        if y is None:
            continue
        if time.shape[0] == y.shape[0]:
            x = time
        else:
            x = np.arange(y.shape[0], dtype=np.float64)

        if not motor_names or len(motor_names) != y.shape[1]:
            names = [f"motor_{i:02d}" for i in range(y.shape[1])]
        else:
            names = motor_names

        save_path = os.path.join(output_dir, f"{field_name}.png")
        _plot_field(x=x, y=y, motor_names=names, title=field_name, save_path=save_path)
        print(f"[plot_log_data] wrote {save_path}")
        num_plots += 1

    action = _to_2d(payload.get("action"))
    motor_pos = _to_2d(obs.get("motor_pos"))
    if action is not None and motor_pos is not None:
        n_samples = min(action.shape[0], motor_pos.shape[0])
        if n_samples > 0:
            action = action[:n_samples]
            motor_pos = motor_pos[:n_samples]
            if time.shape[0] >= n_samples:
                x = time[:n_samples]
            else:
                x = np.arange(n_samples, dtype=np.float64)
            if not motor_names or len(motor_names) < min(
                action.shape[1], motor_pos.shape[1]
            ):
                names = [
                    f"motor_{i:02d}"
                    for i in range(min(action.shape[1], motor_pos.shape[1]))
                ]
            else:
                names = motor_names
            save_path = os.path.join(output_dir, "action_vs_pos.png")
            _plot_action_vs_pos(
                x=x,
                action=action,
                motor_pos=motor_pos,
                motor_names=names,
                save_path=save_path,
            )
            print(f"[plot_log_data] wrote {save_path}")
            num_plots += 1

    if time.shape[0] >= 2:
        save_path = os.path.join(output_dir, "loop_dt.png")
        _plot_loop_dt(time=time, save_path=save_path)
        print(f"[plot_log_data] wrote {save_path}")
        num_plots += 1

    if num_plots == 0:
        raise RuntimeError("No plottable motor fields found in the log.")


if __name__ == "__main__":
    main()
