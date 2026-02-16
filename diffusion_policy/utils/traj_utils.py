"""Trajectory interpolation utilities extracted for standalone diffusion policy."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import numpy.typing as npt


def interpolate(
    p_start: npt.ArrayLike,
    p_end: npt.ArrayLike,
    duration: float,
    t: float,
    interp_type: str = "linear",
) -> npt.NDArray[np.float32]:
    """Interpolate between two points using linear or cosine blend."""
    start = np.asarray(p_start, dtype=np.float32)
    end = np.asarray(p_end, dtype=np.float32)
    if duration <= 1e-8:
        return end.astype(np.float32)

    alpha = float(np.clip(t / duration, 0.0, 1.0))
    if interp_type == "cosine":
        alpha = 0.5 - 0.5 * np.cos(np.pi * alpha)

    return (start + (end - start) * alpha).astype(np.float32)


def binary_search(arr: npt.NDArray[np.float32], t: float) -> int:
    """Return rightmost index i with arr[i] <= t."""
    low, high = 0, len(arr) - 1
    while low <= high:
        mid = (low + high) // 2
        if arr[mid] < t:
            low = mid + 1
        elif arr[mid] > t:
            high = mid - 1
        else:
            return mid
    return low - 1


def interpolate_action(
    t: float,
    time_arr: npt.NDArray[np.float32],
    action_arr: npt.NDArray[np.float32],
    interp_type: str = "linear",
) -> npt.NDArray[np.float32]:
    """Interpolate action at time t."""
    if t <= float(time_arr[0]):
        return action_arr[0]
    if t >= float(time_arr[-1]):
        return action_arr[-1]

    idx = binary_search(time_arr, t)
    idx = max(0, min(idx, len(time_arr) - 2))
    return interpolate(
        action_arr[idx],
        action_arr[idx + 1],
        float(time_arr[idx + 1] - time_arr[idx]),
        float(t - time_arr[idx]),
        interp_type,
    )


def get_action_traj(
    time_curr: float,
    action_curr: npt.ArrayLike,
    action_next: npt.ArrayLike,
    duration: float,
    control_dt: float,
    end_time: float = 0.0,
) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """Generate a dense trajectory from action_curr to action_next."""
    curr = np.asarray(action_curr, dtype=np.float32).reshape(-1)
    nxt = np.asarray(action_next, dtype=np.float32).reshape(-1)

    n_steps = max(int(duration / max(control_dt, 1e-6)), 1)
    action_time = np.linspace(
        0.0,
        float(duration),
        n_steps,
        endpoint=True,
        dtype=np.float32,
    )

    action_traj = np.zeros((len(action_time), curr.shape[0]), dtype=np.float32)
    for i, t in enumerate(action_time):
        if t < duration - end_time:
            pos = interpolate(curr, nxt, duration - end_time, float(t))
        else:
            pos = nxt
        action_traj[i] = pos

    action_time = action_time + float(time_curr)
    return action_time.astype(np.float32), action_traj.astype(np.float32)
