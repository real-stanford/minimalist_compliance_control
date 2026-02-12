"""Minimal normalization utilities required by diffusion inference."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def normalize_data(data: npt.ArrayLike, stats: dict) -> npt.NDArray[np.float32]:
    """Normalize features to [-1, 1] using per-dimension min/max stats."""
    arr = np.asarray(data, dtype=np.float32)
    min_v = np.asarray(stats["min"], dtype=np.float32)
    max_v = np.asarray(stats["max"], dtype=np.float32)

    range_v = max_v - min_v
    zero_mask = range_v == 0

    norm = np.zeros_like(arr, dtype=np.float32)
    valid = ~zero_mask
    norm[..., valid] = (arr[..., valid] - min_v[valid]) / range_v[valid]
    norm[..., valid] = norm[..., valid] * 2.0 - 1.0
    return norm.astype(np.float32)


def unnormalize_data(ndata: npt.ArrayLike, stats: dict) -> npt.NDArray[np.float32]:
    """Undo [-1, 1] normalization using per-dimension min/max stats."""
    arr = np.asarray(ndata, dtype=np.float32)
    min_v = np.asarray(stats["min"], dtype=np.float32)
    max_v = np.asarray(stats["max"], dtype=np.float32)

    data_01 = (arr + 1.0) / 2.0
    data = data_01 * (max_v - min_v) + min_v
    return data.astype(np.float32)
