"""Minimal math utilities used by standalone VLM modules."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import numpy.typing as npt


ArrayLike = npt.ArrayLike


def symmetrize(matrix: ArrayLike) -> npt.NDArray[np.float32]:
    arr = np.asarray(matrix, dtype=np.float32)
    return (0.5 * (arr + np.swapaxes(arr, -1, -2))).astype(np.float32)


def matrix_sqrt(matrix: ArrayLike) -> npt.NDArray[np.float32]:
    sym = symmetrize(matrix)
    eigvals, eigvecs = np.linalg.eigh(sym)
    eigvals_clipped = np.clip(eigvals, 0.0, None)
    sqrt_vals = np.sqrt(eigvals_clipped)[..., None, :]
    scaled_vecs = eigvecs * sqrt_vals
    sqrt_matrix = np.matmul(scaled_vecs, np.swapaxes(eigvecs, -1, -2))
    return symmetrize(sqrt_matrix)


def ensure_matrix(value: ArrayLike | float | Iterable[float]) -> npt.NDArray[np.float32]:
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim == 0:
        return np.eye(3, dtype=np.float32) * float(arr)
    if arr.ndim == 1:
        if arr.shape[0] != 3:
            raise ValueError("Gain vectors must have length 3.")
        return np.diag(arr.astype(np.float32))
    if arr.ndim >= 2:
        if arr.shape[-2:] != (3, 3):
            raise ValueError("Gain matrices must have trailing shape (3, 3).")
        return arr.astype(np.float32)
    raise ValueError("Unsupported gain array shape.")


def get_damping_matrix(
    stiffness: ArrayLike,
    inertia_like: ArrayLike | float | Iterable[float],
) -> npt.NDArray[np.float32]:
    stiffness_matrix = ensure_matrix(stiffness)
    inertia_matrix = ensure_matrix(inertia_like)
    mass_sqrt = matrix_sqrt(inertia_matrix)
    stiffness_sqrt = matrix_sqrt(stiffness_matrix)
    damping = 2.0 * np.matmul(mass_sqrt, stiffness_sqrt)
    return symmetrize(damping).astype(np.float32)
