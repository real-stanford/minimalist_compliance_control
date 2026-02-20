"""Minimal math utilities used by standalone VLM modules."""

from __future__ import annotations

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
