"""Array manipulation utilities using NumPy only."""

from typing import Any, Callable, Optional, Tuple
import numpy as np
import numpy.typing as npt
import scipy
from scipy.spatial.transform import Rotation as ScipyRotation
from scipy.spatial.transform import Slerp as ScipySlerp
from numpy.typing import NDArray


array_lib = np
ArrayType = npt.NDArray[np.float32]
expm = scipy.linalg.expm
R = ScipyRotation
Slerp = ScipySlerp


def inplace_update(
    array: ArrayType,
    idx: int | slice | ArrayType | tuple[int | slice | ArrayType, ...],
    value: Any,
) -> ArrayType:
    """Updates the specified elements of an array in place with a given value.

    Args:
        array (ArrayType): The array to be updated.
        idx (int | slice | ArrayType | tuple[int | slice | ArrayType, ...]): The indices of the elements to update. Can be an integer, slice, array, or a tuple of these.
        value (Any): The value to set at the specified indices.

    Returns:
        ArrayType: The updated array.
    """
    array[idx] = value
    return array


def inplace_add(
    array: ArrayType, idx: int | slice | tuple[int | slice, ...], value: Any
) -> ArrayType:
    """Performs an in-place addition to an array at specified indices.

    Args:
        array (ArrayType): The array to be updated.
        idx (int | slice | tuple[int | slice, ...]): The index or indices where the addition should occur.
        value (Any): The value to add to the specified indices.

    Returns:
        ArrayType: The updated array after performing the in-place addition.
    """
    array[idx] += value
    return array


def conditional_update(
    condition: bool | npt.NDArray[np.bool_],
    true_func: Callable[[], ArrayType],
    false_func: Callable[[], ArrayType],
) -> ArrayType:
    """
    Performs a conditional update using a standard if-else statement for NumPy.

    Args:
        condition: The condition to check.
        true_func: Function to execute if the condition is True.
        false_func: Function to execute if the condition is False.

    Returns:
        The result of true_func if condition is True, otherwise the result of false_func.
    """
    return true_func() if bool(condition) else false_func()


def loop_update(
    update_step: Callable[
        [Tuple[ArrayType, ArrayType], int],
        Tuple[Tuple[ArrayType, ArrayType], ArrayType],
    ],
    x: ArrayType,
    u: ArrayType,
    index_range: Tuple[int, int],
) -> ArrayType:
    """
    A general function to perform loop updates with NumPy.

    Args:
        N: Number of steps.
        traj_x: The state trajectory array.
        traj_u: The control input trajectory array.
        update_step: A function that defines how to update the state at each step.
    Returns:
        The updated trajectory array.
    """
    for i in range(*index_range):
        (x, u), _ = update_step((x, u), i)

    final_traj_x = x

    return final_traj_x


def random_uniform(
    low: float,
    high: float,
    rng: Optional[object] = None,
    shape: Optional[Tuple[int, ...]] = None,
) -> ArrayType:
    """Generates random uniform values with NumPy.

    Args:
        low: Lower bound of the uniform distribution.
        high: Upper bound of the uniform distribution.
        rng: Unused (kept for API compatibility).
        shape: Shape of the output array.

    Returns:
        Random uniform array.
    """
    return np.random.uniform(low=low, high=high, size=shape)


def ensure_array(value: ArrayType | NDArray[np.float32]) -> ArrayType:
    """Returns a NumPy array from input."""
    return np.asarray(value, dtype=np.float32)
