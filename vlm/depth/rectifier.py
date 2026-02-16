"""Utilities for rectifying stereo image pairs."""

from __future__ import annotations

from copy import deepcopy
from typing import Dict, Tuple

import cv2
import numpy as np

from .depth_utils import get_rectification_maps


class Rectifier:
    """Apply stereo rectification and resize to the model input resolution."""

    def __init__(
        self,
        calib_params: Dict[str, np.ndarray],
        rec_params: Dict[str, np.ndarray],
        *,
        calib_size: Tuple[int, int],
        target_size: Tuple[int, int],
    ) -> None:
        self.calib_params = deepcopy(calib_params)
        self.rec_params = deepcopy(rec_params)

        self.map1_left, self.map2_left, self.map1_right, self.map2_right = (
            get_rectification_maps(self.calib_params, self.rec_params, calib_size)
        )

        width, height = target_size
        self.combined_map1_left = cv2.resize(
            self.map1_left, (width, height), interpolation=cv2.INTER_LINEAR
        )
        self.combined_map2_left = cv2.resize(
            self.map2_left, (width, height), interpolation=cv2.INTER_LINEAR
        )
        self.combined_map1_right = cv2.resize(
            self.map1_right, (width, height), interpolation=cv2.INTER_LINEAR
        )
        self.combined_map2_right = cv2.resize(
            self.map2_right, (width, height), interpolation=cv2.INTER_LINEAR
        )

    def rectify(
        self, left_image: np.ndarray, right_image: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Rectify both images using the precomputed maps."""
        rectified_left = cv2.remap(
            left_image,
            self.combined_map1_left,
            self.combined_map2_left,
            interpolation=cv2.INTER_LINEAR,
        )
        rectified_right = cv2.remap(
            right_image,
            self.combined_map1_right,
            self.combined_map2_right,
            interpolation=cv2.INTER_LINEAR,
        )
        return rectified_left, rectified_right
