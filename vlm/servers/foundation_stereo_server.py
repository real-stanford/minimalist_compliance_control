#!/usr/bin/env python3
"""Foundation Stereo depth estimation server."""

from __future__ import annotations

from typing import Any, Dict, Optional

import cv2
import numpy as np

from vlm.depth.foundation_stereo import DepthEstimator, DepthResult
from vlm.servers.foundation_model_server import (
    FoundationModelServer,
    ensure_image_array,
)


class FoundationStereoServer(FoundationModelServer):
    """Processes ZeroMQ requests using the Foundation Stereo depth estimator."""

    model_key = "fs"

    def __init__(
        self,
        *,
        response_ip: str,
        request_port: int,
        response_port: int,
        loop_rate_hz: float = 50.0,
        queue_len: int = 1,
        engine_path: str = "ckpts/foundation_stereo_vitl_480x640_20.engine",
    ) -> None:
        super().__init__(
            response_ip=response_ip,
            request_port=request_port,
            response_port=response_port,
            loop_rate_hz=loop_rate_hz,
            queue_len=queue_len,
        )

        self.depth_estimator = DepthEstimator(engine_path=engine_path)

    def handle_request(
        self, data: Dict[str, Any], config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run the Foundation Stereo depth model."""
        self.log("Foundation Stereo: decoding images from request")
        left_image = self.load_image(data, "left_image")
        right_image = self.load_image(data, "right_image")

        if left_image is None or right_image is None:
            raise ValueError(
                "Both 'left_image' and 'right_image' must be provided for model 'fs'"
            )

        calib_cfg = config.get("calibration")
        if not isinstance(calib_cfg, dict):
            raise ValueError(
                "Missing calibration config. Provide 'calibration' with "
                "calib_params, rec_params, calib_width, calib_height."
            )
        calib_params_raw = calib_cfg.get("calib_params")
        rec_params_raw = calib_cfg.get("rec_params")
        calib_width = calib_cfg.get("calib_width")
        calib_height = calib_cfg.get("calib_height")
        if (
            calib_params_raw is None
            or rec_params_raw is None
            or calib_width is None
            or calib_height is None
        ):
            raise ValueError(
                "Calibration config requires calib_params, rec_params, calib_width, calib_height."
            )

        calib_params = self.ensure_matrix_dict(calib_params_raw, "calib_params")
        rec_params = self.ensure_matrix_dict(rec_params_raw, "rec_params")

        calibration = DepthEstimator.compute_calibration(
            calib_params=calib_params,
            rec_params=rec_params,
            calib_width=int(calib_width),
            calib_height=int(calib_height),
            model_width=self.depth_estimator.width,
            model_height=self.depth_estimator.height,
        )

        remove_invisible = bool(config.get("remove_invisible", True))
        return_all = bool(config.get("return_all", True))
        skip_rectify = bool(config.get("skip_rectify", False)) or bool(
            config.get("pre_rectified", False)
        )

        self.log(
            "Foundation Stereo: executing depth estimation "
            f"(return_all={return_all}, remove_invisible={remove_invisible}, skip_rectify={skip_rectify})"
        )

        depth_result = self.depth_estimator.get_depth(
            img_left=left_image,
            img_right=right_image,
            calibration=calibration,
            remove_invisible=remove_invisible,
            return_all=return_all,
            skip_rectify=skip_rectify,
        )

        return self.depth_result_to_dict(depth_result)

    def ensure_matrix_dict(self, value: Any, field_name: str) -> Dict[str, np.ndarray]:
        """Ensure calibration/rectification payload is a dict of numpy arrays."""
        if not isinstance(value, dict):
            raise ValueError(f"'{field_name}' must be a dictionary.")
        normalized: Dict[str, np.ndarray] = {}
        for key, item in value.items():
            if isinstance(item, np.ndarray):
                normalized[key] = item
            else:
                normalized[key] = np.asarray(item, dtype=np.float64)
        return normalized

    def load_image(self, data: Dict[str, Any], key: str) -> Optional[np.ndarray]:
        """Decode an image provided inline or by path."""
        if key in data and data[key] is not None:
            return ensure_image_array(data[key])

        path_key = f"{key}_path"
        if path_key in data and data[path_key] is not None:
            image = cv2.imread(str(data[path_key]))
            if image is None:
                raise ValueError(f"Failed to load image at '{data[path_key]}'")
            return image

        return None

    def depth_result_to_dict(self, result: DepthResult) -> Dict[str, Any]:
        """Serialize DepthResult to a dictionary."""
        return {
            "depth": result.depth,
            # "disparity": result.disparity,
            # "rectified_left": result.rectified_left,
            # "rectified_right": result.rectified_right,
            # "original_left": result.original_left,
            # "original_right": result.original_right,
        }
