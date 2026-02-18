"""TensorRT/ONNX stereo depth estimation for the foundation stereo model."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from .depth_utils import depth_to_xyzmap, to_o3d_cloud
from .rectifier import Rectifier

try:
    import onnxruntime as ort  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    ort = None  # type: ignore[assignment]

try:
    import tensorrt as trt  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    trt = None  # type: ignore[assignment]

try:
    from onnx_tensorrt import tensorrt_engine  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    tensorrt_engine = None  # type: ignore[assignment]

try:
    import torch  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    torch = None  # type: ignore[assignment]


if trt is not None:
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    trt.init_libnvinfer_plugins(TRT_LOGGER, "")
else:
    TRT_LOGGER = None


def _log(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [DepthEstimator] {message}")


def _preprocess_image(
    image: np.ndarray, target_height: int, target_width: int
) -> np.ndarray:
    if image.shape[0] != target_height or image.shape[1] != target_width:
        image = cv2.resize(image, (target_width, target_height))
    chw = np.transpose(image.astype(np.float32), (2, 0, 1))
    return np.ascontiguousarray(chw[None, ...], dtype=np.float32)


def _get_onnx_model(model_path: str) -> Any:
    if ort is None:
        raise ModuleNotFoundError(
            "onnxruntime is required to load .onnx foundation stereo models."
        )
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(
        model_path,
        sess_options=session_options,
        providers=["CUDAExecutionProvider"],
    )


def _get_model_input_shape(model_path: str) -> tuple[int, int]:
    if model_path.endswith(".onnx"):
        if ort is None:
            raise ModuleNotFoundError(
                "onnxruntime is required to inspect .onnx model shapes."
            )
        session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        input_shape = session.get_inputs()[0].shape
        if len(input_shape) < 4:
            raise ValueError(f"Unexpected model input shape: {input_shape}")
        return int(input_shape[2]), int(input_shape[3])

    if model_path.endswith(".engine") or model_path.endswith(".plan"):
        if trt is None or TRT_LOGGER is None:
            raise ModuleNotFoundError(
                "tensorrt is required to inspect .engine/.plan model shapes."
            )
        with open(model_path, "rb") as f:
            engine_data = f.read()
        runtime = trt.Runtime(TRT_LOGGER)
        engine = runtime.deserialize_cuda_engine(engine_data)
        if engine is None:
            raise RuntimeError(f"Failed to deserialize TensorRT engine: {model_path}")
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                shape = [d if d > 0 else 1 for d in engine.get_tensor_shape(name)]
                if len(shape) >= 4:
                    return int(shape[2]), int(shape[3])
        raise ValueError(f"Could not extract input shape from engine: {model_path}")

    raise ValueError(f"Unsupported model format: {model_path}")


def _get_engine_model(model_path: str) -> Any:
    if trt is None or TRT_LOGGER is None or tensorrt_engine is None:
        raise ModuleNotFoundError(
            "tensorrt and onnx_tensorrt are required to load .engine/.plan models."
        )
    with open(model_path, "rb") as file:
        engine_data = file.read()
    engine = trt.Runtime(TRT_LOGGER).deserialize_cuda_engine(engine_data)
    if engine is None:
        raise RuntimeError(f"Failed to deserialize TensorRT engine: {model_path}")
    return tensorrt_engine.Engine(engine)


@dataclass(frozen=True, slots=True)
class DepthResult:
    """Container for all artifacts produced by depth estimation."""

    depth: np.ndarray
    disparity: Optional[np.ndarray] = None
    rectified_left: Optional[np.ndarray] = None
    rectified_right: Optional[np.ndarray] = None
    original_left: Optional[np.ndarray] = None
    original_right: Optional[np.ndarray] = None


@dataclass(frozen=True, slots=True)
class CalibrationData:
    """Precomputed stereo calibration artifacts required for depth estimation."""

    rectifier: Rectifier
    fx_times_baseline: float
    baseline_sign: float
    scaled_rectified_K: np.ndarray


class DepthEstimator:
    """Depth estimation using an ONNX or TensorRT foundation stereo model."""

    def __init__(self, engine_path: str) -> None:
        self._init_engine(engine_path)

    def _init_engine(self, model_path: str) -> None:
        model_path_obj = Path(model_path)
        if not model_path_obj.exists():
            raise FileNotFoundError(f"Foundation stereo model not found: {model_path}")

        self.height, self.width = _get_model_input_shape(model_path)
        _log(f"Extracted model input shape: {self.height}x{self.width}")

        if model_path.endswith(".onnx"):
            self.model = _get_onnx_model(model_path)
            self.model_type = "onnx"
            _log("ONNX model loaded")
            return

        if model_path.endswith(".engine") or model_path.endswith(".plan"):
            self.model = _get_engine_model(model_path)
            self.model_type = "engine"
            _log("TensorRT engine model loaded")
            return

        raise ValueError(
            f"Unknown model format {model_path}. Supported: .onnx, .engine, .plan"
        )

    @staticmethod
    def compute_calibration(
        calib_params: dict,
        rec_params: dict,
        calib_width: int,
        calib_height: int,
        model_width: int,
        model_height: int,
    ) -> CalibrationData:
        """Build calibration artifacts from loaded calibration dictionaries."""

        t_vec = np.asarray(calib_params["T"], dtype=np.float64).reshape(-1)
        scale_factor = model_width / calib_width
        if not np.isclose(scale_factor, model_height / calib_height):
            raise ValueError(
                "Scale factor must match for width and height: "
                f"{scale_factor} vs {model_height / calib_height}"
            )

        baseline = float(np.linalg.norm(t_vec))
        baseline_sign = float(np.sign(t_vec[0]))
        if baseline_sign == 0.0:
            raise ValueError("Baseline sign is zero; expected non-zero X baseline.")

        p1 = np.asarray(rec_params["P1"], dtype=np.float64)
        scaled_rectified_fx = float(p1[0, 0] * scale_factor)
        scaled_rectified_fy = float(p1[1, 1] * scale_factor)
        scaled_rectified_cx = float(p1[0, 2] * scale_factor)
        scaled_rectified_cy = float(p1[1, 2] * scale_factor)
        scaled_rectified_k = np.array(
            [
                [scaled_rectified_fx, 0.0, scaled_rectified_cx],
                [0.0, scaled_rectified_fy, scaled_rectified_cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

        rectifier = Rectifier(
            calib_params,
            rec_params,
            calib_size=(int(calib_width), int(calib_height)),
            target_size=(int(model_width), int(model_height)),
        )

        return CalibrationData(
            rectifier=rectifier,
            fx_times_baseline=float(scaled_rectified_fx * baseline),
            baseline_sign=baseline_sign,
            scaled_rectified_K=scaled_rectified_k,
        )

    def _infer(self, left_img: np.ndarray, right_img: np.ndarray) -> np.ndarray:
        """Run model inference and return disparity."""
        left_tensor = np.ascontiguousarray(
            _preprocess_image(left_img, self.height, self.width), dtype=np.float32
        )
        right_tensor = np.ascontiguousarray(
            _preprocess_image(right_img, self.height, self.width), dtype=np.float32
        )

        if torch is not None and torch.cuda.is_available():
            torch.cuda.synchronize()
        if self.model_type == "onnx":
            disparity = self.model.run(
                None, {"left": left_tensor, "right": right_tensor}
            )[0]
        else:
            disparity = self.model.run([left_tensor, right_tensor])[0]
        if torch is not None and torch.cuda.is_available():
            torch.cuda.synchronize()

        return np.asarray(disparity).squeeze()

    def get_depth(
        self,
        img_left: np.ndarray,
        img_right: np.ndarray,
        *,
        calibration: CalibrationData,
        remove_invisible: bool = False,
        return_all: bool = False,
        skip_rectify: bool = False,
    ) -> DepthResult:
        """Estimate per-pixel depth from a stereo pair."""
        original_left = img_left.copy() if return_all else None
        original_right = img_right.copy() if return_all else None

        if not skip_rectify:
            img_left, img_right = calibration.rectifier.rectify(img_left, img_right)

        rectified_left = img_left.copy() if return_all else None
        rectified_right = img_right.copy() if return_all else None

        if calibration.baseline_sign < 0:
            disparity = self._infer(img_left, img_right)
        elif calibration.baseline_sign > 0:
            disparity = self._infer(img_right, img_left)
        else:
            raise ValueError("Baseline sign is zero.")

        if remove_invisible:
            xx = np.arange(disparity.shape[1])[None, :]
            invalid = (xx - disparity) < 0
            disparity = disparity.copy()
            disparity[invalid] = np.inf

        depth = calibration.fx_times_baseline / disparity
        return DepthResult(
            depth=depth,
            disparity=disparity if return_all else None,
            rectified_left=rectified_left,
            rectified_right=rectified_right,
            original_left=original_left,
            original_right=original_right,
        )

    def get_point_cloud(
        self,
        depth: np.ndarray,
        resized_image: np.ndarray,
        calibration: CalibrationData,
        *,
        is_bgr: bool = True,
        zmin: float = 0.0,
        zmax: Optional[float] = None,
        denoise_cloud: bool = False,
        denoise_nb_points: int = 30,
        denoise_radius: float = 0.03,
    ) -> tuple[Any, np.ndarray]:
        """Convert a depth map into an Open3D point cloud and XYZ map."""
        max_depth = np.inf if zmax is None else zmax
        xyz_map = depth_to_xyzmap(depth, calibration.scaled_rectified_K, zmin=zmin)

        color_image = resized_image
        if is_bgr:
            color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

        pcd = to_o3d_cloud(xyz_map.reshape(-1, 3), color_image.reshape(-1, 3))
        points = np.asarray(pcd.points)
        keep_ids = np.where((points[:, 2] > 0) & (points[:, 2] <= max_depth))[0]
        pcd = pcd.select_by_index(keep_ids)

        if denoise_cloud:
            _, ind = pcd.remove_radius_outlier(
                nb_points=denoise_nb_points,
                radius=denoise_radius,
            )
            pcd = pcd.select_by_index(ind)

        return pcd, xyz_map
