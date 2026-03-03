#!/usr/bin/env python3
"""Affordance predictor entry point without external config dependency."""

from __future__ import annotations

import json
import os
import pickle
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import open3d as o3d
import pycocotools.mask as mask_util
import yaml
import zmq

from ..depth.depth_utils import depth_to_xyzmap, to_o3d_cloud, vis_disparity
from ..depth.rectifier import Rectifier
from ..utils.comm_utils import ZMQNode
from .compliance_predictor import CompliancePredictor
from .plan_ee_pose import (
    LEAP_CAMERA_EXTRINSICS,
    TODDY_CAMERA_EXTRINSICS,
)

# from toddlerbot.utils.misc_utils import profile

DEFAULT_SERVER_IP = "192.168.0.144"
DEFAULT_FS_PORTS: Tuple[int, int] = (5555, 5556)
DEFAULT_GSAM_PORTS: Tuple[int, int] = (5560, 5561)
DEFAULT_FS_TIMEOUT = 5.0
DEFAULT_GSAM_TIMEOUT = 5.0
DEFAULT_SAM3_PORTS: Tuple[int, int] = (5580, 5581)
DEFAULT_SAM3_TIMEOUT = 5.0
MIN_MASK_PIXELS = 200
MAX_MASK_CENTER_DISTANCE_FRAC = 1.0
DEFAULT_TASK = "wipe up the black ink on the whiteboard with an eraser."


_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parents[1]
_DEFAULT_CAMERA_CONFIG = _REPO_ROOT / "assets" / "toddlerbot_camera.yml"

DEPTH_CONFIG = {
    "camera_config": str(_DEFAULT_CAMERA_CONFIG),
    "calib_height": 480,
    "calib_width": 640,
    "zmax": 0.5,
}

DEFAULT_CAMERA_SIDE = "left"

# Placeholder workspace sizes (in meters) that will be populated by the user.
TODDY_WORK_SPACE_SIZE: np.ndarray = np.array([0.08, 0.08], dtype=np.float32)
LEAP_WORKSPACE_SIZE_BY_SITE: Dict[str, np.ndarray] = {
    "if_tip": np.array([0.06, 0.06], dtype=np.float32),
    "mf_tip": np.array([0.08, 0.08], dtype=np.float32),
    "rf_tip": np.array([0.06, 0.06], dtype=np.float32),
    "th_tip": np.array([0.04, 0.06], dtype=np.float32),
}

CAMERA_EXTRINSICS_BY_VARIANT = {
    "leap": LEAP_CAMERA_EXTRINSICS,
    "toddlerbot": TODDY_CAMERA_EXTRINSICS,
}

WORKSPACE_SIZE_BY_VARIANT = {
    "leap": LEAP_WORKSPACE_SIZE_BY_SITE,
    "toddlerbot": TODDY_WORK_SPACE_SIZE,
}

# TODO: remove the hard code
LEAP_EE_LOCATIONS: Dict[str, np.ndarray] = {
    "if_tip": np.array([0.0927 + 0.05, 0.017, 0.0], dtype=np.float32),
    "mf_tip": np.array([0.0927 + 0.05, 0.0624, 0.0], dtype=np.float32),
    "rf_tip": np.array([0.0927 + 0.05, 0.1078, 0.0], dtype=np.float32),
    "th_tip": np.array([0.0174 + 0.05, 0.0127, 0.0], dtype=np.float32),
}
TODDY_EE_LOCATIONS: Dict[str, np.ndarray] = {
    "left_hand_center": np.array([0.0, 0.0, 0.1], dtype=np.float32),
    "right_hand_center": np.array([0.0, -0.0, 0.1], dtype=np.float32),
}
EE_LOCATIONS_BY_VARIANT = {
    "leap": LEAP_EE_LOCATIONS,
    "toddlerbot": TODDY_EE_LOCATIONS,
}


@dataclass
class ZMQClient:
    sender: ZMQNode
    receiver: ZMQNode
    lock: Lock
    poller: zmq.Poller


def _to_numpy_fields(mapping: Dict[str, Any]) -> Dict[str, Any]:
    converted: Dict[str, Any] = {}
    for key, value in mapping.items():
        if isinstance(value, list):
            converted[key] = np.asarray(value, dtype=np.float64)
        else:
            converted[key] = value
    return converted


def _load_camera_config_params(
    camera_config_path: Union[str, Path],
) -> tuple[dict, dict]:
    config_path = Path(camera_config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Missing camera config: {config_path}")
    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Camera config must be a mapping: {config_path}")
    calibration = data.get("calibration")
    rectification = data.get("rectification")
    if not isinstance(calibration, dict) or not isinstance(rectification, dict):
        raise ValueError(
            f"Camera config must include mapping keys 'calibration' and 'rectification': {config_path}"
        )
    return _to_numpy_fields(calibration), _to_numpy_fields(rectification)


def merge_point_cloud(
    rgb_arr: np.ndarray,
    xyz_map: np.ndarray,
    contact_points_3d: np.ndarray,
    zmax: float,
    voxel_size: float = 0.003,
) -> Tuple[o3d.geometry.PointCloud, List[int]]:
    if rgb_arr.ndim == 2:
        color_for_cloud = cv2.cvtColor(rgb_arr, cv2.COLOR_GRAY2BGR)
    else:
        color_for_cloud = rgb_arr

    color_rgb = cv2.cvtColor(color_for_cloud, cv2.COLOR_BGR2RGB)

    pcd = to_o3d_cloud(xyz_map.reshape(-1, 3), color_rgb.reshape(-1, 3))

    points = np.asarray(pcd.points)
    depth_mask = np.isfinite(points[:, 2]) & (points[:, 2] > 0)
    if np.isfinite(zmax) and zmax > 0:
        depth_mask &= points[:, 2] <= zmax
    keep_ids = np.where(depth_mask)[0]
    pcd = pcd.select_by_index(keep_ids)
    pcd = pcd.voxel_down_sample(voxel_size)

    contact_points_arr = np.asarray(contact_points_3d, dtype=np.float64)
    contact_indices: List[int] = []
    if contact_points_arr.size == 0:
        return pcd, contact_indices

    existing_points = np.asarray(pcd.points)
    start_index = existing_points.shape[0]
    stacked_points = np.vstack([existing_points, contact_points_arr])
    pcd.points = o3d.utility.Vector3dVector(stacked_points)

    existing_colors = np.asarray(pcd.colors)
    if existing_colors.shape[0] == existing_points.shape[0]:
        contact_colors = np.tile(
            np.array([[1.0, 0.0, 0.0]]), (contact_points_arr.shape[0], 1)
        )
        stacked_colors = np.vstack([existing_colors, contact_colors])
        pcd.colors = o3d.utility.Vector3dVector(stacked_colors)

    contact_indices = list(
        range(start_index, start_index + contact_points_arr.shape[0])
    )

    return pcd, contact_indices


def postprocess_sam_result(
    sam_result: Dict[str, Any],
    target_priority: List[str],
    target_shape: Optional[Tuple[int, int]] = None,
) -> Tuple[Dict[str, Any], np.ndarray, Optional[np.ndarray]]:
    """Combine SAM3 masks by prompt label (as in test_foundation_server) and pick the best target."""

    def normalize(value: Any) -> Optional[str]:
        if isinstance(value, str):
            return value.replace(".", "").strip().lower()
        return None

    mask_rles = sam_result.get("mask_rles") or []
    if not mask_rles:
        raise RuntimeError("Segmentation response contained no masks")

    prompts = sam_result.get("prompts") or []
    prompt_ids = sam_result.get("prompt_ids") or []
    mask_prompt_ids = sam_result.get("mask_prompt_ids") or []
    boxes = sam_result.get("boxes") or []
    scores = sam_result.get("scores") or []

    candidate_labels = [label for label in target_priority if str(label).strip()]
    if not candidate_labels:
        raise RuntimeError(
            "No valid target object labels provided for annotation lookup."
        )

    priority_map: Dict[str, int] = {}
    for idx, label in enumerate(candidate_labels):
        normalized = normalize(label)
        if normalized:
            priority_map.setdefault(normalized, idx)
    if not priority_map:
        raise RuntimeError(
            "Failed to derive normalized target labels from annotation specification."
        )

    def prompt_for_mask(mask_idx: int) -> tuple[str, Optional[int]]:
        pid = None
        if mask_idx < len(mask_prompt_ids) and mask_prompt_ids[mask_idx] is not None:
            pid = mask_prompt_ids[mask_idx]
        elif mask_idx < len(prompt_ids):
            pid = prompt_ids[mask_idx]
        if pid is not None and 0 <= pid < len(prompts):
            return prompts[pid], pid
        return f"prompt_{pid}", pid

    combined: Dict[str, Dict[str, Any]] = {}
    union_mask: Optional[np.ndarray] = None
    for idx, rle in enumerate(mask_rles):
        label, pid = prompt_for_mask(idx)
        decoded = decode_mask(rle)
        if target_shape is not None and decoded.shape != target_shape:
            decoded = cv2.resize(
                decoded,
                (target_shape[1], target_shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        entry = combined.setdefault(label, {"mask": None, "scores": [], "bbox": None})
        if entry["mask"] is None:
            entry["mask"] = np.zeros(decoded.shape, dtype=np.uint8)
        entry["mask"] = np.maximum(entry["mask"], (decoded > 0).astype(np.uint8))
        if idx < len(scores):
            entry["scores"].append(scores[idx])
        if idx < len(boxes):
            entry["bbox"] = boxes[idx]
        if union_mask is None:
            union_mask = np.zeros(decoded.shape, dtype=np.uint8)
        union_mask = np.maximum(union_mask, (decoded > 0).astype(np.uint8))

    best_label = None
    best_priority = None
    for label in combined.keys():
        norm = normalize(label)
        if norm is None or norm not in priority_map:
            continue
        priority = priority_map[norm]
        if best_priority is None or priority < best_priority:
            best_priority = priority
            best_label = label

    if best_label is None:
        best_label = next(iter(combined.keys()))

    selected = combined[best_label]
    mask_arr = selected["mask"]
    if mask_arr is None:
        raise RuntimeError("Combined mask was empty")

    ys, xs = np.nonzero(mask_arr)
    bbox = None
    if ys.size > 0 and xs.size > 0:
        bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
    # Prefer bbox derived from the merged selected mask. Raw model boxes can
    # represent only one component/proposal and may under-cover the target.
    if bbox is None and selected.get("bbox") is not None:
        bbox = selected["bbox"]

    score = float(np.mean(selected["scores"])) if selected["scores"] else None

    annotation = {
        "label": best_label or (candidate_labels[0] if candidate_labels else ""),
        "bbox": bbox,
        "score": score,
    }

    return annotation, mask_arr, union_mask


def decode_mask(rle: Dict[str, Any]) -> np.ndarray:
    mask = mask_util.decode(rle)
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    return (mask > 0).astype(np.uint8)


def trim_mask_by_center(
    mask: np.ndarray, max_center_distance_frac: float = MAX_MASK_CENTER_DISTANCE_FRAC
) -> Tuple[np.ndarray, int, int, float]:
    """Remove mask pixels farther than a fraction of the image half-diagonal."""
    ys, xs = np.nonzero(mask)
    total = int(ys.size)
    if total == 0:
        return mask, 0, 0, 0.0
    height, width = mask.shape[:2]
    center_x = (width - 1) * 0.5
    center_y = (height - 1) * 0.5
    dist_px = np.hypot(
        xs.astype(np.float32) - center_x, ys.astype(np.float32) - center_y
    )
    max_dist = 0.5 * float(np.hypot(width, height)) * float(max_center_distance_frac)
    keep = dist_px <= max_dist
    removed = int(total - int(np.count_nonzero(keep)))
    if removed == 0:
        return mask, 0, total, max_dist
    trimmed = np.zeros_like(mask)
    trimmed[ys[keep], xs[keep]] = 1
    return trimmed, removed, total, max_dist


def extract_mask_pixels(
    mask: np.ndarray, grid_size: int = 30, trim_region: int = 0
) -> List[Dict[str, int]]:
    """Subsample mask pixels using approximate grid sampling."""

    if grid_size <= 0:
        raise ValueError("grid_size must be positive")
    if trim_region < 0:
        raise ValueError("trim_region must be non-negative")

    def fill_small_holes(binary_mask: np.ndarray, radius_pixels: int) -> np.ndarray:
        """Fill interior background islands smaller than the trim band."""

        if radius_pixels <= 0:
            return binary_mask

        mask_uint8 = binary_mask.astype(np.uint8)
        inverse_mask = (1 - mask_uint8).astype(np.uint8)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            inverse_mask, connectivity=4
        )
        if num_labels <= 1:
            return binary_mask

        height, width = binary_mask.shape
        max_hole_area = max(int(np.pi * radius_pixels * radius_pixels), 64)
        filled_mask = mask_uint8.copy()

        for label_idx in range(1, num_labels):
            area = stats[label_idx, cv2.CC_STAT_AREA]
            if area > max_hole_area:
                continue
            left = stats[label_idx, cv2.CC_STAT_LEFT]
            top = stats[label_idx, cv2.CC_STAT_TOP]
            right = left + stats[label_idx, cv2.CC_STAT_WIDTH]
            bottom = top + stats[label_idx, cv2.CC_STAT_HEIGHT]
            touches_border = left == 0 or top == 0 or right == width or bottom == height
            if touches_border:
                continue
            filled_mask[labels == label_idx] = 1

        return filled_mask.astype(bool)

    positive_mask = mask > 0
    if trim_region > 0:
        positive_mask = fill_small_holes(positive_mask, trim_region)

    ys, xs = np.where(positive_mask)
    if len(xs) == 0:
        raise RuntimeError("Segmentation mask contains no positive pixels")

    if trim_region > 0:
        distance_map = cv2.distanceTransform(
            positive_mask.astype(np.uint8), cv2.DIST_L2, 3
        )
        keep = distance_map[ys, xs] > trim_region
        if np.any(keep):
            ys = ys[keep]
            xs = xs[keep]
        else:
            print(
                "[AffordancePredictor] trim_region removed all mask pixels; "
                "falling back to untrimmed mask."
            )
            ys, xs = np.where(positive_mask)

    x_min, x_max = xs.min(), xs.max() + 1
    y_min, y_max = ys.min(), ys.max() + 1

    x_edges = np.linspace(x_min, x_max, grid_size + 1)
    y_edges = np.linspace(y_min, y_max, grid_size + 1)

    candidates: List[Dict[str, int]] = []
    xs_float = xs.astype(np.float64)
    ys_float = ys.astype(np.float64)

    for y_idx in range(grid_size):
        y_lower, y_upper = y_edges[y_idx], y_edges[y_idx + 1]
        y_mask = (ys_float >= y_lower) & (ys_float < y_upper)
        if not np.any(y_mask):
            continue

        for x_idx in range(grid_size):
            x_lower, x_upper = x_edges[x_idx], x_edges[x_idx + 1]
            cell_mask = y_mask & (xs_float >= x_lower) & (xs_float < x_upper)
            if not np.any(cell_mask):
                continue

            cell_indices = np.nonzero(cell_mask)[0]
            cell_xs = xs_float[cell_indices]
            cell_ys = ys_float[cell_indices]
            center_x = (x_lower + x_upper) * 0.5
            center_y = (y_lower + y_upper) * 0.5
            distances = (cell_xs - center_x) ** 2 + (cell_ys - center_y) ** 2
            best_idx = np.argmin(distances)
            selected_x = int(cell_xs[best_idx])
            selected_y = int(cell_ys[best_idx])

            candidates.append(
                {
                    "id": int(len(candidates)),
                    "x": selected_x,
                    "y": selected_y,
                }
            )

    return candidates


def prepare_candidate_points(
    annotations: Dict[str, Dict[str, Any]],
    image_shape: Tuple[int, int],
    site_names: List[str],
    grid_size: int = 30,
    bbox_padding: Union[int, Tuple[int, int]] = 0,
) -> Dict[str, List[Dict[str, int]]]:
    height, width = image_shape
    candidates_by_site: Dict[str, List[Dict[str, int]]] = {}

    stride = max(1, int(grid_size))
    if bbox_padding is None:
        pad_x_steps = 0
        pad_y_steps = 0
    elif isinstance(bbox_padding, (tuple, list)):
        if len(bbox_padding) != 2:
            raise ValueError("bbox_padding must have two values: (pad_x, pad_y)")
        pad_x_steps = int(bbox_padding[0])
        pad_y_steps = int(bbox_padding[1])
    else:
        pad_x_steps = int(bbox_padding)
        pad_y_steps = int(bbox_padding)

    pad_x = pad_x_steps * stride
    pad_y = pad_y_steps * stride

    def axis_values(start: int, span: int, limit: int) -> np.ndarray:
        steps = int(span // stride)
        values = start + np.arange(steps + 1, dtype=int) * stride
        values = np.clip(values, 0, limit - 1)
        values = np.unique(values)
        if values.size == 0:
            values = np.array([np.clip(start, 0, limit - 1)], dtype=int)
        return values

    for site in site_names:
        ann = annotations.get(site)
        if not ann:
            continue
        bbox = ann.get("bbox")
        if not bbox or len(bbox) != 4:
            continue

        x1, y1, x2, y2 = map(float, bbox)
        x_min = int(np.floor(min(x1, x2)))
        x_max = int(np.ceil(max(x1, x2)))
        y_min = int(np.floor(min(y1, y2)))
        y_max = int(np.ceil(max(y1, y2)))

        x_min -= pad_x
        x_max += pad_x
        y_min -= pad_y
        y_max += pad_y

        if width <= 0 or height <= 0:
            continue

        x_min = int(np.clip(x_min, 0, width - 1))
        x_max = int(np.clip(x_max, 0, width - 1))
        y_min = int(np.clip(y_min, 0, height - 1))
        y_max = int(np.clip(y_max, 0, height - 1))

        if x_max <= x_min:
            x_max = min(width - 1, x_min + 1)
        if y_max <= y_min:
            y_max = min(height - 1, y_min + 1)

        span_x = max(1, x_max - x_min)
        span_y = max(1, y_max - y_min)

        x_values = axis_values(x_min, span_x, width)
        y_values = axis_values(y_min, span_y, height)

        candidates: List[Dict[str, int]] = []
        seen_coords: set[Tuple[int, int]] = set()

        for y_coord in y_values:
            for x_coord in x_values:
                coord = (x_coord, y_coord)
                if coord in seen_coords:
                    continue
                seen_coords.add(coord)
                candidates.append({"id": len(candidates), "x": x_coord, "y": y_coord})

        if not candidates:
            candidates.append({"id": 0, "x": x_min, "y": y_min})

        candidates_by_site[site] = candidates

    return candidates_by_site


def draw_points_overlay(
    image: np.ndarray,
    grouped: Dict[str, List[Dict[str, int]]],
    annotate: bool = False,
    circle_color: Tuple[int, int, int] = (0, 0, 255),
    circle_radius: int = 4,
    circle_thickness: int = -1,
    label_color: Tuple[int, int, int] = (0, 255, 0),
    label_offset: Tuple[int, int] = (0, -10),
    font_scale: float = 0.4,
    font_thickness: int = 1,
) -> np.ndarray:
    """Overlay circles (and optional labels) onto an image."""

    if image.ndim == 2:
        canvas = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        canvas = image.copy()

    def extract_point(point):
        if isinstance(point, dict):
            return int(point["x"]), int(point["y"])
        return int(point[0]), int(point[1])

    for site, pts in grouped.items():
        x, y = extract_point(pts[0])
        cv2.putText(
            canvas,
            site,
            (x + label_offset[0], y + label_offset[1]),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            label_color,
            font_thickness,
            cv2.LINE_AA,
        )
        for i, pt in enumerate(pts):
            x, y = extract_point(pt)
            cv2.circle(canvas, (x, y), circle_radius, circle_color, circle_thickness)
            if annotate:
                cv2.putText(
                    canvas,
                    str(i),
                    (x - circle_radius, y - circle_radius),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    label_color,
                    font_thickness,
                    cv2.LINE_AA,
                )

    return canvas


def draw_annotation_overlay(
    image: np.ndarray,
    annotation: Dict[str, Any],
    box_color: Tuple[int, int, int] = (0, 255, 255),
    box_thickness: int = 2,
    label_color: Tuple[int, int, int] = (0, 0, 0),
    label_bg_color: Tuple[int, int, int] = (0, 255, 255),
    font_scale: float = 0.5,
    font_thickness: int = 1,
) -> np.ndarray:
    """Draw a bounding box (and optional label/score) for a segmentation annotation."""

    if image.ndim == 2:
        canvas = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        canvas = image.copy()

    height, width = canvas.shape[:2]
    bbox = annotation.get("bbox")
    if bbox and len(bbox) == 4 and width > 0 and height > 0:
        x1, y1, x2, y2 = map(float, bbox)
        pt1 = (
            int(np.clip(round(x1), 0, width - 1)),
            int(np.clip(round(y1), 0, height - 1)),
        )
        pt2 = (
            int(np.clip(round(x2), 0, width - 1)),
            int(np.clip(round(y2), 0, height - 1)),
        )
        cv2.rectangle(canvas, pt1, pt2, box_color, box_thickness)

        label = annotation.get("label") or annotation.get("class_name")
        score = annotation.get("score")
        if label:
            parts = []
            if isinstance(score, (int, float)):
                parts.append(f"{score:.2f}")
            parts.append(str(label))
            label_text = " ".join(parts)

            (text_width, text_height), baseline = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
            )
            text_origin = (pt1[0], max(pt1[1] - 4, text_height + baseline))
            rect_start = (text_origin[0], text_origin[1] - text_height - baseline)
            rect_end = (text_origin[0] + text_width, text_origin[1] + baseline)
            rect_start = (
                int(np.clip(rect_start[0], 0, width - 1)),
                int(np.clip(rect_start[1], 0, height - 1)),
            )
            rect_end = (
                int(np.clip(rect_end[0], 0, width - 1)),
                int(np.clip(rect_end[1], 0, height - 1)),
            )
            cv2.rectangle(canvas, rect_start, rect_end, label_bg_color, -1)
            cv2.putText(
                canvas,
                label_text,
                (rect_start[0], rect_end[1] - baseline),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                label_color,
                font_thickness,
                cv2.LINE_AA,
            )

    return canvas


def project_head_points_to_pixels(
    camera_points: np.ndarray, intrinsic_matrix: np.ndarray
) -> np.ndarray:
    """Project camera-frame points into pixel coordinates."""
    positive_depth = camera_points[:, 2] > 1e-6
    if not np.any(positive_depth):
        return np.empty((0, 2), dtype=np.float32)
    camera_points = camera_points[positive_depth]

    K = np.asarray(intrinsic_matrix, dtype=np.float32)
    if K.shape[1] > 3:
        K = K[:, :3]
    if K.shape[0] != 3 or K.shape[1] != 3:
        return np.empty((0, 2), dtype=np.float32)

    pixels_h = (K @ camera_points.T).T
    depth = pixels_h[:, 2:3]
    valid_depth = np.abs(depth[:, 0]) > 1e-6
    if not np.any(valid_depth):
        return np.empty((0, 2), dtype=np.float32)

    pixels_h = pixels_h[valid_depth]
    depth = depth[valid_depth]
    pixels = pixels_h[:, :2] / depth
    return pixels.astype(np.float32)


def compute_workspace_rectangles(
    ee_pos: Dict[str, np.ndarray],
    workspace_size: Dict[str, np.ndarray],
    camera_extrinsics: Dict[str, np.ndarray],
    intrinsic_matrix: np.ndarray,
    depth_value: Optional[float] = None,
    image_shape: Optional[Tuple[int, int]] = None,
    site_names: Optional[List[str]] = None,
) -> List[Tuple[str, Tuple[int, int, int, int]]]:
    """Compute pixel rectangles for each body root using head-frame offsets."""

    if not ee_pos:
        return []

    camera_to_head = camera_extrinsics.get(DEFAULT_CAMERA_SIDE)
    if camera_to_head is None:
        return []

    try:
        head_to_camera = np.linalg.inv(camera_to_head)
    except np.linalg.LinAlgError:
        head_to_camera = None

    size_lookup = {
        k: np.asarray(v, dtype=np.float32).reshape(-1)
        for k, v in workspace_size.items()
    }

    target_sites = set(map(str, site_names)) if site_names else None

    rectangles: List[Tuple[str, Tuple[int, int, int, int]]] = []

    for name, root in ee_pos.items():
        if target_sites and str(name) not in target_sites:
            continue
        size_arr = size_lookup.get(name)
        if size_arr is None or size_arr.size < 2:
            continue
        half_w, half_h = float(size_arr[0] * 0.5), float(size_arr[1] * 0.5)
        root_arr = np.asarray(root, dtype=np.float32)
        if root_arr.shape[0] != 3:
            continue

        if head_to_camera is None:
            continue

        center_cam = (head_to_camera @ np.append(root_arr, 1.0))[:3]
        if depth_value is not None and np.isfinite(depth_value):
            center_cam = center_cam.astype(np.float32)
            center_cam[2] = float(depth_value)

        offsets_cam = np.array(
            [
                [-half_w, -half_h, 0.0],
                [-half_w, half_h, 0.0],
                [half_w, -half_h, 0.0],
                [half_w, half_h, 0.0],
            ],
            dtype=np.float32,
        )
        camera_points = center_cam.reshape(1, 3) + offsets_cam

        pixels = project_head_points_to_pixels(camera_points, intrinsic_matrix)
        if pixels.shape[0] == 0:
            continue

        x_min, y_min = np.floor(pixels.min(axis=0)).astype(int)
        x_max, y_max = np.ceil(pixels.max(axis=0)).astype(int)

        if image_shape is not None:
            height, width = image_shape
            if width <= 0 or height <= 0:
                continue
            x_min = int(np.clip(x_min, 0, width - 1))
            x_max = int(np.clip(x_max, 0, width - 1))
            y_min = int(np.clip(y_min, 0, height - 1))
            y_max = int(np.clip(y_max, 0, height - 1))

        if x_max <= x_min or y_max <= y_min:
            continue

        rectangles.append((str(name), (x_min, y_min, x_max, y_max)))

    return rectangles


def draw_workspace_overlay(
    image: np.ndarray,
    rectangles: List[Tuple[str, Tuple[int, int, int, int]]],
    box_color: Tuple[int, int, int] = (255, 0, 0),
    box_thickness: int = 2,
) -> np.ndarray:
    """Draw workspace rectangles (centered at body roots) onto an image."""

    if not rectangles:
        return image

    if image.ndim == 2:
        canvas = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        canvas = image.copy()

    height, width = canvas.shape[:2]

    for _, (x1, y1, x2, y2) in rectangles:
        if width <= 0 or height <= 0:
            break
        pt1 = (
            int(np.clip(x1, 0, width - 1)),
            int(np.clip(y1, 0, height - 1)),
        )
        pt2 = (
            int(np.clip(x2, 0, width - 1)),
            int(np.clip(y2, 0, height - 1)),
        )
        cv2.rectangle(canvas, pt1, pt2, box_color, box_thickness)

    return canvas


def ensure_uint8_image(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image
    clipped = np.clip(image, 0, 255)
    return clipped.astype(np.uint8)


def median_depth_from_mask(
    depth_map: Optional[np.ndarray], mask: Optional[np.ndarray]
) -> Optional[float]:
    if depth_map is None or mask is None:
        return None
    if depth_map.shape[:2] != mask.shape[:2]:
        return None
    valid = np.isfinite(depth_map) & (depth_map > 0) & (mask > 0)
    if not np.any(valid):
        return None
    return float(np.median(depth_map[valid]))


def nearest_valid_xyz(
    xyz_map: np.ndarray, u: int, v: int, max_radius: int = 5
) -> Optional[np.ndarray]:
    """Find the nearest valid xyz sample within a growing window.

    The returned z is the median depth of samples on the ring at ``max_radius``.
    x/y are fit with a simple affine model (u,v,1 -> x/y) using ring samples,
    avoiding reuse of the nearest pixel's x/y.
    """
    h, w = xyz_map.shape[:2]
    if not (0 <= u < w and 0 <= v < h):
        return None
    num_samples = max(32, 16 * max_radius)
    angles = np.linspace(0, 2 * np.pi, num_samples, endpoint=False)
    coords = set()
    for ang in angles:
        x_samp = int(round(u + max_radius * np.cos(ang)))
        y_samp = int(round(v + max_radius * np.sin(ang)))
        if 0 <= x_samp < w and 0 <= y_samp < h:
            coords.add((x_samp, y_samp))

    if not coords:
        return None

    xs_all = np.fromiter((c[0] for c in coords), dtype=np.int64)
    ys_all = np.fromiter((c[1] for c in coords), dtype=np.int64)
    valid_mask = xyz_map[ys_all, xs_all, 2] > 0
    if not np.any(valid_mask):
        return None

    xs = xs_all[valid_mask]
    ys = ys_all[valid_mask]
    if len(xs) < 2:
        return None

    z_vals = xyz_map[ys, xs, 2]
    z_med = float(np.median(z_vals))

    design = np.stack([xs, ys, np.ones_like(xs)], axis=1).astype(np.float32)
    x_vals = xyz_map[ys, xs, 0]
    y_vals = xyz_map[ys, xs, 1]
    if design.shape[0] < 3:
        return None

    query = np.array([u, v, 1.0], dtype=np.float32)
    try:
        coeff_x, _, rank_x, _ = np.linalg.lstsq(design, x_vals, rcond=None)
        coeff_y, _, rank_y, _ = np.linalg.lstsq(design, y_vals, rcond=None)
        if rank_x < 2 or rank_y < 2:
            return None
        pred_x = float(query @ coeff_x)
        pred_y = float(query @ coeff_y)
    except np.linalg.LinAlgError:
        return None

    return np.array([pred_x, pred_y, z_med], dtype=xyz_map.dtype)


def fit_plane_from_xyz_map(
    xyz_map: np.ndarray, max_points: int = 20000
) -> Optional[Tuple[np.ndarray, float]]:
    """Fit a plane (n · p + d = 0) from valid xyz samples in camera frame."""
    points = xyz_map.reshape(-1, 3)
    valid_mask = np.isfinite(points).all(axis=1) & (points[:, 2] > 0)
    points = points[valid_mask]
    if points.shape[0] < 3:
        return None

    if points.shape[0] > max_points:
        idx = np.linspace(0, points.shape[0] - 1, num=max_points, dtype=np.int64)
        points = points[idx]

    centroid = points.mean(axis=0)
    demeaned = points - centroid
    try:
        _, _, vh = np.linalg.svd(demeaned, full_matrices=False)
    except np.linalg.LinAlgError:
        return None

    normal = vh[-1]
    norm = np.linalg.norm(normal)
    if norm == 0:
        return None
    normal = normal / norm
    d = -float(normal @ centroid)
    return normal.astype(np.float32), d


def sample_xyz_on_plane(
    u: int,
    v: int,
    plane_params: Tuple[np.ndarray, float],
    intrinsic_matrix: np.ndarray,
) -> Optional[np.ndarray]:
    """Project a pixel ray onto a fitted plane to recover a 3D point."""
    normal, d = plane_params
    if intrinsic_matrix.shape[1] > 3:
        K = intrinsic_matrix[:, :3]
    else:
        K = intrinsic_matrix

    fx = float(K[0, 0])
    fy = float(K[1, 1])
    cx = float(K[0, 2])
    cy = float(K[1, 2])
    if fx == 0.0 or fy == 0.0:
        return None

    direction = np.array([(u - cx) / fx, (v - cy) / fy, 1.0], dtype=np.float32)
    denom = float(normal @ direction)
    if abs(denom) < 1e-6:
        return None

    t = -d / denom
    if t <= 0:
        return None
    return direction * t


class AffordancePredictor:
    """High-level orchestrator for affordance prediction pipeline."""

    def __init__(
        self,
        *,
        default_task: str = DEFAULT_TASK,
        provider: str = "gemini",
        model: str = "gemini-3-flash-preview",
        depth_config: Dict[str, Any] = DEPTH_CONFIG,
        server_ip: str = DEFAULT_SERVER_IP,
        fs_ports: Tuple[int, int] = DEFAULT_FS_PORTS,
        fs_timeout: float = DEFAULT_FS_TIMEOUT,
        gsam_ports: Tuple[int, int] = DEFAULT_GSAM_PORTS,
        gsam_timeout: float = DEFAULT_GSAM_TIMEOUT,
        sam3_ports: Tuple[int, int] = DEFAULT_SAM3_PORTS,
        sam3_timeout: float = DEFAULT_SAM3_TIMEOUT,
    ) -> None:
        self.default_task = default_task
        self.provider = provider
        self.model = model
        self.depth_config = dict(depth_config)
        self.zmax = self.depth_config.get("zmax", float("inf"))
        self.server_ip = server_ip
        self.fs_ports = tuple(fs_ports)
        self.fs_timeout = fs_timeout
        self.gsam_ports = tuple(gsam_ports)
        self.gsam_timeout = gsam_timeout
        self.sam3_ports = tuple(sam3_ports)
        self.sam3_timeout = sam3_timeout

        camera_config = self.depth_config.get("camera_config")
        calib_params_raw = self.depth_config.get("calib_params")
        rec_params_raw = self.depth_config.get("rec_params")
        calib_width = int(self.depth_config.get("calib_width", 640))
        calib_height = int(self.depth_config.get("calib_height", 480))
        if camera_config:
            self.calib_params, self.rec_params = _load_camera_config_params(
                camera_config
            )
        elif isinstance(calib_params_raw, dict) and isinstance(rec_params_raw, dict):
            self.calib_params = _to_numpy_fields(calib_params_raw)
            self.rec_params = _to_numpy_fields(rec_params_raw)
        else:
            raise ValueError(
                "Depth configuration requires either 'camera_config' (YAML path) "
                "or inline mappings for 'calib_params' and 'rec_params'."
            )

        self.intrinsic_matrix = self.rec_params["P1"]
        self.rectifier = Rectifier(
            self.calib_params,
            self.rec_params,
            calib_size=(calib_width, calib_height),
            target_size=(calib_width, calib_height),
        )
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.compliance_predictor = CompliancePredictor(
            provider=self.provider, model=self.model
        )
        self.zmq_clients: Dict[Tuple[str, int, int], ZMQClient] = {}
        self.zmq_clients_lock = Lock()
        self.last_wiping_done = False

    def __del__(self):
        try:
            self.close_zmq_clients()
        except Exception:
            pass

    def close(self) -> None:
        """Release any open ZMQ sockets created by this predictor."""
        self.close_zmq_clients()

    def close_zmq_clients(self) -> None:
        clients_map = getattr(self, "zmq_clients", None)
        clients_lock = getattr(self, "zmq_clients_lock", None)
        if clients_map is None or clients_lock is None:
            return
        with clients_lock:
            clients = list(clients_map.values())
            clients_map.clear()
        for client in clients:
            if client.receiver.socket is not None:
                try:
                    client.poller.unregister(client.receiver.socket)
                except (KeyError, ValueError):
                    pass
            client.sender.close()
            client.receiver.close()

    def get_zmq_client(self, request_port: int, response_port: int) -> ZMQClient:
        key = (self.server_ip, request_port, response_port)
        with self.zmq_clients_lock:
            client = self.zmq_clients.get(key)
            if client is None:
                receiver = ZMQNode(type="receiver", port=response_port)
                sender = ZMQNode(type="sender", ip=self.server_ip, port=request_port)
                poller = zmq.Poller()
                poller.register(receiver.socket, zmq.POLLIN)
                client = ZMQClient(
                    sender=sender,
                    receiver=receiver,
                    lock=Lock(),
                    poller=poller,
                )
                self.zmq_clients[key] = client
        return client

    # @profile()
    def send_zmq_request(
        self,
        payload: Dict[str, Any],
        *,
        request_port: int,
        response_port: int,
        timeout_s: float,
    ) -> Dict[str, Any]:
        client = self.get_zmq_client(request_port, response_port)
        serialized = pickle.dumps(payload)

        with client.lock:
            # Drop any stale responses from earlier requests before issuing a new one.
            client.receiver.get_msg()
            client.sender.socket.send(serialized, flags=0)

            deadline = time.time() + timeout_s
            while True:
                remaining = deadline - time.time()
                if remaining <= 0:
                    break
                timeout_ms = max(int(remaining * 1000), 1)
                events = dict(client.poller.poll(timeout_ms))
                if (
                    client.receiver.socket in events
                    and events[client.receiver.socket] & zmq.POLLIN
                ):
                    response = client.receiver.get_msg()
                    if response is not None:
                        return response

        raise TimeoutError(
            f"No response from server on port {response_port} within {timeout_s} seconds"
        )

    def request_foundation_stereo(
        self,
        left_image: np.ndarray,
        right_image: np.ndarray,
        timeout_s: Optional[float] = None,
        skip_rectify: bool = False,
    ) -> Dict[str, Any]:
        request_port, response_port = self.fs_ports
        timeout = timeout_s if timeout_s is not None else self.fs_timeout
        payload = {
            "model": "fs",
            "data": {
                "left_image": left_image,
                "right_image": right_image,
            },
            "config": {
                "remove_invisible": True,
                "return_all": True,
                "skip_rectify": skip_rectify,
                "calibration": {
                    "calib_params": self.calib_params,
                    "rec_params": self.rec_params,
                    "calib_width": int(self.depth_config.get("calib_width", 640)),
                    "calib_height": int(self.depth_config.get("calib_height", 480)),
                },
            },
        }
        if skip_rectify:
            payload["config"]["pre_rectified"] = True

        response = self.send_zmq_request(
            payload,
            request_port=request_port,
            response_port=response_port,
            timeout_s=timeout,
        )

        if response.get("status") != "ok":
            raise RuntimeError(f"Foundation Stereo error: {response.get('error')}")

        result = response.get("result") or {}
        depth = result.get("depth")
        if depth is None:
            raise RuntimeError("Foundation Stereo response missing depth data")

        return {
            "depth": depth,
            # "disparity": result.get("disparity"),
            # "rectified_left": result.get("rectified_left"),
            # "rectified_right": result.get("rectified_right"),
            # "original_left": result.get("original_left"),
            # "original_right": result.get("original_right"),
        }

    def request_grounded_sam2(
        self,
        image: np.ndarray,
        text_prompt: str,
        timeout_s: Optional[float] = None,
    ) -> Dict[str, Any]:
        request_port, response_port = self.gsam_ports
        timeout = timeout_s if timeout_s is not None else self.gsam_timeout
        payload = {
            "model": "gsam2",
            "data": {
                "image": image,
            },
            "config": {
                "text_prompt": text_prompt,
                "box_threshold": 0.35,
                "text_threshold": 0.25,
                "annotate": False,
            },
        }

        response = self.send_zmq_request(
            payload,
            request_port=request_port,
            response_port=response_port,
            timeout_s=timeout,
        )

        if response.get("status") != "ok":
            raise RuntimeError(f"Grounded SAM2 error: {response.get('error')}")

        return response.get("result") or {}

    def request_sam3(
        self,
        image: np.ndarray,
        text_prompt: str,
        timeout_s: Optional[float] = None,
        return_masks: bool = True,
    ) -> Dict[str, Any]:
        """Send a SAM3 segmentation request over ZMQ."""
        request_port, response_port = self.sam3_ports
        timeout = timeout_s if timeout_s is not None else self.sam3_timeout
        payload = {
            "model": "sam3",
            "data": {
                "image": image,
            },
            "config": {
                "text_prompt": text_prompt,
                "return_masks": return_masks,
            },
        }

        response = self.send_zmq_request(
            payload,
            request_port=request_port,
            response_port=response_port,
            timeout_s=timeout,
        )

        if response.get("status") != "ok":
            raise RuntimeError(f"SAM3 error: {response.get('error')}")

        return response.get("result") or {}

    # @profile()
    def render_workspaces(
        self,
        image: np.ndarray,
        robot_name: str,
        depth_value: float,
        site_names: List[str],
    ) -> Tuple[np.ndarray, List[Tuple[str, Tuple[int, int, int, int]]]]:
        """Overlay workspace rectangles derived from head-frame body roots."""

        ee_pos = EE_LOCATIONS_BY_VARIANT.get(robot_name, {})
        if not ee_pos:
            return image, []

        camera_extrinsics = CAMERA_EXTRINSICS_BY_VARIANT.get(robot_name)
        workspace_size = WORKSPACE_SIZE_BY_VARIANT.get(robot_name)

        if isinstance(workspace_size, dict):
            # ensure only requested sites are used
            workspace_size = {name: workspace_size.get(name) for name in site_names}
        else:
            workspace_size = {name: workspace_size for name in site_names}

        rectangles = compute_workspace_rectangles(
            ee_pos,
            workspace_size,
            camera_extrinsics,
            self.intrinsic_matrix,
            image_shape=image.shape[:2],
            depth_value=depth_value,
            site_names=site_names,
        )
        overlay = draw_workspace_overlay(image, rectangles)
        return overlay, rectangles

    # @profile()
    def predict(
        self,
        left_image: np.ndarray,
        right_image: np.ndarray,
        robot_name: str,
        site_names: List[str],
        is_wiping: bool = True,
        output_dir: Optional[str] = None,
        object_label: Optional[str] = None,
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Run the affordance prediction pipeline end-to-end.

        Args:
            left_image: Raw left camera image.
            right_image: Raw right camera image.
            output_dir: Optional directory for dumping debug artifacts.
            object_label: Optional semantic label describing the region we want the
                model to act on.
            robot_variant: Robot variant key to choose camera extrinsics/workspace
                size defaults. Workspace rectangles are derived from the module-level
                EE_LOCATIONS_BY_VARIANT for the chosen variant, if populated.
        """

        if not isinstance(left_image, np.ndarray) or left_image.ndim not in (2, 3):
            raise ValueError("left_image must be a numpy array")

        # Reset status before running a new prediction.
        self.last_wiping_done = False
        debug = output_dir is not None
        if debug:
            os.makedirs(output_dir, exist_ok=True)

        def write_image(name: str, image: np.ndarray) -> None:
            if not debug:
                return
            path = os.path.join(output_dir, name)
            if not cv2.imwrite(path, image):
                print(f"Warning: failed to write debug image {path}")

        left_image_uint8 = ensure_uint8_image(left_image)
        right_image_uint8 = ensure_uint8_image(right_image)
        write_image("left_raw.png", left_image_uint8)
        write_image("right_raw.png", right_image_uint8)

        rectified_left, rectified_right = self.rectifier.rectify(
            left_image_uint8, right_image_uint8
        )
        write_image("left_rectified.png", rectified_left)
        write_image("right_rectified.png", rectified_right)

        object_label_str = (
            str(object_label).strip()
            if object_label is not None and str(object_label).strip()
            else "object"
        )

        if is_wiping:
            segmentation_target = object_label_str + ".whiteboard.vase"
        else:
            segmentation_target = "whiteboard.vase"

        segments = [segment.strip() for segment in object_label_str.split(".")]
        target_priority = [segment for segment in segments if segment]

        sam_result = self.request_sam3(
            rectified_left, segmentation_target, timeout_s=self.sam3_timeout
        )
        try:
            annotation, segmentation_mask, left_union_mask = postprocess_sam_result(
                sam_result, target_priority, rectified_left.shape[:2]
            )
        except RuntimeError as exc:
            target_spec_for_logging = ", ".join(target_priority)
            print(
                "[AffordancePredictor] No annotation found for "
                f"[{target_spec_for_logging}]: {exc}. Returning no affordance."
            )
            if is_wiping:
                self.last_wiping_done = True
            return None

        if segmentation_mask.shape != rectified_left.shape[:2]:
            segmentation_mask = cv2.resize(
                segmentation_mask,
                (rectified_left.shape[1], rectified_left.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

        annotation_overlay = draw_annotation_overlay(rectified_left, annotation)
        write_image("annotation_overlay.png", annotation_overlay)

        segmentation_mask, removed, total, max_dist = trim_mask_by_center(
            segmentation_mask
        )
        if removed > 0:
            print(
                "[AffordancePredictor] Trimmed segmentation mask: removed "
                f"{removed}/{total} pixels beyond {max_dist:.1f}px from center."
            )

        segmentation_mask_visual = (segmentation_mask * 255).astype(np.uint8)
        write_image("segmentation_mask.png", segmentation_mask_visual)

        if is_wiping:
            if "whiteboard" in annotation["label"] or "vase" in annotation["label"]:
                print(
                    "[AffordancePredictor] Wiping done: whiteboard/vase detected in annotation."
                )
                self.last_wiping_done = True
                return None

            mask_pixel_count = int(np.count_nonzero(segmentation_mask))
            if mask_pixel_count < MIN_MASK_PIXELS:
                print(
                    f"[AffordancePredictor] Wiping done: segmentation mask has too few pixels ({mask_pixel_count} < {MIN_MASK_PIXELS})."
                )
                self.last_wiping_done = True
                return None

            ys, xs = np.nonzero(segmentation_mask)
            mask_bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
            rect_map = {name: {"bbox": mask_bbox} for name in site_names}
            candidate_point_groups = prepare_candidate_points(
                rect_map,
                rectified_left.shape[:2],
                site_names,
                grid_size=20,
                bbox_padding=2,
            )
            depth_result = None
            compliance_image = rectified_left
        else:
            depth_result = self.request_foundation_stereo(
                rectified_left,
                rectified_right,
                timeout_s=self.fs_timeout,
                skip_rectify=True,
            )
            depth_map = depth_result.get("depth")
            if depth_map is not None and left_union_mask is not None:
                depth_map = depth_map * (left_union_mask > 0)

            workspace_depth = float(np.median(depth_map[depth_map > 0]))
            compliance_image, workspace_rects = self.render_workspaces(
                rectified_left, robot_name, workspace_depth, site_names=site_names
            )
            write_image("compliance_workspace.png", compliance_image)
            rect_map = {name: {"bbox": rect} for name, rect in workspace_rects}
            candidate_point_groups = prepare_candidate_points(
                rect_map, rectified_left.shape[:2], site_names, grid_size=20
            )

        print(
            "Predicting affordance for task "
            f"{self.default_task} on '{object_label_str}'."
        )
        if is_wiping:
            target_object_label = object_label_str
        else:
            object_list = [object.strip() for object in object_label_str.split(".")]
            if len(object_list) > len(site_names):
                object_list = object_list[: len(site_names)]
            elif len(object_list) < len(site_names):
                object_list += [object_list[-1]] * (len(site_names) - len(object_list))

            target_object_label = ", ".join(
                [f"{site}: {obj}" for site, obj in zip(site_names, object_list)]
            )

        compliance_future = self.executor.submit(
            self.compliance_predictor.predict_compliance,
            compliance_image,
            self.default_task,
            target_object_label,
            candidate_point_groups=candidate_point_groups,
        )

        # Obtain a right-image mask separately (left/right views differ).
        right_union_mask = None
        try:
            _, _, right_union_mask = postprocess_sam_result(
                self.request_sam3(
                    rectified_right, segmentation_target, timeout_s=self.sam3_timeout
                ),
                target_priority,
                rectified_right.shape[:2],
            )
        except Exception as exc:  # pragma: no cover - diagnostic aid
            print(f"[AffordancePredictor] Warning: right-view SAM failed: {exc}")

        if left_union_mask is not None:
            write_image(
                "left_union_mask.png", (left_union_mask > 0).astype(np.uint8) * 255
            )
        if right_union_mask is not None:
            write_image(
                "right_union_mask.png", (right_union_mask > 0).astype(np.uint8) * 255
            )

        if depth_result is None:
            depth_result = self.request_foundation_stereo(
                rectified_left,
                rectified_right,
                timeout_s=self.fs_timeout,
                skip_rectify=True,
            )
            depth_map = depth_result.get("depth")
            if depth_map is not None and left_union_mask is not None:
                depth_map = depth_map * (left_union_mask > 0)

        overlay_image = draw_points_overlay(
            rectified_left, grouped=candidate_point_groups
        )
        write_image("candidate_points.png", overlay_image)

        xyz_map: Optional[np.ndarray] = None

        if depth_map is not None:
            xyz_map = depth_to_xyzmap(depth_map, self.intrinsic_matrix, zmin=0.0)

        if debug and depth_map is not None:
            depth_vis = vis_disparity(
                depth_map,
                min_val=0,
                max_val=self.zmax,
                invalid_upper_thres=self.zmax,
                invalid_bottom_thres=0.0,
            )
            write_image("depth_map_vis.png", depth_vis)

        prompt, compliance_result = compliance_future.result()
        if not compliance_result:
            print("Prediction failed.")
            return None

        if debug:
            with open(os.path.join(output_dir, "prompt.txt"), "w") as f:
                f.write(prompt)

            serializable_result = {}
            for key, value in compliance_result.items():
                if isinstance(value, np.ndarray):
                    serializable_result[key] = value.tolist()
                else:
                    serializable_result[key] = value

            with open(os.path.join(output_dir, "prediction.json"), "w") as f:
                json.dump(serializable_result, f, indent=2)

            contact_vis = draw_points_overlay(
                rectified_left, grouped=compliance_result, annotate=True
            )
            write_image("contact_points_overlay.png", contact_vis)

        contact_coords = np.concatenate(list(compliance_result.values()), axis=0)
        contact_group_indices = {}
        idx = 0
        for group_key, coords in compliance_result.items():
            contact_group_indices[group_key] = slice(idx, idx + len(coords))
            idx += len(coords)

        if xyz_map is not None:
            height, width = xyz_map.shape[:2]
            contact_points_3d_list: List[List[float]] = []
            plane_params: Optional[Tuple[np.ndarray, float]] = None
            for u, v in contact_coords:
                if v >= height or u >= width:
                    print(f"Warning: pixel ({u}, {v}) is out of bounds")
                    continue
                xyz = xyz_map[v, u]
                if xyz[2] <= 0:
                    print(
                        f"Info: missing depth at pixel ({u}, {v}), projecting to plane"
                    )
                    if plane_params is None:
                        plane_params = fit_plane_from_xyz_map(xyz_map)
                        if plane_params is None:
                            raise RuntimeError(
                                "Unable to fit plane from xyz map for missing depth."
                            )
                    xyz = sample_xyz_on_plane(
                        int(u), int(v), plane_params, self.intrinsic_matrix
                    )
                    if xyz is None or xyz[2] <= 0:
                        raise RuntimeError(
                            f"Unable to sample plane depth at pixel ({u}, {v})"
                        )

                contact_points_3d_list.append(xyz)

            contact_points_3d = np.array(contact_points_3d_list, dtype=np.float32)
            merged_pcd, contact_indices = merge_point_cloud(
                rectified_left, xyz_map, contact_points_3d, zmax=self.zmax
            )
        else:
            contact_points_3d = np.empty((0, 3), dtype=np.float32)
            merged_pcd = None
            contact_indices = []

        contact_normals = np.empty((0, 3), dtype=np.float32)
        if merged_pcd is not None and np.asarray(merged_pcd.points).shape[0] >= 3:
            merged_pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30),
                fast_normal_computation=True,
            )
            if contact_indices:
                normals_camera = np.asarray(merged_pcd.normals, dtype=np.float32)
                contact_normals = normals_camera[contact_indices]
                # Ensure normals point toward the camera: flip if pointing away (z > 0).
                flip_mask = contact_normals[:, 2] > 0
                contact_normals[flip_mask] *= -1.0

        if debug and merged_pcd is not None:
            pcd_filename = os.path.join(output_dir, "point_cloud.ply")
            o3d.io.write_point_cloud(pcd_filename, merged_pcd)
            print(f"Saved point cloud to: {pcd_filename}")

        contact_points_3d_group = {}
        contact_normals_group = {}
        for group_key, indices in contact_group_indices.items():
            contact_points_3d_group[group_key] = contact_points_3d[indices]
            contact_normals_group[group_key] = contact_normals[indices]

        return contact_points_3d_group, contact_normals_group
