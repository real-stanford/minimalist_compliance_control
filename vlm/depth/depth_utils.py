"""Utility functions for depth processing, point cloud conversion, and stereo rectification.

This module provides core utilities for depth map processing including conversion
to point clouds, stereo rectification, disparity visualization, and image padding.
"""

import cv2
import numpy as np
import open3d as o3d


def to_o3d_cloud(points, colors=None, normals=None):
    """Convert point arrays to Open3D PointCloud with optional colors and normals."""
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    if colors is not None:
        if colors.max() > 1:
            colors = colors / 255.0
        cloud.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
    if normals is not None:
        cloud.normals = o3d.utility.Vector3dVector(normals.astype(np.float64))
    return cloud


def depth_to_xyzmap(depth: np.ndarray, K, uvs: np.ndarray = None, zmin=0.0):
    """Convert depth map to 3D coordinate map using camera intrinsics."""
    H, W = depth.shape[:2]
    if uvs is None:
        vs, us = np.meshgrid(
            np.arange(0, H), np.arange(0, W), sparse=False, indexing="ij"
        )
        vs = vs.reshape(-1)
        us = us.reshape(-1)
    else:
        uvs = uvs.round().astype(int)
        us = uvs[:, 0]
        vs = uvs[:, 1]
    zs = depth[vs, us]
    xs = (us - K[0, 2]) * zs / K[0, 0]
    ys = (vs - K[1, 2]) * zs / K[1, 1]
    pts = np.stack((xs.reshape(-1), ys.reshape(-1), zs.reshape(-1)), 1)  # (N,3)
    xyz_map = np.zeros((H, W, 3), dtype=np.float32)
    xyz_map[vs, us] = pts

    invalid_mask = depth < zmin
    if invalid_mask.any():
        xyz_map[invalid_mask] = 0

    return xyz_map


def vis_disparity(
    disp,
    min_val=None,
    max_val=None,
    invalid_upper_thres=np.inf,
    invalid_bottom_thres=-np.inf,
    color_map=cv2.COLORMAP_TURBO,
    cmap=None,
    no_color=False,
    other_output={},
):
    """
    @disp: np array (H,W)
    @invalid_upper_thres: > thres is invalid
    @invalid_bottom_thres: < thres is invalid
    """
    disp = disp.copy()
    H, W = disp.shape[:2]
    invalid_mask = (disp >= invalid_upper_thres) | (disp <= invalid_bottom_thres)
    if (invalid_mask == 0).sum() == 0:
        other_output["min_val"] = None
        other_output["max_val"] = None
        return np.zeros((H, W, 3)).astype(np.uint8)
    if min_val is None:
        min_val = disp[invalid_mask == 0].min()
    if max_val is None:
        max_val = disp[invalid_mask == 0].max()
    other_output["min_val"] = min_val
    other_output["max_val"] = max_val
    vis = ((disp - min_val) / (max_val - min_val)).clip(0, 1) * 255
    if no_color:
        vis = cv2.cvtColor(vis.clip(0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    elif cmap is None:
        vis = cv2.applyColorMap(vis.clip(0, 255).astype(np.uint8), color_map)[..., ::-1]
    else:
        vis = (np.array(cmap(vis.astype(np.uint8)))[..., :3] * 255)[:, :, ::-1].astype(
            np.uint8
        )

    # if invalid_mask.any():
    #     # Clip invalid inf values to white
    #     vis[invalid_mask] = 255

    # Crop the leftmost invalid 0 depth values
    # vis = vis[:, 10:]

    return vis.astype(np.uint8)


def get_rectification_maps(calib_params, rec_params, image_size):
    """Generate stereo rectification maps from calibration parameters."""
    K1, D1, K2, D2, _, _ = (
        calib_params["K1"],
        calib_params["D1"],
        calib_params["K2"],
        calib_params["D2"],
        calib_params["R"],
        calib_params["T"],
    )

    R1, R2, P1, P2, _ = (
        rec_params["R1"],
        rec_params["R2"],
        rec_params["P1"],
        rec_params["P2"],
        rec_params["Q"],
    )
    map1_left, map2_left = cv2.fisheye.initUndistortRectifyMap(
        K1, D1, R1, P1, image_size, cv2.CV_16SC2
    )
    map1_right, map2_right = cv2.fisheye.initUndistortRectifyMap(
        K2, D2, R2, P2, image_size, cv2.CV_16SC2
    )

    return map1_left, map2_left, map1_right, map2_right


def pad_images_np(img0, img1, divis_by=32):
    """Pad stereo image pair to be divisible by specified value."""
    _, _, h, w = img0.shape
    new_h = ((h + divis_by - 1) // divis_by) * divis_by
    new_w = ((w + divis_by - 1) // divis_by) * divis_by
    pad_h = new_h - h
    pad_w = new_w - w

    img0_padded = np.pad(
        img0, ((0, 0), (0, 0), (0, pad_h), (0, pad_w)), mode="constant"
    )
    img1_padded = np.pad(
        img1, ((0, 0), (0, 0), (0, pad_h), (0, pad_w)), mode="constant"
    )
    return img0_padded, img1_padded, (pad_h, pad_w)


def unpad_image_np(img, pad_shape, original_shape):
    """Remove padding from image to restore original dimensions."""
    pad_h, pad_w = pad_shape
    H, W = original_shape
    return img[:, :, :H, :W]
