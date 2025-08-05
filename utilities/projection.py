"""
Core Point Cloud Projection Module
==================================

Description:
    This script is a central library module and not intended for direct execution.
    It provides the core `project_point_cloud` function, which is the fundamental
    engine for projecting a 3D point cloud into a 2D image plane.

    This function handles:
    - Transformation of 3D points from world space to camera space.
    - Projection of 3D camera points to 2D image coordinates using camera intrinsics.
    - Z-buffering to handle occlusions correctly.
    - Optional post-processing steps like smoothing, hole filling, and near clipping.
    - Generation of both a projected color image and a 16-bit depth map.

    It is imported and used by numerous scripts throughout the `depth_generation`
    and `validation` pipelines.

"""
import numpy as np
import cv2
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dataset_config import DATASET_CONFIGS

def create_circular_mask(image_size, radius=None, dataset_name=None):
    """
    Create a circular mask for an image to simulate endoscope view.
    The center of the mask can be adjusted based on the dataset name.
    """
    h, w = image_size
    config = DATASET_CONFIGS.get(dataset_name, DATASET_CONFIGS["default"])
    center = config["mask_center"]

    if radius is None:
        radius = min(center[0], center[1], w - center[0], h - center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)
    mask = dist_from_center <= radius
    return mask

def project_point_cloud(
    points,
    colors,
    camera_matrix,
    dist_coeffs,
    camera_pose,
    image_size,
    dataset_name=None,
    point_size=2,
    near_clip=0.0,
    fill_holes=False,
    hole_fill_kernel_size=5,
    smooth=True,
):
    """Project 3D point cloud onto a 2D image plane using a camera model."""
    R = camera_pose[:3, :3]
    t = camera_pose[:3, 3]
    R_inv = R.T
    t_inv = -R_inv @ t

    points_cam = (R_inv @ points.T + t_inv.reshape(3, 1)).T
    
    # Near clipping
    depths = points_cam[:, 2]
    clip_mask = depths > near_clip
    points_cam = points_cam[clip_mask]
    depths = depths[clip_mask]

    if colors is not None:
        colors = colors[clip_mask]

    if len(points_cam) == 0:
        return np.zeros(image_size + (3,), dtype=np.uint8), np.zeros(image_size, dtype=np.uint16)

    points_cam_meters = points_cam / 1000.0
    points_2d, _ = cv2.projectPoints(
        points_cam_meters, np.zeros(3), np.zeros(3), camera_matrix, dist_coeffs
    )
    points_2d = points_2d.reshape(-1, 2)
    center_x, center_y = image_size[1] / 2, image_size[0] / 2
    points_2d = -(points_2d - [center_x, center_y]) + [center_x, center_y]

    rendered_image = np.zeros(image_size + (3,), dtype=np.uint8)
    depth_buffer = np.full(image_size, np.inf)

    mask = (
        (points_2d[:, 0] >= 0)
        & (points_2d[:, 0] < image_size[1])
        & (points_2d[:, 1] >= 0)
        & (points_2d[:, 1] < image_size[0])
    )
    valid_points = points_2d[mask].astype(int)
    valid_depths = depths[mask]

    if colors is not None:
        valid_colors = colors[mask]
        if valid_colors.dtype == np.float64:
            valid_colors = (valid_colors * 255).astype(np.uint8)
        valid_colors = valid_colors[:, ::-1] # RGB to BGR
    else:
        depths_normalized = (
            (valid_depths - valid_depths.min())
            / (valid_depths.max() - valid_depths.min() + 1e-10)
            * 255
        ).astype(np.uint8)
        valid_colors = cv2.applyColorMap(
            depths_normalized.reshape(-1, 1), cv2.COLORMAP_COPPER
        ).squeeze()

    for point, color, depth in zip(valid_points, valid_colors, valid_depths):
        y1, y2 = max(0, point[1] - point_size), min(image_size[0], point[1] + point_size + 1)
        x1, x2 = max(0, point[0] - point_size), min(image_size[1], point[0] + point_size + 1)
        region = depth_buffer[y1:y2, x1:x2]
        is_closer = depth < region
        rendered_image[y1:y2, x1:x2][is_closer] = color
        region[is_closer] = depth

    mask_radius = 635
    circular_mask = create_circular_mask(
        image_size, radius=mask_radius, dataset_name=dataset_name
    )
    rendered_image[~circular_mask] = 0
    
    if smooth:
        non_zero_mask = (rendered_image > 0).any(axis=2)
        kernel = np.ones((3, 3), np.uint8)
        dilated_mask = cv2.dilate(non_zero_mask.astype(np.uint8), kernel, iterations=1)
        smoothed = cv2.bilateralFilter(rendered_image, 5, 75, 75)
        smoothed = cv2.fastNlMeansDenoisingColored(smoothed, None, 10, 10, 7, 21)
        for c in range(3):
            channel = rendered_image[..., c]
            valid_pixels = channel > 0
            if valid_pixels.any():
                channel = cv2.inpaint(channel, (1 - valid_pixels).astype(np.uint8), 3, cv2.INPAINT_TELEA)
                smoothed[..., c] = channel
        rendered_image = np.where(dilated_mask[..., None], smoothed, rendered_image)
        rendered_image[~circular_mask] = 0

    depth_map = depth_buffer.copy()
    depth_map[depth_map == np.inf] = 0
    depth_map = np.clip(depth_map, 0, 300) # Clip depth to 300mm
    depth_map = ((depth_map / 300.0) * 65535).astype(np.uint16) # Normalize to 16-bit
    depth_map[~circular_mask] = 0

    if fill_holes:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (hole_fill_kernel_size, hole_fill_kernel_size))
        depth_map = cv2.morphologyEx(depth_map, cv2.MORPH_CLOSE, kernel)

    return rendered_image, depth_map