#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Initial Pose Offset Finder
==========================
VERSION: 1.0

Description:
    This script provides an interactive tool to find the initial global
    transformation offset (rotation and translation) between a 3D point cloud
    and its corresponding 2D endoscope image. By projecting the point cloud
    onto the image and adjusting the pose with sliders, users can visually
    align the two data sources. The resulting offset is saved to a YAML file,
    which serves as the starting point for more precise registration steps.

Usage:
    python 1_find_initial_offset.py --dataset dataset6 --frame 1
"""

import numpy as np
import open3d as o3d
import cv2
import yaml
import os
import sys
from scipy.spatial.transform import Rotation
import argparse
import logging
import time

# Add project root to system path for local module imports.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dataset_config import DATASET_CONFIGS

# --- Logging and Configuration ---
def setup_logger():
    """Sets up a logger for consistent and formatted output."""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

logger = setup_logger()

# --- Core Functions ---

def load_pose(yaml_path):
    """Loads the raw camera pose from a frame's .yml file and converts to mm."""
    with open(yaml_path, "r") as file:
        data = yaml.safe_load(file)
    pos = data["pose"]["position"]
    quat = data["pose"]["orientation"]
    
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_quat([quat["x"], quat["y"], quat["z"], quat["w"]]).as_matrix()
    matrix[:3, 3] = [pos["x"] * 1000, pos["y"] * 1000, pos["z"] * 1000] # Convert meters to mm
    return matrix

def create_circular_mask(image_size, radius=None, dataset_name=None):
    """
    Creates a circular mask to simulate an endoscope's field of view.
    The mask center can be adjusted based on the dataset configuration.
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
):
    """Projects a 3D point cloud onto a 2D image plane using a camera model."""
    R = camera_pose[:3, :3]
    t = camera_pose[:3, 3]
    R_inv = R.T
    t_inv = -R_inv @ t

    # Transform points to camera coordinates and apply near clipping plane.
    points_cam = (R_inv @ points.T + t_inv.reshape(3, 1)).T
    mask_front = points_cam[:, 2] > 20
    points_cam = points_cam[mask_front]
    if colors is not None:
        colors = colors[mask_front]

    if len(points_cam) == 0:
        return np.zeros(image_size + (3,), dtype=np.uint8)

    # Project 3D points to 2D image plane.
    points_cam_meters = points_cam / 1000.0
    points_2d, _ = cv2.projectPoints(
        points_cam_meters, np.zeros(3), np.zeros(3), camera_matrix, dist_coeffs
    )
    points_2d = points_2d.reshape(-1, 2)
    
    # Correct for 180-degree rotation in the camera model.
    center_x, center_y = image_size[1] / 2, image_size[0] / 2
    points_2d = -(points_2d - [center_x, center_y]) + [center_x, center_y]

    rendered_image = np.zeros(image_size + (3,), dtype=np.uint8)
    depth_buffer = np.full(image_size, np.inf)

    # Filter points that are outside the image boundaries.
    mask = (
        (points_2d[:, 0] >= 0) & (points_2d[:, 0] < image_size[1]) &
        (points_2d[:, 1] >= 0) & (points_2d[:, 1] < image_size[0])
    )
    valid_points = points_2d[mask].astype(int)
    valid_depths = points_cam[mask][:, 2]

    if colors is not None:
        valid_colors = colors[mask]
        if valid_colors.dtype == np.float64:
            valid_colors = (valid_colors * 255).astype(np.uint8)
        valid_colors = valid_colors[:, ::-1] # Convert RGB to BGR for OpenCV
    else: # If no colors, use depth to generate a colormap.
        depths_normalized = ((valid_depths - valid_depths.min()) / 
                             (valid_depths.max() - valid_depths.min() + 1e-10) * 255).astype(np.uint8)
        valid_colors = cv2.applyColorMap(depths_normalized.reshape(-1, 1), cv2.COLORMAP_COPPER).squeeze()

    # Render points using a z-buffer for occlusion.
    for point, color, depth in zip(valid_points, valid_colors, valid_depths):
        y1, y2 = max(0, point[1] - point_size), min(image_size[0], point[1] + point_size + 1)
        x1, x2 = max(0, point[0] - point_size), min(image_size[1], point[0] + point_size + 1)
        region = depth_buffer[y1:y2, x1:x2]
        is_closer = depth < region
        rendered_image[y1:y2, x1:x2][is_closer] = color
        region[is_closer] = depth

    # Apply the final circular mask to the rendered image.
    mask_radius = 635
    circular_mask = create_circular_mask(image_size, radius=mask_radius, dataset_name=dataset_name)
    rendered_image[~circular_mask] = 0
    
    return rendered_image

# --- Main Application Logic ---

def main():
    """Main function to run the interactive offset finding tool."""
    parser = argparse.ArgumentParser(description="Interactively find a global manual offset for a dataset.")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (e.g., 'dataset6').")
    parser.add_argument("--frame", type=int, required=True, help="Frame number to use for visualization.")
    parser.add_argument("--pcd_path", type=str, help="Optional path to a specific point cloud file to use for alignment.")
    args = parser.parse_args()

    # --- Setup Paths ---
    endoscope_dir = os.path.join("raw_data", "endoscope", args.dataset)
    raw_pose_path = os.path.join(endoscope_dir, f"frame{args.frame}", f"frame{args.frame}.yml")
    image_path = os.path.join(endoscope_dir, f"frame{args.frame}", f"frame{args.frame}.png")
    offset_file_path = os.path.join(endoscope_dir, "initial_manual_offset.yml")
    pcd_path = args.pcd_path or f"./raw_data/point_clouds/{args.dataset}/merged/merged_cloud_standard.ply"

    # --- Load Data ---
    logger.info(f"Loading data for Dataset: {args.dataset}, Frame: {args.frame}")
    try:
        real_image = cv2.imread(image_path)
        raw_pose = load_pose(raw_pose_path)
        pcd = o3d.io.read_point_cloud(pcd_path)
        points, colors = np.asarray(pcd.points), np.asarray(pcd.colors)
    except Exception as e:
        logger.error(f"Failed to load data: {e}", exc_info=True)
        sys.exit(1)

    # --- Load or Initialize Offset ---
    if os.path.exists(offset_file_path):
        logger.info(f"Loading existing offset from: {offset_file_path}")
        with open(offset_file_path, 'r') as f:
            offset_data = yaml.safe_load(f)
        initial_rotation = offset_data['rotation_xyz_deg']
        initial_translation = offset_data['translation_xyz_mm']
    else:
        logger.info("No existing offset file found. Starting with zero offset.")
        initial_rotation = [0.0, 0.0, 0.0]
        initial_translation = [0.0, 0.0, 0.0]

    # --- Interactive Window Setup ---
    window_name = "Initial Offset Finder"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    def trackbar_callback(val): pass

    cv2.createTrackbar("Rot X", window_name, int(initial_rotation[0] * 10 + 180), 360, trackbar_callback)
    cv2.createTrackbar("Rot Y", window_name, int(initial_rotation[1] * 10 + 180), 360, trackbar_callback)
    cv2.createTrackbar("Rot Z", window_name, int(initial_rotation[2] * 10 + 180), 360, trackbar_callback)
    cv2.createTrackbar("Trans X", window_name, int(initial_translation[0] * 10 + 200), 400, trackbar_callback)
    cv2.createTrackbar("Trans Y", window_name, int(initial_translation[1] * 10 + 200), 400, trackbar_callback)
    cv2.createTrackbar("Trans Z", window_name, int(initial_translation[2] * 10 + 200), 400, trackbar_callback)
    cv2.createTrackbar("Alpha", window_name, 75, 100, trackbar_callback)

    logger.info("Starting interactive session. Press 's' to save the offset, 'q' to quit.")
    camera_matrix = np.array([[1.1573e03, 0, 1.0291e03], [0, 1.1562e03, 5.2774e02], [0, 0, 1]])
    dist_coeffs = np.array([[-0.489, 0.329, -0.001, -0.0001, -0.014]])
    image_size = (real_image.shape[0], real_image.shape[1])

    last_params = {}
    projected_image = np.zeros_like(real_image)

    # --- Main Interactive Loop ---
    while True:
        rx = (cv2.getTrackbarPos("Rot X", window_name) - 180) / 10.0
        ry = (cv2.getTrackbarPos("Rot Y", window_name) - 180) / 10.0
        rz = (cv2.getTrackbarPos("Rot Z", window_name) - 180) / 10.0
        tx = (cv2.getTrackbarPos("Trans X", window_name) - 200) / 10.0
        ty = (cv2.getTrackbarPos("Trans Y", window_name) - 200) / 10.0
        tz = (cv2.getTrackbarPos("Trans Z", window_name) - 200) / 10.0
        alpha = cv2.getTrackbarPos("Alpha", window_name) / 100.0
        
        current_params = {'rot': [rx, ry, rz], 'trans': [tx, ty, tz]}

        # Only re-project if transformation parameters have changed.
        if current_params != last_params:
            # The 'zyx' Euler order is used for creating the rotation matrix.
            R_additional = Rotation.from_euler('zyx', [rz, ry, rx], degrees=True).as_matrix()
            
            # Apply the manual offset to the raw pose.
            final_pose = raw_pose.copy()
            final_pose[:3, :3] = raw_pose[:3, :3] @ R_additional
            final_pose[:3, 3] = raw_pose[:3, 3] + np.array([tx, ty, tz])
            
            projected_image = project_point_cloud(points, colors, camera_matrix, dist_coeffs, final_pose, image_size, args.dataset, point_size=1)
            last_params = current_params
            print(f"\rRot: [{rx:.1f}, {ry:.1f}, {rz:.1f}], Trans: [{tx:.1f}, {ty:.1f}, {tz:.1f}]", end="")

        # Blend the real and projected images for visualization.
        blended_image = cv2.addWeighted(real_image, 1 - alpha, projected_image, alpha, 0)
        cv2.imshow(window_name, blended_image)

        key = cv2.waitKey(20) & 0xFF
        if key == ord('q'):
            logger.info("Quit command received. Closing window.")
            break
        elif key == ord('s'):
            offset_to_save = {
                'rotation_xyz_deg': last_params['rot'],
                'translation_xyz_mm': last_params['trans']
            }
            with open(offset_file_path, 'w') as f:
                yaml.dump(offset_to_save, f, default_flow_style=False)
            logger.info(f"Successfully saved offset to {offset_file_path}")

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
