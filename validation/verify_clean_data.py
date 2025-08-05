#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Clean Data Verification Tool
============================

Description:
    This script provides a visual interface to verify the quality and alignment of 
    the final "clean" dataset. For each specified frame, it performs the following:

    1.  Loads the master point cloud from the `data_clean` directory.
    2.  Loads the corresponding final camera pose (`.yml`).
    3.  Generates a new projected color view and a 16-bit depth map on-the-fly.
    4.  Crops the newly generated data and the existing clean data using a shared bounding box.
    5.  Displays four images side-by-side for immediate visual comparison:
        - Reconstructed (Projected) View (New) vs. Reconstructed (Existing)
        - Depth Map (New) vs. Depth Map (Existing)

    This allows for a quick manual check to ensure that the stored poses accurately 
    recreate the provided depth and color data, confirming the integrity of the clean dataset.

Usage:
    python validation/verify_clean_data.py --dataset DATASET_NAME --frame_start START --frame_end END

Controls:
    - Press 'q' to quit the visualization.
    - Press any other key to advance to the next frame.
"""

import os
import numpy as np
import open3d as o3d
import cv2
import yaml
from scipy.spatial.transform import Rotation
import argparse
import sys
import logging

from math import sqrt

# Add project root to path to allow module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utilities.projection import project_point_cloud
from dataset_config import DATASET_CONFIGS

# --- Logging and Configuration ---
def setup_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

logger = setup_logger()

# --- Core Functions ---

def load_pose(yaml_path):
    """Loads a camera pose from a .yml file."""
    with open(yaml_path, "r") as file:
        data = yaml.safe_load(file)
    pos = data["pose"]["position"]
    quat = data["pose"]["orientation"]
    
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_quat([quat["x"], quat["y"], quat["z"], quat["w"]]).as_matrix()
    matrix[:3, 3] = [pos["x"] * 1000, pos["y"] * 1000, pos["z"] * 1000] # Convert to mm
    return matrix

def crop_image(image, center_x, center_y, size, output_size=(448, 448)):
    """Crops an image to a square region and resizes it."""
    if image is None:
        # Return an appropriately sized black image if input is None
        if len(output_size) == 2: # Grayscale
             return np.zeros(output_size, dtype=np.uint8)
        else: # Color
             return np.zeros((output_size[1], output_size[0], 3), dtype=np.uint8)


    half_size = size // 2
    x1, y1 = center_x - half_size, center_y - half_size
    x2, y2 = center_x + half_size, center_y + half_size

    if not (0 <= y1 < y2 <= image.shape[0] and 0 <= x1 < x2 <= image.shape[1]):
        logger.error(f"Crop coordinates [{y1}:{y2}, {x1}:{x2}] are out of bounds for image size {image.shape}.")
        # Return a black image of the correct size if crop is invalid
        if len(image.shape) == 3:
            return np.zeros((output_size[1], output_size[0], 3), dtype=np.uint8)
        else:
            return np.zeros(output_size, dtype=image.dtype)


    cropped_img = image[y1:y2, x1:x2]
    return cv2.resize(cropped_img, output_size, interpolation=cv2.INTER_AREA)

def normalize_depth_for_display(depth_map_16bit, output_size=(448, 448)):
    """
    Converts a 16-bit depth map to a displayable 8-bit BGR image
    by directly scaling the values. This preserves the overall darkness
    of the original depth data.
    """
    if depth_map_16bit is None or not np.any(depth_map_16bit):
        return np.zeros((output_size[1], output_size[0], 3), dtype=np.uint8)

    # Directly scale the 16-bit data (0-65535) to 8-bit (0-255)
    # This is done by integer division by 256 (or bit-shifting right by 8)
    depth_8u = (depth_map_16bit / 256).astype(np.uint8)
    
    # Convert the 8-bit grayscale to a 3-channel BGR image for display
    return cv2.cvtColor(depth_8u, cv2.COLOR_GRAY2BGR)


# --- Main Execution ---

def main():
    parser = argparse.ArgumentParser(description="Visually verify the alignment of clean data.")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (e.g., 'dataset6').")
    parser.add_argument("--frame_start", type=int, required=True, help="Starting frame number.")
    parser.add_argument("--frame_end", type=int, required=True, help="Ending frame number.")
    args = parser.parse_args()

    logger.info(f"--- Starting Verification for Dataset: {args.dataset} ---")

    # Define the output resolution for display images
    output_res = (448, 448)

    # --- Load Master Point Cloud from data_clean ---
    pcd_path = f"./data_clean/point_clouds/{args.dataset}/merged/point_cloud.ply"
    if not os.path.exists(pcd_path):
        logger.error(f"Clean point cloud not found: {pcd_path}")
        sys.exit(1)

    logger.info(f"Loading clean point cloud from: {pcd_path}")
    pcd = o3d.io.read_point_cloud(pcd_path)

    logger.info("Performing statistical outlier removal...")
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    logger.info("Outlier removal complete.")
    
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors) if pcd.has_colors() else None

    # --- Camera Intrinsics (should be consistent) ---
    camera_matrix = np.array([[1.1573e03, 0, 1.0291e03], [0, 1.1562e03, 5.2774e02], [0, 0, 1]])
    dist_coeffs = np.array([[-0.489, 0.329, -0.001, -0.0001, -0.014]])
    image_size = (1088, 2048)

    # --- Get Cropping Parameters from Config ---
    config = DATASET_CONFIGS.get(args.dataset, DATASET_CONFIGS["default"])
    center_x, center_y = config["mask_center"]
    radius = 635
    square_size = int((2 * radius) / sqrt(2))

    # --- Process and Verify Frames ---
    for frame_num in range(args.frame_start, args.frame_end + 1):
        logger.info(f"Verifying frame {frame_num}...")
        
        frame_dir = os.path.join("depth_data", "endoscope", args.dataset, f"frame{frame_num}")
        pose_path = os.path.join(frame_dir, f"frame{frame_num}.yml")
        existing_depth_path = os.path.join(frame_dir, f"frame{frame_num}_depth.png")
        existing_recon_path = os.path.join(frame_dir, f"frame{frame_num}_projected.png")

        if not all(os.path.exists(p) for p in [pose_path, existing_depth_path, existing_recon_path]):
            logger.warning(f"Missing one or more clean data files for frame {frame_num}. Skipping.")
            continue

        try:
            # --- Generate New Data On-the-Fly ---
            camera_pose = load_pose(pose_path)
            
            # Assuming project_point_cloud returns (color_image, depth_map_16bit)
            new_recon, new_depth_16bit = project_point_cloud(
                points, colors, camera_matrix, dist_coeffs, camera_pose, image_size,
                dataset_name=args.dataset, 
                point_size=0, # Use a smaller point size for clarity
                near_clip=30,
                fill_holes=True,
                hole_fill_kernel_size=4,
                smooth=True
            )

            # --- Load Existing Data ---
            existing_recon = cv2.imread(existing_recon_path)
            existing_depth_16bit = cv2.imread(existing_depth_path, cv2.IMREAD_UNCHANGED)

            # --- Crop and Prepare for Visualization ---
            cropped_new_recon = crop_image(new_recon, center_x, center_y, square_size, output_size=output_res)
            resized_existing_recon = cv2.resize(existing_recon, output_res, interpolation=cv2.INTER_AREA)
            
            cropped_new_depth = crop_image(new_depth_16bit, center_x, center_y, square_size, output_size=output_res)
            resized_existing_depth = cv2.resize(existing_depth_16bit, output_res, interpolation=cv2.INTER_AREA)

            # --- Display for Comparison ---
            display_new_depth = normalize_depth_for_display(cropped_new_depth, output_size=output_res)
            display_existing_depth = normalize_depth_for_display(resized_existing_depth, output_size=output_res)

            # Create labels for each image
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.0 # Increased font size for better readability on larger images
            font_color = (255, 255, 255)
            thickness = 2
            cv2.putText(cropped_new_recon, 'New Recon', (15, 35), font, font_scale, font_color, thickness)
            cv2.putText(resized_existing_recon, 'Existing Recon', (15, 35), font, font_scale, font_color, thickness)
            cv2.putText(display_new_depth, 'New Depth', (15, 35), font, font_scale, font_color, thickness)
            cv2.putText(display_existing_depth, 'Existing Depth', (15, 35), font, font_scale, font_color, thickness)

            # Combine images into a single view
            top_row = np.hstack((cropped_new_recon, resized_existing_recon))
            bottom_row = np.hstack((display_new_depth, display_existing_depth))
            comparison_view = np.vstack((top_row, bottom_row))

            window_name = f'Verification for {args.dataset} - Frame {frame_num}'
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.imshow(window_name, comparison_view)
            # Adjust window size to fit the new 896x896 comparison view
            cv2.resizeWindow(window_name, 960, 960) 
            
            key = cv2.waitKey(0)
            if key == ord('q'):
                logger.info("Quit signal received. Exiting.")
                break
            cv2.destroyAllWindows()

        except Exception as e:
            logger.error(f"Failed to verify frame {frame_num}: {e}", exc_info=True)
            cv2.destroyAllWindows()
            continue
            
    logger.info(f"--- Verification Complete for Dataset: {args.dataset} ---")

if __name__ == "__main__":
    main()