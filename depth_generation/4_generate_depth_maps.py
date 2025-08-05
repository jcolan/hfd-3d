#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Depth Map and Projected View Generation
=======================================
VERSION: 1.0

Description:
    This script generates 16-bit depth maps and the corresponding projected color views
    for a range of frames. It uses the final optimized camera poses and the master
    point cloud to create these data products.

Usage:
    python 4_generate_depth_maps.py --dataset DATASET_NAME --frame_start START --frame_end END [options]

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

# Assuming modules are accessible from the project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utilities.projection import project_point_cloud

# --- Logging and Configuration ---
def setup_logger():
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

def load_optimized_pose(yaml_path):
    """Loads an optimized camera pose from a .yml file."""
    with open(yaml_path, "r") as file:
        data = yaml.safe_load(file)
    pos = data["pose"]["position"]
    quat = data["pose"]["orientation"]
    
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_quat([quat["x"], quat["y"], quat["z"], quat["w"]]).as_matrix()
    matrix[:3, 3] = [pos["x"] * 1000, pos["y"] * 1000, pos["z"] * 1000] # Convert to mm
    return matrix

# --- Main Execution ---

def main():
    parser = argparse.ArgumentParser(description="Generate depth maps and projected views for a range of frames.")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (e.g., 'dataset6').")
    parser.add_argument("--frame_start", type=int, required=True, help="Starting frame number.")
    parser.add_argument("--frame_end", type=int, required=True, help="Ending frame number.")
    parser.add_argument("--quality", type=str, default="standard", choices=['draft', 'standard', 'fine', 'ultra_hd'], help="Quality of the merged point cloud to use.")
    parser.add_argument("--point_size", type=int, default=2, help="Size of the points in the rendered image.")
    parser.add_argument("--near_clip", type=float, default=0.0, help="Near clipping distance in mm.")
    parser.add_argument("--remove_outliers", action="store_true", help="Enable statistical outlier removal.")
    parser.add_argument("--fill_holes", action="store_true", help="Enable morphological closing to fill holes.")
    parser.add_argument("--hole_fill_kernel_size", type=int, default=5, help="Kernel size for hole filling.")
    parser.add_argument("--no_smooth", action="store_true", help="Disable the smoothing and inpainting of the projected view.")
    parser.add_argument("--pose_type", type=str, default="optimized", choices=['initial', 'optimized'], help="Type of pose to use ('initial' or 'optimized').")
    args = parser.parse_args()

    logger.info(f"--- Starting Depth Map Generation for Dataset: {args.dataset} ---")
    logger.info(f"Using '{args.pose_type}' poses.")

    # --- Load Master Point Cloud ---
    pcd_path = f"./raw_data/point_clouds/{args.dataset}/merged/merged_cloud_{args.quality}.ply"
    if not os.path.exists(pcd_path):
        pcd_path = f"./raw_data/point_clouds/{args.dataset}/transformed/pose0_transformed.ply"
        if not os.path.exists(pcd_path):
            logger.error(f"Master point cloud not found at either merged or transformed path.")
            sys.exit(1)
        logger.warning(f"Merged cloud not found. Falling back to: {pcd_path}")

    logger.info(f"Loading master point cloud from: {pcd_path}")
    pcd = o3d.io.read_point_cloud(pcd_path)

    if args.remove_outliers:
        logger.info("Performing statistical outlier removal...")
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        logger.info("Outlier removal complete.")

    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors) if pcd.has_colors() else None

    # --- Camera Intrinsics ---
    camera_matrix = np.array([[1.1573e03, 0, 1.0291e03], [0, 1.1562e03, 5.2774e02], [0, 0, 1]])
    dist_coeffs = np.array([[-0.489, 0.329, -0.001, -0.0001, -0.014]])
    image_size = (1088, 2048)

    # --- Process Frames ---
    processed_count = 0
    for frame_num in range(args.frame_start, args.frame_end + 1):
        logger.info(f"Processing frame {frame_num}...")
        frame_dir = os.path.join("raw_data", "endoscope", args.dataset, f"frame{frame_num}")
        
        pose_path = os.path.join(frame_dir, f"frame{frame_num}_{args.pose_type}.yml")

        if not os.path.exists(pose_path):
            logger.warning(f"Pose not found, skipping: {pose_path}")
            continue

        try:
            camera_pose = load_optimized_pose(pose_path)
            
            projected_image, depth_map = project_point_cloud(
                points, colors, camera_matrix, dist_coeffs, camera_pose, image_size, 
                dataset_name=args.dataset, 
                point_size=args.point_size,
                near_clip=args.near_clip,
                fill_holes=args.fill_holes,
                hole_fill_kernel_size=args.hole_fill_kernel_size,
                smooth=not args.no_smooth
            )

            # --- Save Outputs ---
            output_suffix = f"_merged_{args.quality}_{args.pose_type}"
            depth_filename = os.path.join(frame_dir, f"frame{frame_num}_depth{output_suffix}.png")
            projected_filename = os.path.join(frame_dir, f"frame{frame_num}_projected{output_suffix}.png")

            cv2.imwrite(depth_filename, depth_map)
            logger.debug(f"Saved depth map to: {depth_filename}")
            
            cv2.imwrite(projected_filename, projected_image)
            logger.debug(f"Saved projected view to: {projected_filename}")
            
            processed_count += 1

        except Exception as e:
            logger.error(f"Failed to process frame {frame_num}: {e}", exc_info=True)
            continue

    logger.info(f"--- Generation Complete. Successfully processed {processed_count} frames. ---")

if __name__ == "__main__":
    main()