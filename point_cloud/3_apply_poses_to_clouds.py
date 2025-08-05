#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Apply Final Poses to Point Clouds
===================================
VERSION: 1.0

Description:
    This script is the third step in the point cloud processing pipeline. It takes
    the raw point clouds and applies their corresponding final transformation poses.

    The script intelligently finds the best available pose for each point cloud,
    prioritizing manually refined poses (e.g., `pose_1_refined.yaml`) over the
    initial automated poses (e.g., `pose_1_initial.yaml`). The resulting
    transformed point clouds are saved to a new directory, ready for merging.

Usage:
    python point_cloud/3_apply_poses_to_clouds.py --dataset <dataset_name>

"""
import open3d as o3d
import numpy as np
import yaml
import os
import glob
import sys
import argparse
import logging

# --- Logging Configuration ---
def setup_logger():
    """Sets up a logger for consistent output format."""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
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

def read_transform_yml(file_path):
    """Reads a 4x4 transformation matrix from a YAML file."""
    try:
        with open(file_path, "r") as f:
            data = yaml.safe_load(f)
        return np.array(data["FloatMatrix"]["Data"])
    except Exception as e:
        logger.error(f"Error reading transformation file {file_path}: {e}")
        raise

def find_best_pose_file(transform_dir, pose_num):
    """Finds the best available pose file, prioritizing refined over initial."""
    refined_path = os.path.join(transform_dir, f'pose_{pose_num}_refined.yaml')
    initial_path = os.path.join(transform_dir, f'pose_{pose_num}_initial.yaml')

    if os.path.exists(refined_path):
        logger.debug(f"Found refined pose for pose {pose_num}: {refined_path}")
        return refined_path
    elif os.path.exists(initial_path):
        logger.debug(f"Found initial pose for pose {pose_num}: {initial_path}")
        return initial_path
    else:
        logger.warning(f"No suitable pose file found for pose {pose_num}.")
        return None

# --- Main Execution ---

def main():
    """Main function to apply final poses to raw point clouds."""
    parser = argparse.ArgumentParser(
        description="Apply transformation poses to raw point clouds.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '--dataset', 
        type=str, 
        required=True,
        help='Dataset name (e.g., "dataset6").'
    )
    args = parser.parse_args()
    
    logger.info(f"--- Starting Point Cloud Transformation for Dataset: {args.dataset} ---")

    try:
        # Define directories based on the new structure
        base_dir = os.path.join("raw_data", "point_clouds", args.dataset)
        transformations_dir = os.path.join(base_dir, "transformations")
        output_dir = os.path.join(base_dir, "transformed")
        os.makedirs(output_dir, exist_ok=True)

        # Find all raw point cloud files
        raw_ply_pattern = os.path.join(base_dir, 'pose*_raw.ply')
        ply_files = sorted(glob.glob(raw_ply_pattern), key=lambda x: int(os.path.basename(x).split('_')[0].replace('pose', '')))

        if not ply_files:
            logger.warning(f"No raw PLY files found in {base_dir}. Exiting.")
            return

        logger.info(f"Found {len(ply_files)} raw PLY files to process.")
        successful_transforms = 0

        for ply_file in ply_files:
            try:
                base_name = os.path.basename(ply_file)
                pose_num = int(base_name.split('_')[0].replace('pose', ''))
                
                # Find the best available transformation file for this pose
                transform_file = find_best_pose_file(transformations_dir, pose_num)

                if not transform_file:
                    continue

                logger.info(f"Processing pose {pose_num}: {base_name} -> using {os.path.basename(transform_file)}")

                # Define the output path
                output_path = os.path.join(output_dir, f"pose_{pose_num}_transformed.ply")

                # Load, transform, and save the point cloud
                pcd = o3d.io.read_point_cloud(ply_file)
                transform_matrix = read_transform_yml(transform_file)
                pcd.transform(transform_matrix)
                o3d.io.write_point_cloud(output_path, pcd)
                
                logger.info(f"Successfully saved transformed cloud to {output_path}")
                successful_transforms += 1

            except Exception as e:
                logger.error(f"Error processing pose {pose_num}: {e}", exc_info=True)
                continue

        logger.info(f"--- Transformation Complete. Successfully transformed {successful_transforms}/{len(ply_files)} point clouds. ---")

    except Exception as e:
        logger.error(f"A fatal error occurred: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
