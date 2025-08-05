#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Initial Pose Generation for Point Clouds
========================================

Description:
    This script is the first step in the point cloud processing pipeline. It generates
    initial transformation poses for a set of raw point clouds. The process relies
    on having a reference pose from the robot's coordinate system for each raw
    point cloud, a single reference pose for the robot at a known location (e.g., pose 0),
    and a new desired reference pose that will become the origin for the merged scene.

    The script calculates the transformation of each point cloud relative to the
    robot's reference pose and then applies that transformation to the new desired
    reference pose. This effectively brings all point clouds into a common,
    user-defined coordinate system.

Usage:
    python point_cloud/1_generate_initial_poses.py --dataset <dataset_name> --num_poses <number_of_poses>

"""
import yaml
import numpy as np
import os
import logging
import argparse
import sys

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
    """Reads a 4x4 transformation matrix from a specified YAML file."""
    logger.info(f"Reading transformation from: {file_path}")
    try:
        with open(file_path, "r") as f:
            data = yaml.safe_load(f)
        matrix = np.array(data["FloatMatrix"]["Data"])
        logger.debug(f"Successfully read matrix of shape {matrix.shape}")
        return matrix
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Error reading {file_path}: {e}")
        raise

def save_transform_yml(matrix, file_path):
    """Saves a 4x4 transformation matrix to a specified YAML file."""
    logger.info(f"Saving transformation to: {file_path}")
    data = {
        "__version__": {"serializer": 1, "data": 1},
        "FloatMatrix": {"Data": matrix.tolist()},
    }
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as f:
            yaml.dump(data, f, default_flow_style=None)
        logger.debug(f"Successfully saved transformation matrix.")
    except Exception as e:
        logger.error(f"Error saving to {file_path}: {e}")
        raise

def calculate_new_pose(robot_pose, robot_reference_pose, new_reference_pose):
    """
    Calculates the new pose based on a new reference frame.
    Formula: T_new = (T_robot_N * T_robot_0^-1) * T_new_ref_0
    """
    logger.debug("Calculating relative transformation from robot reference to current pose.")
    try:
        # T_relative = T_robot_N * T_robot_0^-1
        robot_reference_inv = np.linalg.inv(robot_reference_pose)
        relative_transform = robot_pose @ robot_reference_inv
        
        logger.debug("Applying relative transform to the new reference pose.")
        # T_new = T_relative * T_new_ref_0
        new_pose = relative_transform @ new_reference_pose
        logger.debug("New pose calculated successfully.")
        return new_pose
    except np.linalg.LinAlgError as e:
        logger.error(f"Matrix inversion failed. Ensure transformation matrices are valid. Error: {e}")
        raise

# --- Main Execution ---

def main():
    """Main function to generate initial poses for a dataset."""
    parser = argparse.ArgumentParser(
        description="Generate initial camera poses for a dataset based on a new reference.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '--dataset', 
        type=str, 
        required=True,
        help="Name of the dataset to process (e.g., 'dataset6')."
    )
    parser.add_argument(
        '--num_poses', 
        type=int, 
        default=10,
        help="Number of poses to process (from 1 to N). Default is 10."
    )
    args = parser.parse_args()
    
    logger.info(f"--- Starting Initial Pose Generation for Dataset: {args.dataset} ---")
    
    # Define paths based on the new directory structure
    base_dir = os.path.join("raw_data", "point_clouds", args.dataset)
    transformations_dir = os.path.join(base_dir, "transformations")
    
    if not os.path.isdir(transformations_dir):
        logger.error(f"Transformations directory not found: {transformations_dir}")
        sys.exit(1)

    # Define file paths for key transformations
    robot_ref_path = os.path.join(transformations_dir, "pose_0_from_robot.yaml")
    new_ref_path = os.path.join(transformations_dir, "reference_pose_0.yaml")
    output_pose0_path = os.path.join(transformations_dir, "pose_0_initial.yaml")

    try:
        # Load the key reference transformations
        robot_reference_pose = read_transform_yml(robot_ref_path)
        new_reference_pose = read_transform_yml(new_ref_path)

        # The initial pose for frame 0 is simply the new reference pose.
        logger.info(f"Creating initial pose for frame 0 from the new reference.")
        save_transform_yml(new_reference_pose, output_pose0_path)
        
        processed_count = 1 # Start with 1 for pose 0
        # Process all other poses from 1 to N
        for i in range(1, args.num_poses + 1):
            logger.info(f"Processing pose {i}...")
            robot_pose_path = os.path.join(transformations_dir, f"pose_{i}_from_robot.yaml")
            output_pose_path = os.path.join(transformations_dir, f"pose_{i}_initial.yaml")

            if not os.path.exists(robot_pose_path):
                logger.warning(f"Robot pose file not found, skipping: {robot_pose_path}")
                continue
            
            robot_pose = read_transform_yml(robot_pose_path)
            
            # Calculate the new pose
            new_pose = calculate_new_pose(robot_pose, robot_reference_pose, new_reference_pose)
            
            # Save the newly calculated pose
            save_transform_yml(new_pose, output_pose_path)
            logger.info(f"Successfully generated and saved {os.path.basename(output_pose_path)}")
            processed_count += 1

    except FileNotFoundError:
        logger.error("A required reference file was not found. Please ensure both "
                     f"'{os.path.basename(robot_ref_path)}' and '{os.path.basename(new_ref_path)}' exist.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)
        sys.exit(1)

    logger.info(f"--- Pose Generation Complete for Dataset: {args.dataset} ---")
    logger.info(f"Successfully processed {processed_count} poses.")

if __name__ == "__main__":
    main()
