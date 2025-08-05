#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Apply Initial Manual Offset to All Camera Poses
=================================================
VERSION: 1.0

Description:
    This script is the second step in the depth generation pipeline. It takes the
    global manual offset found using the `1_find_initial_offset.py` tool and
    applies it to every raw camera pose in a specified dataset.

    For each frame, it loads the original pose, applies the rotation and translation
    from the `initial_manual_offset.yml` file, and saves the result as a new
    `_initial.yml` pose file. This provides a good starting point for the automated
    per-frame optimization that follows.

Usage:
    python depth_generation/2_apply_initial_offset.py --dataset <dataset_name>

"""
import yaml
import numpy as np
import os
import glob
import argparse
import sys
from scipy.spatial.transform import Rotation
import logging

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

def create_transformation_matrix(rotation_deg, translation_mm):
    """Creates a 4x4 transformation matrix from Euler angles and translation."""
    # Use 'xyz' order to match the reference script.
    r = Rotation.from_euler('xyz', rotation_deg, degrees=True).as_matrix()
    t = np.array(translation_mm)
    matrix = np.eye(4)
    matrix[:3, :3] = r
    matrix[:3, 3] = t
    return matrix

def load_pose(yaml_path):
    """Loads a raw camera pose from a .yml file."""
    with open(yaml_path, "r") as file:
        data = yaml.safe_load(file)
    pos = data["pose"]["position"]
    quat = data["pose"]["orientation"]
    
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_quat([quat["x"], quat["y"], quat["z"], quat["w"]]).as_matrix()
    matrix[:3, 3] = [pos["x"] * 1000, pos["y"] * 1000, pos["z"] * 1000]
    return matrix

def save_pose(matrix, file_path):
    """Saves a 4x4 transformation matrix back to a .yml file."""
    r = Rotation.from_matrix(matrix[:3, :3])
    quat = r.as_quat()
    pos = matrix[:3, 3] / 1000.0 # Convert back to meters

    data = {
        'pose': {
            'position': {'x': float(pos[0]), 'y': float(pos[1]), 'z': float(pos[2])},
            'orientation': {'w': float(quat[3]), 'x': float(quat[0]), 'y': float(quat[1]), 'z': float(quat[2])}
        }
    }
    with open(file_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)

# --- Main Execution ---

def main():
    parser = argparse.ArgumentParser(description="Apply a global manual offset to all raw poses in a dataset.")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (e.g., 'dataset6').")
    args = parser.parse_args()

    logger.info(f"--- Applying Global Offset for Dataset: {args.dataset} ---")

    endoscope_dir = os.path.join("raw_data", "endoscope", args.dataset)
    offset_file_path = os.path.join(endoscope_dir, "initial_manual_offset.yml")

    # --- Load Global Offset ---
    try:
        logger.info(f"Loading global offset from: {offset_file_path}")
        with open(offset_file_path, 'r') as f:
            offset_data = yaml.safe_load(f)
        rotation_deg = offset_data['rotation_xyz_deg']
        translation_mm = offset_data['translation_xyz_mm']
        
        # Create the rotation matrix from the offset using the 'xyz' order to match the old script
        R_offset = Rotation.from_euler('xyz', rotation_deg, degrees=True).as_matrix()
        t_offset = np.array(translation_mm)

    except FileNotFoundError:
        logger.error(f"Offset file not found: {offset_file_path}")
        logger.error("Please run '1_find_initial_offset.py' first to generate this file.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to load or parse offset file: {e}", exc_info=True)
        sys.exit(1)

    # --- Process All Frames ---
    raw_pose_files = glob.glob(os.path.join(endoscope_dir, "frame*", "frame*.yml"))
    if not raw_pose_files:
        logger.warning(f"No raw pose files found in {endoscope_dir}. Nothing to do.")
        return

    logger.info(f"Found {len(raw_pose_files)} total yml files. Filtering for raw poses.")
    processed_count = 0
    for raw_pose_path in raw_pose_files:
        # Skip files that have already been processed to avoid creating 'initial_initial.yml' files
        if '_initial.yml' in raw_pose_path or '_optimized.yml' in raw_pose_path:
            continue
        
        try:
            frame_name = os.path.basename(raw_pose_path).replace('.yml', '')
            logger.debug(f"Processing {frame_name}...")

            raw_pose = load_pose(raw_pose_path)
            
            # Apply offset exactly as it's done in the interactive script
            initial_pose = raw_pose.copy()
            initial_pose[:3, :3] = raw_pose[:3, :3] @ R_offset
            initial_pose[:3, 3] = raw_pose[:3, 3] + t_offset

            output_path = raw_pose_path.replace('.yml', '_initial.yml')
            save_pose(initial_pose, output_path)
            logger.debug(f"Saved initial pose to: {output_path}")
            processed_count += 1
        except Exception as e:
            logger.error(f"Failed to process {raw_pose_path}: {e}", exc_info=True)
            continue

    logger.info(f"--- Processing Complete. Successfully created {processed_count} initial poses. ---")

if __name__ == "__main__":
    main()
