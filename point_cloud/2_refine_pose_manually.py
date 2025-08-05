#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Manual Point Cloud Pose Refinement Tool
=======================================
VERSION: 1.0

Description:
    This script provides an interactive 3D environment to manually refine the
    pose of a target point cloud relative to a static base point cloud.
    It allows for fine-tuned adjustments to translation (x, y, z) and
    rotation (roll, pitch, yaw) via console input.

Usage:
    python 2_refine_pose_manually.py --dataset dataset6 --base_pose_num 1 --align_pose_num 2
"""

import open3d as o3d
import numpy as np
import argparse
import yaml
import os
import sys
import threading
import time
from scipy.spatial.transform import Rotation as R

# --- Core Functions ---

def load_point_cloud(path, voxel_size):
    """Loads, downsamples, and estimates normals for a point cloud."""
    print(f"Loading point cloud: {path}")
    pcd = o3d.io.read_point_cloud(path)
    if not pcd.has_points():
        raise FileNotFoundError(f"Could not read or found empty point cloud at {path}")
    
    print(f"Downsampling with voxel size: {voxel_size}")
    pcd_down = pcd.voxel_down_sample(voxel_size)
    
    print("Estimating normals...")
    pcd_down.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30)
    )
    return pcd_down

def read_transform_yml(file_path):
    """Reads a 4x4 transformation matrix from a YAML file."""
    print(f"Reading initial transformation from: {file_path}")
    with open(file_path, 'r') as f:
        data = yaml.safe_load(f)
    return np.array(data['FloatMatrix']['Data'])

def save_transform_yml(matrix, file_path):
    """Saves a 4x4 transformation matrix to a YAML file."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    data = {
        "__version__": {"serializer": 1, "data": 1},
        "FloatMatrix": {"Data": matrix.tolist()},
    }
    with open(file_path, "w") as f:
        yaml.dump(data, f, default_flow_style=None)
    print(f"\nTransformation saved to {file_path}")

def create_transformation_matrix(x, y, z, roll, pitch, yaw):
    """Creates a 4x4 transformation matrix from translation and Euler angles."""
    rotation = R.from_euler('xyz', [roll, pitch, yaw], degrees=True).as_matrix()
    translation = np.array([x, y, z])
    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = translation
    return matrix

# --- Interactive Console Handling ---

def handle_console_input(state, lock):
    """Runs in a separate thread to handle non-blocking console input for adjustments."""
    while state.get("is_running", False):
        try:
            with lock:
                prompt = (
                    "\nEnter 6 space-separated values for x y z roll pitch yaw (or press Enter to apply):\n"
                    f"Current Adjustments: {state['x']:.4f} {state['y']:.4f} {state['z']:.4f} "
                    f"{state['roll']:.2f} {state['pitch']:.2f} {state['yaw']:.2f}\n> "
                )
            input_str = input(prompt)
            if not input_str.strip():
                with lock:
                    state["needs_update"] = True # Re-apply current values
                continue

            values = list(map(float, input_str.split()))
            if len(values) != 6:
                print("Invalid input: Please provide exactly 6 values.")
                continue

            with lock:
                state["x"], state["y"], state["z"], state["roll"], state["pitch"], state["yaw"] = values
                state["needs_update"] = True

        except ValueError:
            print("Invalid input. Please enter numbers only.")
        except (EOFError, KeyboardInterrupt):
            with lock:
                state["is_running"] = False
            break

# --- Main Execution ---

def main():
    """Main function to set up the environment and run the refinement loop."""
    parser = argparse.ArgumentParser(
        description="Manually refine the pose of a point cloud relative to a base cloud.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (e.g., 'dataset6').")
    parser.add_argument("--base_pose_num", type=int, required=True, help="Pose number of the static base point cloud.")
    parser.add_argument("--align_pose_num", type=int, required=True, help="Pose number of the point cloud to be aligned.")
    parser.add_argument("--voxel_size", type=float, default=0.1, help="Voxel size for downsampling.")
    args = parser.parse_args()

    # Define file paths based on input arguments
    base_cloud_dir = os.path.join("raw_data", "point_clouds", args.dataset)
    transform_dir = os.path.join(base_cloud_dir, "transformations")
    pc_base_raw_path = os.path.join(base_cloud_dir, f"pose{args.base_pose_num}_raw.ply")
    base_transform_path = os.path.join(transform_dir, f"pose_{args.base_pose_num}_initial.yaml")
    pc_to_align_raw_path = os.path.join(base_cloud_dir, f"pose{args.align_pose_num}_raw.ply")
    initial_transform_path = os.path.join(transform_dir, f"pose_{args.align_pose_num}_initial.yaml")
    output_transform_path = os.path.join(transform_dir, f"pose_{args.align_pose_num}_refined.yaml")

    try:
        # Load base cloud and apply its transformation to place it in the world
        pc_base_raw = load_point_cloud(pc_base_raw_path, args.voxel_size)
        base_transform = read_transform_yml(base_transform_path)
        pc_base_transformed = pc_base_raw.transform(base_transform)
        pc_base_transformed.paint_uniform_color([0.5, 0.5, 0.5]) # Gray for reference

        # Load the raw point cloud that will be moved
        pc_to_align_raw = load_point_cloud(pc_to_align_raw_path, args.voxel_size)
        pc_to_align_raw.paint_uniform_color([0.9, 0.1, 0.1]) # Red for visibility
        pc_to_align_transformed = o3d.geometry.PointCloud(pc_to_align_raw)

        # Load the initial transformation for the cloud to be aligned
        initial_transform = read_transform_yml(initial_transform_path)

    except FileNotFoundError as e:
        print(f"\nError: {e}")
        sys.exit(1)

    # --- Visualization and State Management ---
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window("Manual Pose Refinement", 1280, 720)
    vis.add_geometry(pc_base_transformed)
    vis.add_geometry(pc_to_align_transformed)

    # State dictionary to manage variables between threads
    state = {
        "x": 0.0, "y": 0.0, "z": 0.0,             # Manual adjustments in mm
        "roll": 0.0, "pitch": 0.0, "yaw": 0.0,   # Manual adjustments in degrees
        "initial_transform": initial_transform,
        "final_transform": initial_transform,
        "is_running": True,
        "needs_update": True, # Flag to trigger a visual update
    }
    lock = threading.RLock()

    # --- Key Callbacks for the Visualizer ---
    def save_callback(vis):
        with lock:
            save_transform_yml(state["final_transform"], output_transform_path)

    def close_callback(vis):
        with lock:
            state["is_running"] = False

    vis.register_key_callback(ord("S"), save_callback)
    vis.register_key_callback(ord("Q"), close_callback)

    # --- Initial Visualizer Setup ---
    vis.get_render_option().point_size = 2.0
    vis.get_render_option().light_on = True
    vis.reset_view_point(True)

    print("\n" + "="*50)
    print("MANUAL ALIGNMENT INSTRUCTIONS")
    print("="*50)
    print("  - The RED cloud is the one you are aligning.")
    print("  - The GRAY cloud is the static reference.")
    print("\nKEYBINDINGS:")
    print("  'S': Save the final transformation.")
    print("  'Q': Quit the application.")

    # --- Main Application Loop ---
    # The input handler runs in a separate thread to prevent blocking the visualizer.
    input_thread = threading.Thread(target=handle_console_input, args=(state, lock))
    input_thread.start()

    while True:
        with lock:
            if not state["is_running"]:
                break
            
            # Only update the geometry if new adjustments have been entered.
            if state["needs_update"]:
                # Create a matrix for the manual adjustments.
                manual_adjust_matrix = create_transformation_matrix(
                    state["x"], state["y"], state["z"],
                    state["roll"], state["pitch"], state["yaw"]
                )
                # The final transform is the manual adjustment applied to the initial pose.
                final_transform = manual_adjust_matrix @ state["initial_transform"]
                state["final_transform"] = final_transform
                
                # To prevent cumulative transformations, always start from the raw cloud.
                pc_to_align_transformed.points = pc_to_align_raw.points
                pc_to_align_transformed.normals = pc_to_align_raw.normals
                pc_to_align_transformed.colors = pc_to_align_raw.colors
                pc_to_align_transformed.transform(final_transform)
                
                state["needs_update"] = False
                vis.update_geometry(pc_to_align_transformed)

        # Poll for events and update the renderer in the main thread.
        vis.poll_events()
        vis.update_renderer()
        time.sleep(0.01)

    vis.destroy_window()
    input_thread.join()
    print("\nApplication closed.")

if __name__ == "__main__":
    main()
