#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fix Incomplete Dataset
======================

Description:
    This script scans a specified dataset for frames that are missing either their
    depth map or their projected color image. It then generates only the missing
    files for those specific frames. This avoids reprocessing the entire dataset
    and is much more efficient than calling the extraction script for each
    missing frame individually, as the main point cloud is loaded only once.

Usage:
    python fix_incomplete_dataset.py --dataset DATASET_NAME [options]

Arguments:
    --dataset           : Dataset name (e.g., "dataset1") to check and fix.
    --quality           : The quality suffix of the files to check (e.g., 'ultra_hd').
                          Defaults to 'ultra_hd'.
    --point_cloud       : Optional: Path to the point cloud file to use.
                          Overrides the default path.
    --save_projected_view: Optional: Also generate and save the projected color view.
                           Defaults to True.
"""
import os
import numpy as np
import open3d as o3d
import cv2
import yaml
from scipy.spatial.transform import Rotation
import argparse
import sys

# --- Dataset Specific Configurations (copied from original script) ---
DATASET_CONFIGS = {
    "default": {"mask_center": (1040, 550)},
    "dataset1": {"mask_center": (1020, 555)},
    "dataset2": {"mask_center": (1025, 545)},
    "dataset3": {"mask_center": (1028, 540)},
    "dataset4": {"mask_center": (1028, 545)},
    "dataset5": {"mask_center": (1025, 555)},
    "dataset6": {"mask_center": (1025, 560)},
    "dataset7": {"mask_center": (1025, 555)},
    "dataset8": {"mask_center": (1025, 555)},
    "dataset9": {"mask_center": (1025, 555)},
    "dataset10": {"mask_center": (1025, 555)},
    "dataset11": {"mask_center": (1025, 555)},
    "dataset12": {"mask_center": (1025, 545)},
    "dataset13": {"mask_center": (1028, 555)},
    "dataset14": {"mask_center": (1028, 555)},
    "dataset15": {"mask_center": (1028, 550)},
    "dataset16": {"mask_center": (1028, 550)},
    "dataset17": {"mask_center": (1028, 550)},
    "dataset18": {"mask_center": (1028, 550)},
    "dataset19": {"mask_center": (1028, 555)},
    "dataset20": {"mask_center": (1028, 550)},
    "dataset21": {"mask_center": (1028, 550)},
    "dataset22": {"mask_center": (1028, 550)},
    "dataset23": {"mask_center": (1025, 555)},
    "dataset24": {"mask_center": (1025, 545)},
    "dataset25": {"mask_center": (1025, 545)},
    "dataset26": {"mask_center": (1025, 555)},
    "dataset27": {"mask_center": (1025, 550)},
    "dataset28": {"mask_center": (1025, 550)},
}

# --- Functions from original script (with minor modifications for clarity) ---

def parse_args():
    parser = argparse.ArgumentParser(description="Fix incomplete dataset by generating missing depth maps and projected views.")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (e.g., 'dataset1')")
    parser.add_argument("--total_frames", type=int, required=True, help="The total number of frames that should be in the dataset.")
    parser.add_argument("--quality", type=str, default="ultra_hd", choices=['debug', 'uhd', 'ultra_hd'], help="Quality suffix for files.")
    parser.add_argument("--point_cloud", type=str, help="Path to point cloud file")
    parser.add_argument("--initial_rotation", type=float, nargs=3, help="Initial rotation in degrees (rx ry rz)")
    parser.add_argument("--initial_translation", type=float, nargs=3, help="Initial translation in mm (tx ty tz)")
    parser.add_argument("--save_projected_view", action=argparse.BooleanOptionalAction, default=True, help="Save the projected color view.")
    parser.add_argument("--near_clip", type=float, default=30.0, help="Near clipping distance in mm.")
    parser.add_argument("--remove_outliers", action="store_true", help="Enable statistical outlier removal.")
    parser.add_argument("--point_size", type=int, default=1, help="Size of the points in the rendered image.")
    parser.add_argument("--fill_holes", action="store_true", help="Enable morphological closing to fill holes.")
    parser.add_argument("--hole_fill_kernel_size", type=int, default=5, help="Kernel size for hole-filling.")
    return parser.parse_args()

def find_incomplete_frames(dataset_name, quality_suffix, total_frames):
    """Scans the dataset directory to find frames with missing files, up to the expected total number of frames."""
    base_dir = f"./data/endoscope/{dataset_name}"
    incomplete_frames = []

    # Check if the base directory exists, but don't fail if it doesn't,
    # as we might need to create it for the missing frames.
    if not os.path.isdir(base_dir):
        print(f"Warning: Dataset directory not found at {base_dir}. Assuming all frames are missing.")

    for frame_number in range(total_frames):
        frame_dir_path = os.path.join(base_dir, f"frame{frame_number}")
        
        depth_filename = f"{frame_dir_path}/frame{frame_number}_depth_merged_{quality_suffix}.png"
        projected_filename = f"{frame_dir_path}/frame{frame_number}_projected_merged_{quality_suffix}.png"

        # A frame is considered incomplete if the directory or either of the files are missing.
        if not os.path.isdir(frame_dir_path) or not os.path.exists(depth_filename) or not os.path.exists(projected_filename):
            incomplete_frames.append(frame_number)
            
    return incomplete_frames

def create_circular_mask(image_size, radius=None, dataset_name=None):
    h, w = image_size
    config = DATASET_CONFIGS.get(dataset_name, DATASET_CONFIGS["default"])
    center = config["mask_center"]
    if radius is None:
        radius = min(center[0], center[1], w - center[0], h - center[1])
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)
    mask = dist_from_center <= radius
    return mask

def quaternion_to_matrix(q):
    r = Rotation.from_quat([q["x"], q["y"], q["z"], q["w"]])
    return r.as_matrix()

def load_pose(yaml_path, rotation_deg=(0, 0, 0), trans_offset_mm=(0, 0, 0)):
    with open(yaml_path, "r") as file:
        data = yaml.safe_load(file)
    pos = data["pose"]["position"]
    quat = data["pose"]["orientation"]
    T = np.eye(4)
    T[:3, :3] = quaternion_to_matrix(quat)
    T[:3, 3] = [pos["x"] * 1000, pos["y"] * 1000, pos["z"] * 1000]
    x_rad, y_rad, z_rad = np.radians(rotation_deg)
    Rx = np.array([[1, 0, 0], [0, np.cos(x_rad), -np.sin(x_rad)], [0, np.sin(x_rad), np.cos(x_rad)]])
    Ry = np.array([[np.cos(y_rad), 0, np.sin(y_rad)], [0, 1, 0], [-np.sin(y_rad), 0, np.cos(y_rad)]])
    Rz = np.array([[np.cos(z_rad), -np.sin(z_rad), 0], [np.sin(z_rad), np.cos(z_rad), 0], [0, 0, 1]])
    R_combined = Rz @ Ry @ Rx
    T_final = np.eye(4)
    T_final[:3, :3] = T[:3, :3] @ R_combined
    T_final[:3, 3] = T[:3, 3] + np.array(trans_offset_mm)
    return T_final

def project_point_cloud(points, colors, camera_matrix, dist_coeffs, camera_pose, image_size, point_size=1, smooth=True, dataset_name=None, near_clip=10, far_clip=500, fill_holes=False, hole_fill_kernel_size=5):
    # This function is copied directly from the original script
    camera_pose_mm = camera_pose.copy()
    R = camera_pose_mm[:3, :3]
    t = camera_pose_mm[:3, 3]
    R_inv = R.T
    t_inv = -R_inv @ t
    points_cam = (R_inv @ points.T + t_inv.reshape(3, 1)).T
    depths = points_cam[:, 2]
    mask = (depths > near_clip) & (depths < far_clip)
    points_cam = points_cam[mask]
    depths = depths[mask]
    if colors is not None:
        colors = colors[mask]
        colors = colors[:, [2, 1, 0]]
    if len(points_cam) == 0:
        return (np.zeros(image_size + (3,), dtype=np.uint8), 
                np.zeros(image_size, dtype=np.uint16))
    points_cam_meters = points_cam / 1000.0
    points_2d, _ = cv2.projectPoints(points_cam_meters, np.zeros(3), np.zeros(3), camera_matrix, dist_coeffs)
    points_2d = points_2d.reshape(-1, 2)
    center_x = image_size[1] / 2
    center_y = image_size[0] / 2
    points_2d_centered = points_2d - [center_x, center_y]
    points_2d_rotated = -points_2d_centered
    points_2d = points_2d_rotated + [center_x, center_y]
    rendered_image = np.zeros(image_size + (3,), dtype=np.uint8)
    depth_buffer = np.full(image_size, np.inf)
    mask = ((points_2d[:, 0] >= 0) & (points_2d[:, 0] < image_size[1]) & (points_2d[:, 1] >= 0) & (points_2d[:, 1] < image_size[0]))
    valid_points = points_2d[mask].astype(int)
    valid_depths = depths[mask]
    if colors is not None:
        valid_colors = colors[mask]
        if valid_colors.dtype == np.float64:
            valid_colors = (valid_colors * 255).astype(np.uint8)
    else:
        depths_normalized = ((valid_depths - valid_depths.min()) / (valid_depths.max() - valid_depths.min()) * 255).astype(np.uint8)
        colormap = cv2.COLORMAP_COPPER
        valid_colors = cv2.applyColorMap(depths_normalized.reshape(-1, 1), colormap).squeeze()
    for point, color, depth in zip(valid_points, valid_colors, valid_depths):
        y1, y2 = max(0, point[1] - point_size), min(image_size[0], point[1] + point_size + 1)
        x1, x2 = max(0, point[0] - point_size), min(image_size[1], point[0] + point_size + 1)
        for y in range(y1, y2):
            for x in range(x1, x2):
                if depth < depth_buffer[y, x]:
                    rendered_image[y, x] = color
                    depth_buffer[y, x] = depth
    mask_radius = 635
    circular_mask = create_circular_mask(image_size, radius=mask_radius, dataset_name=dataset_name)
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
    depth_map = np.clip(depth_map, 0, 300)
    depth_map = ((depth_map / 300.0) * 65535).astype(np.uint16)
    depth_map[~circular_mask] = 0
    if fill_holes:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (hole_fill_kernel_size, hole_fill_kernel_size))
        depth_map = cv2.morphologyEx(depth_map, cv2.MORPH_CLOSE, kernel)
    return rendered_image, depth_map

if __name__ == "__main__":
    args = parse_args()

    # 1. Find all incomplete frames first
    incomplete_frames = find_incomplete_frames(args.dataset, args.quality, args.total_frames)

    if not incomplete_frames:
        print(f"Dataset '{args.dataset}' is complete. No files to generate.")
        sys.exit(0)

    print(f"Found {len(incomplete_frames)} incomplete frames in dataset '{args.dataset}'.")
    print("Frames to process:", incomplete_frames)

    # 2. Load point cloud and camera parameters ONCE
    camera_matrix = np.array([[1.15734935e03, 0, 1.02914674e03],
                              [0, 1.15627765e03, 5.27741541e02],
                              [0, 0, 1]])
    dist_coeffs = np.array([[-0.4894732329144252, 0.32992541053980134, -0.0010032472743986569, -0.00014656349448021337, -0.14193195002328468 * 0.1]])
    
    output_suffix = f"_merged_{args.quality}"
    pcd_file = (args.point_cloud or f"./data/point_clouds/{args.dataset}/merged/merged_colored_cloud_{args.quality}.ply")

    print(f"Loading point cloud from: {pcd_file}")
    try:
        pcd = o3d.io.read_point_cloud(pcd_file)
    except Exception as e:
        print(f"Error: Could not read point cloud file at {pcd_file}. Aborting.", file=sys.stderr)
        print(e, file=sys.stderr)
        sys.exit(1)

    if args.remove_outliers:
        print("Performing statistical outlier removal...")
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        print("Outlier removal complete.")

    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors) if pcd.has_colors() else None
    image_size = (1088, 2048)
    initial_rotation = args.initial_rotation or [0, 0, 0]
    initial_translation = args.initial_translation or [0, 0, 0]

    # 3. Process only the incomplete frames
    for frame_number in incomplete_frames:
        print(f"\nProcessing frame {frame_number}")
        pose_file = f"raw_data/endoscope/{args.dataset}/frame{frame_number}/frame{frame_number}_aligned.yml"
        
        if not os.path.exists(pose_file):
            print(f"  - Warning: Pose file not found for frame {frame_number} at {pose_file}. Skipping.")
            continue

        try:
            camera_pose = load_pose(pose_file, rotation_deg=tuple(initial_rotation), trans_offset_mm=tuple(initial_translation))
            
            projected_image, depth_map = project_point_cloud(
                points, colors, camera_matrix, dist_coeffs, camera_pose, image_size, 
                dataset_name=args.dataset, near_clip=args.near_clip, point_size=args.point_size, 
                fill_holes=args.fill_holes, hole_fill_kernel_size=args.hole_fill_kernel_size
            )
            
            output_dir = f"raw_data/endoscope/{args.dataset}/frame{frame_number}"
            os.makedirs(output_dir, exist_ok=True)
            
            # Since this frame was marked as incomplete, we will save/overwrite both files to ensure consistency.
            depth_filename = f"{output_dir}/frame{frame_number}_depth{output_suffix}.png"
            if depth_map is not None and depth_map.size > 0:
                cv2.imwrite(depth_filename, depth_map)
                print(f"  - Saved/Overwrote depth map: {depth_filename}")
            else:
                print(f"  - Warning: Generated depth map for frame {frame_number} was empty. Not saving.")

            if args.save_projected_view:
                projected_filename = f"{output_dir}/frame{frame_number}_projected{output_suffix}.png"
                if projected_image is not None and projected_image.size > 0:
                    cv2.imwrite(projected_filename, projected_image)
                    print(f"  - Saved/Overwrote projected view: {projected_filename}")
                else:
                    print(f"  - Warning: Generated projected image for frame {frame_number} was empty. Not saving.")

        except Exception as e:
            # This will catch errors during pose loading, projection, or file writing (like 'no space left')
            print(f"  - Error processing frame {frame_number}: {str(e)}", file=sys.stderr)
            continue
            
    print("\n--- Fix operation complete! ---")
