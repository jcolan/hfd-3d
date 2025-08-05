#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Point Cloud Merging Tool
========================
VERSION: 1.0

Description:
    This script merges multiple transformed point cloud files (.ply) from a
    specified dataset directory into a single, coherent point cloud. It uses
    the Colored Iterative Closest Point (ICP) algorithm for robust alignment
    and provides quality presets for balancing speed and accuracy.

Usage:
    python 4_merge_transformed_clouds.py --dataset dataset6 --quality standard
"""

import open3d as o3d
import numpy as np
import time
import os
import glob
import argparse
import logging
import sys

# --- Logging Configuration ---
def setup_logger():
    """Sets up a logger for consistent and formatted output."""
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

def find_transformed_ply_files(transformed_dir):
    """Finds and sorts all transformed PLY files in the specified directory."""
    logger.info(f'Searching for transformed PLY files in: {transformed_dir}')
    pattern = os.path.join(transformed_dir, 'pose_*_transformed.ply')
    files = glob.glob(pattern)
    
    # Sort files by pose number to ensure sequential alignment.
    files.sort(key=lambda x: int(os.path.basename(x).split('_')[1]))
    
    logger.info(f"Found {len(files)} transformed PLY files.")
    return files

def merge_colored_point_clouds(cloud_files, quality_mode='standard'):
    """
    Merges multiple point clouds using a robust Colored ICP pipeline with
    configurable quality presets.
    """
    logger.info(f"Starting point cloud merging in {quality_mode.upper()} mode...")

    # Configuration parameters for different quality levels.
    params = {
        'draft': {
            'initial_voxel_size': 1.0, 'normal_radius': 2.0, 'normal_max_nn': 30, 'normal_orient_k': 20,
            'threshold': 1.0, 'max_iterations': 30, 'relative_fitness': 1e-6, 'relative_rmse': 1e-6,
            'final_voxel_size': 0.5, 'outlier_nb_neighbors': 20, 'outlier_std_ratio': 2.0
        },
        'standard': {
            'initial_voxel_size': 0.2, 'normal_radius': 0.8, 'normal_max_nn': 50, 'normal_orient_k': 50,
            'threshold': 0.3, 'max_iterations': 120, 'relative_fitness': 1e-8, 'relative_rmse': 1e-8,
            'final_voxel_size': 0.1, 'outlier_nb_neighbors': 50, 'outlier_std_ratio': 1.6
        },
        'fine': {
            'initial_voxel_size': 0.1, 'normal_radius': 0.4, 'normal_max_nn': 50, 'normal_orient_k': 50,
            'threshold': 0.15, 'max_iterations': 200, 'relative_fitness': 1e-9, 'relative_rmse': 1e-9,
            'final_voxel_size': 0.05, 'outlier_nb_neighbors': 60, 'outlier_std_ratio': 1.5
        }
    }
    config = params.get(quality_mode, params['standard'])
    logger.info(f"Using parameters for '{quality_mode}' mode.")

    # --- Load and Preprocess Clouds ---
    preprocessed_clouds = []
    for i, file in enumerate(cloud_files):
        logger.info(f"Loading and preprocessing cloud {i+1}/{len(cloud_files)}: {os.path.basename(file)}")
        pcd = o3d.io.read_point_cloud(file)
        if not pcd.has_colors():
            raise ValueError(f"Point cloud {file} must have color information for Colored ICP.")

        pcd_down = pcd.voxel_down_sample(voxel_size=config['initial_voxel_size'])
        pcd_down.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=config['normal_radius'], max_nn=config['normal_max_nn']
            )
        )
        pcd_down.orient_normals_consistent_tangent_plane(k=config['normal_orient_k'])
        preprocessed_clouds.append(pcd_down)

    # --- Colored ICP Alignment ---
    if not preprocessed_clouds:
        return None, {}

    # The first cloud is the reference; subsequent clouds are aligned to it.
    reference_cloud = preprocessed_clouds[0]
    aligned_clouds = [reference_cloud]
    trans_init = np.identity(4) # Initial transformation guess
    all_fitness, all_rmse = [], []

    for i in range(1, len(preprocessed_clouds)):
        logger.info(f"Aligning cloud {i+1} to the reference cloud...")
        source_cloud = preprocessed_clouds[i]
        
        reg_result = o3d.pipelines.registration.registration_colored_icp(
            source_cloud, reference_cloud, config['threshold'], trans_init,
            o3d.pipelines.registration.TransformationEstimationForColoredICP(),
            o3d.pipelines.registration.ICPConvergenceCriteria(
                relative_fitness=config['relative_fitness'],
                relative_rmse=config['relative_rmse'],
                max_iteration=config['max_iterations']
            )
        )
        
        logger.info(f"  - Fitness: {reg_result.fitness:.4f}, RMSE: {reg_result.inlier_rmse:.4f}")
        all_fitness.append(reg_result.fitness)
        all_rmse.append(reg_result.inlier_rmse)

        source_cloud.transform(reg_result.transformation)
        aligned_clouds.append(source_cloud)

    # --- Final Combination and Post-processing ---
    logger.info("Combining all aligned clouds into a final point cloud...")
    final_cloud = o3d.geometry.PointCloud()
    for cloud in aligned_clouds:
        final_cloud += cloud

    logger.info(f"Final cloud has {len(final_cloud.points)} points before post-processing.")
    final_cloud = final_cloud.voxel_down_sample(voxel_size=config['final_voxel_size'])
    logger.info(f"Downsampled to {len(final_cloud.points)} points.")

    logger.info("Removing statistical outliers...")
    final_cloud, _ = final_cloud.remove_statistical_outlier(
        nb_neighbors=config['outlier_nb_neighbors'], std_ratio=config['outlier_std_ratio']
    )
    logger.info(f"Final point count after outlier removal: {len(final_cloud.points)}")

    metrics = {
        "mean_fitness": np.mean(all_fitness) if all_fitness else 0,
        "std_fitness": np.std(all_fitness) if all_fitness else 0,
        "mean_rmse": np.mean(all_rmse) if all_rmse else 0,
        "std_rmse": np.std(all_rmse) if all_rmse else 0
    }
    return final_cloud, metrics

# --- Main Execution ---

def main():
    """Main function to parse arguments and run the merging process."""
    parser = argparse.ArgumentParser(
        description="Merge transformed point clouds using Colored ICP.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name (e.g., "dataset6").')
    parser.add_argument(
        '--quality', type=str, default='standard',
        choices=['draft', 'standard', 'fine'],
        help="""Quality preset for the merging process:
- draft: Fastest, for quick previews.
- standard: Good balance of speed and quality (default).
- fine: Highest quality, slowest processing."""
    )
    args = parser.parse_args()

    logger.info(f"--- Starting Point Cloud Merging for Dataset: {args.dataset} ---")

    try:
        base_dir = os.path.join("raw_data", "point_clouds", args.dataset)
        transformed_dir = os.path.join(base_dir, "transformed")
        merged_dir = os.path.join(base_dir, "merged")
        os.makedirs(merged_dir, exist_ok=True)

        cloud_files = find_transformed_ply_files(transformed_dir)
        if not cloud_files:
            raise FileNotFoundError(f"No transformed PLY files found in {transformed_dir}")

        result_pcd, icp_metrics = merge_colored_point_clouds(cloud_files, quality_mode=args.quality)

        if result_pcd is None:
            logger.warning("Merging resulted in an empty point cloud. Nothing to save.")
            return

        # Save the merged point cloud
        output_path_pcd = os.path.join(merged_dir, f"merged_cloud_{args.quality}.ply")
        logger.info(f"Saving merged cloud to {output_path_pcd}...")
        o3d.io.write_point_cloud(output_path_pcd, result_pcd)

        # Save the ICP metrics
        output_path_metrics = os.path.join(merged_dir, f"icp_metrics_{args.quality}.txt")
        with open(output_path_metrics, "w") as f:
            f.write(f"ICP Alignment Metrics for quality='{args.quality}'\n")
            f.write("="*40 + "\n")
            f.write(f"Mean Fitness: {icp_metrics['mean_fitness']:.4f}\n")
            f.write(f"Std Dev Fitness: {icp_metrics['std_fitness']:.4f}\n")
            f.write(f"Mean Inlier RMSE (mm): {icp_metrics['mean_rmse']:.4f}\n")
            f.write(f"Std Dev Inlier RMSE (mm): {icp_metrics['std_rmse']:.4f}\n")
        logger.info(f"Saved ICP metrics to {output_path_metrics}")

        logger.info(f"--- Merging Process Completed Successfully for {args.dataset}! ---")
        print(f"\nOutput saved to: {output_path_pcd}")

    except Exception as e:
        logger.error(f"A fatal error occurred: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
