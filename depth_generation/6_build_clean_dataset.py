#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Clean Dataset Builder
=======================
VERSION: 1.0

Description:
    This is the final step in the data generation pipeline. It gathers all the
    necessary data products (cropped images, optimized poses, merged point cloud)
    and organizes them into a new, clean directory structure. This 'depth_data'
    directory serves as the definitive source for all downstream tasks like
    validation and machine learning.

Usage:
    python 6_build_clean_dataset.py --dataset DATASET_NAME --frame_start START --frame_end END [options]

"""

import os
import shutil
import argparse
import sys
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

# --- Main Execution ---

def main():
    parser = argparse.ArgumentParser(description="Build the final clean dataset from processed components.")
    parser.add_argument("--dataset", type=str, required=True, help="Name of the dataset to process (e.g., 'dataset6').")
    parser.add_argument("--frame_start", type=int, required=True, help="Starting frame number.")
    parser.add_argument("--frame_end", type=int, required=True, help="Ending frame number.")
    parser.add_argument("--quality", type=str, default="standard", help="Quality suffix of the data products to use.")
    parser.add_argument("--output_dir", type=str, default="depth_data", help="Name of the root directory for the clean dataset.")
    parser.add_argument("--pose_type", type=str, default="optimized", choices=['initial', 'optimized'], help="Pose type to use ('initial' or 'optimized').")
    args = parser.parse_args()

    logger.info(f"--- Building Clean Dataset for: {args.dataset} ---")

    # --- Define Source and Destination Paths ---
    source_endoscope_dir = os.path.join("raw_data", "endoscope", args.dataset)
    source_pcd_dir = os.path.join("raw_data", "point_clouds", args.dataset, "merged")
    
    dest_endoscope_dir = os.path.join(args.output_dir, "endoscope", args.dataset)
    dest_pcd_dir = os.path.join(args.output_dir, "point_clouds", args.dataset, "merged")

    os.makedirs(dest_endoscope_dir, exist_ok=True)
    os.makedirs(dest_pcd_dir, exist_ok=True)

    # --- Copy Endoscope Data (Images and Poses) ---
    logger.info(f"Processing frames {args.frame_start} to {args.frame_end}...")
    frames_processed = 0
    for frame_num in range(args.frame_start, args.frame_end + 1):
        source_frame_dir = os.path.join(source_endoscope_dir, f"frame{frame_num}")
        if not os.path.isdir(source_frame_dir):
            logger.warning(f"Source frame directory not found, skipping: {source_frame_dir}")
            continue

        dest_frame_dir = os.path.join(dest_endoscope_dir, f"frame{frame_num}")
        os.makedirs(dest_frame_dir, exist_ok=True)

        # MODIFICATION: Correctly build the suffix for image filenames to include pose_type
        image_suffix = f"_merged_{args.quality}_{args.pose_type}_cropped.png"
        
        file_mappings = {
            f"frame{frame_num}_cropped.png": f"frame{frame_num}.png",
            f"frame{frame_num}_{args.pose_type}.yml": f"frame{frame_num}.yml",
            f"frame{frame_num}_depth{image_suffix}": f"frame{frame_num}_depth.png",
            f"frame{frame_num}_projected{image_suffix}": f"frame{frame_num}_projected.png"
        }

        files_copied_for_frame = 0
        for source_name, dest_name in file_mappings.items():
            source_path = os.path.join(source_frame_dir, source_name)
            dest_path = os.path.join(dest_frame_dir, dest_name)

            if os.path.exists(source_path):
                shutil.copy(source_path, dest_path)
                logger.debug(f"Copied: {source_path} -> {dest_path}")
                files_copied_for_frame += 1
            else:
                logger.warning(f"Source file not found, skipping: {source_path}")
        
        if files_copied_for_frame > 0:
            frames_processed += 1

    logger.info(f"Finished processing {frames_processed} frames.")

    # --- Copy Merged Point Cloud ---
    logger.info("Copying merged point cloud...")
    source_pc_path = os.path.join(source_pcd_dir, f"merged_cloud_{args.quality}.ply")
    dest_pc_path = os.path.join(dest_pcd_dir, "point_cloud.ply")

    if os.path.exists(source_pc_path):
        shutil.copy(source_pc_path, dest_pc_path)
        logger.info(f"Copied point cloud to: {dest_pc_path}")
    else:
        logger.error(f"Merged point cloud not found: {source_pc_path}")

    logger.info(f"--- Clean Dataset Build Complete for: {args.dataset} ---")
    logger.info(f"Output ready in: {args.output_dir}")

if __name__ == "__main__":
    main()