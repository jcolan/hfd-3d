#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Final Image Cropping
======================
VERSION: 1.0

Description:
    This script takes the final generated images (original color, projected view,
    and depth map) and crops them to the largest possible square centered on the
    circular endoscopic view. This prepares the images for dataset creation.

Usage:
    python 5_crop_final_images.py --dataset DATASET_NAME --frame_start START --frame_end END [options]

"""

import numpy as np
import cv2
import os
import argparse
import logging
from math import sqrt

# Assuming config.py is in the project root
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dataset_config import DATASET_CONFIGS

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

def crop_image(image_path, new_path, center_x, center_y, size):
    """Crops an image to a square region and saves it."""
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        logger.warning(f"Could not read image, skipping: {image_path}")
        return

    half_size = size // 2
    x1, y1 = center_x - half_size, center_y - half_size
    x2, y2 = center_x + half_size, center_y + half_size

    # Ensure crop coordinates are within image bounds
    if not (0 <= y1 < y2 <= img.shape[0] and 0 <= x1 < x2 <= img.shape[1]):
        logger.error(f"Crop coordinates [{y1}:{y2}, {x1}:{x2}] are out of bounds for image size {img.shape}. Skipping {image_path}")
        return

    cropped_img = img[y1:y2, x1:x2]
    cv2.imwrite(new_path, cropped_img)
    logger.debug(f"Saved cropped image to: {new_path}")

# --- Main Execution ---

def main():
    parser = argparse.ArgumentParser(description="Crop circular endoscopic images to the largest inscribed square.")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (e.g., 'dataset6').")
    parser.add_argument("--frame_start", type=int, required=True, help="Starting frame number.")
    parser.add_argument("--frame_end", type=int, required=True, help="Ending frame number.")
    parser.add_argument("--quality", type=str, default="standard", help="Quality suffix of the generated images to crop.")
    # MODIFICATION 1: Add the --pose_type argument
    parser.add_argument("--pose_type", type=str, default="optimized", choices=['initial', 'optimized'], help="Pose type suffix ('initial' or 'optimized').")
    args = parser.parse_args()

    logger.info(f"--- Starting Final Image Cropping for Dataset: {args.dataset} ---")

    dataset_dir = os.path.join("raw_data", "endoscope", args.dataset)
    config = DATASET_CONFIGS.get(args.dataset, DATASET_CONFIGS["default"])
    center_x, center_y = config["mask_center"]
    
    radius = 635 
    square_size = int((2 * radius) / sqrt(2))

    processed_count = 0
    for frame_num in range(args.frame_start, args.frame_end + 1):
        frame_dir = os.path.join(dataset_dir, f"frame{frame_num}")
        if not os.path.isdir(frame_dir):
            logger.warning(f"Frame directory not found, skipping: {frame_dir}")
            continue
        
        logger.info(f"Processing frame {frame_num}...")
        
        # Build the correct filename suffix based on quality and pose type.
        suffix = f"_merged_{args.quality}_{args.pose_type}"
        
        image_files_to_crop = {
            "color": f"frame{frame_num}.png",
            "projected": f"frame{frame_num}_projected{suffix}.png",
            "depth": f"frame{frame_num}_depth{suffix}.png"
        }

        for key, filename in image_files_to_crop.items():
            image_path = os.path.join(frame_dir, filename)
            if os.path.exists(image_path):
                base, ext = os.path.splitext(filename)
                new_filename = f"{base}_cropped{ext}"
                new_path = os.path.join(frame_dir, new_filename)
                crop_image(image_path, new_path, center_x, center_y, square_size)
            else:
                logger.debug(f"Image not found for cropping, skipping: {image_path}")
        processed_count += 1

    logger.info(f"--- Cropping Complete. Processed {processed_count} frames. ---")

if __name__ == "__main__":
    main()