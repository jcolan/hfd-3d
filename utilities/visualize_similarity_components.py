#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Point Cloud Projection and Similarity Visualization
==================================================

Description:
    This script projects a 3D point cloud onto a 2D image plane and visualizes 
    the similarity between the projected image and a real endoscope frame.
    It provides various visualization components including edge detection, 
    gradient comparison, and similarity metrics.

Usage:
    python visualize_projection.py FRAME_NUMBER --dataset DATASET_NAME [options]

Arguments:
    FRAME_NUMBER        : Frame number to process
    --dataset           : Dataset name (e.g., "dataset6") used to construct paths:
                          - Point cloud: data/point_clouds/DATASET/transformed/pose0_transformed.ply
                          - Pose file: data/endoscope/DATASET/frame{FRAME_NUMBER}/frame{FRAME_NUMBER}.yml
                          - Real image: data/endoscope/DATASET/frame{FRAME_NUMBER}/frame{FRAME_NUMBER}.png
    --initial_rotation  : Initial rotation in degrees [rx ry rz] (default: [3.0, 0.5, -3.0])
    --initial_translation: Initial translation in mm [tx ty tz] (default: [5.5, 15, -10])
    --point_cloud       : Override the default point cloud path (optional)
    --pose_file         : Override the default pose file path (optional)

Visualization Components:
    - Enhanced Images (CLAHE and Blur)
    - Edge Comparison (Canny edge detection)
    - Gradient Comparison
    - Gradient Magnitudes

Similarity Metrics:
    - Edge Score: Mean absolute difference between edge maps
    - Gradient Score: Mean absolute difference between gradient magnitudes
    - Overlap Score: Proportion of overlapping edges

Requirements:
    - OpenCV, NumPy, Open3D, SciPy

Author: [Your Name]
Date: [Current Date]
Version: 1.0
"""

import numpy as np
import open3d as o3d
import cv2
import yaml
import os
from scipy.spatial.transform import Rotation
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utilities.projection import project_point_cloud, load_pose
import argparse
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def parse_args():
    """
    Parse command-line arguments.
    
    Returns:
        argparse.Namespace: Parsed command-line arguments
    """
    parser = argparse.ArgumentParser(
        description="Project a point cloud onto a 2D image plane and visualize similarity with a real image."
    )
    parser.add_argument(
        "frame_number", 
        type=int, 
        help="Frame number to process"
    )
    parser.add_argument(
        "--dataset", 
        type=str, 
        required=True,
        help="Dataset name (e.g., 'dataset6') used to construct file paths"
    )
    parser.add_argument(
        "--point_cloud",
        type=str,
        help="Optional: Override path to the point cloud PLY file"
    )
    parser.add_argument(
        "--initial_rotation",
        type=float,
        nargs=3,
        default=[3.0, 0.5, -3.0],
        help="Initial rotation in degrees (rx ry rz)"
    )
    parser.add_argument(
        "--initial_translation",
        type=float,
        nargs=3,
        default=[5.5, 15, -10],
        help="Initial translation in mm (tx ty tz)"
    )
    parser.add_argument(
        "--pose_file",
        type=str,
        help="Optional: Override path to the pose YAML file"
    )
    return parser.parse_args()


def visualize_similarity_components(
    img1,
    img2,
    max_width=2048,
    max_height=1088,
    # Preprocessing parameters
    blur_kernel_size=(5, 5),  # Must be odd number: 3, 5, 7, etc.
    blur_sigma=(1.5, 1.5),  # Gaussian blur intensity
    clahe_clip_limit=(2.0, 3.0),  # Contrast enhancement limit
    clahe_grid_size=(16, 9),  # Contrast enhancement grid size
    # Edge detection parameters
    canny_low=(100, 100),  # Canny low threshold
    canny_high=(200, 200),  # Canny high threshold
    edge_dilate_size=(1, 1),  # Edge dilation kernel size
    edge_dilate_iterations=(1, 1),  # Number of dilations
    # Gradient parameters
    sobel_kernel_size=5,  # Sobel kernel size: 3, 5, 7
    grad_threshold_percentile=90,  # Percentile for gradient threshold
):
    """
    Visualize different components of image similarity with tunable parameters.
    
    This function creates a comprehensive visualization showing:
    1. Enhanced images with CLAHE contrast enhancement
    2. Edge detection comparison between the two images
    3. Gradient comparison showing directional changes
    4. Gradient magnitudes for both images
    
    The visualization helps assess how well the projected point cloud aligns
    with the real endoscope image.
    
    Args:
        img1 (numpy.ndarray): First image (real endoscope image)
        img2 (numpy.ndarray): Second image (projected point cloud)
        max_width (int): Maximum width for visualization window
        max_height (int): Maximum height for visualization window
        blur_kernel_size (tuple): Kernel sizes for Gaussian blur (img1, img2)
        blur_sigma (tuple): Sigma values for Gaussian blur (img1, img2)
        clahe_clip_limit (tuple): CLAHE clip limits for contrast enhancement (img1, img2)
        clahe_grid_size (tuple): CLAHE grid sizes (img1, img2)
        canny_low (tuple): Canny edge low thresholds (img1, img2)
        canny_high (tuple): Canny edge high thresholds (img1, img2)
        edge_dilate_size (tuple): Edge dilation kernel sizes (img1, img2)
        edge_dilate_iterations (tuple): Edge dilation iterations (img1, img2)
        sobel_kernel_size (int): Sobel operator kernel size
        grad_threshold_percentile (float): Percentile threshold for gradient visualization
        
    Returns:
        dict: Dictionary containing computed similarity metrics:
              - edge_score: Mean absolute difference between edge maps
              - gradient_score: Mean absolute difference between gradient magnitudes
              - overlap_score: Proportion of overlapping edges
    """
    logger.info("Visualizing similarity components between real and projected images")
    
    # Convert images to grayscale if needed
    if len(img1.shape) == 3:
        logger.debug("Converting color images to grayscale")
        img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    else:
        img1_gray = img1.copy()
        img2_gray = img2.copy()

    # Set maximum visualization dimensions
    max_height = img1_gray.shape[0] * 3
    max_width = img1_gray.shape[1] * 2

    logger.debug(f"Max visualization dimensions: {max_width}x{max_height}")

    # Calculate scaling factor for visualization
    scale = min(
        max_width / (img1_gray.shape[1] * 2), max_height / (img1_gray.shape[0] * 3)
    )
    logger.debug(f"Using scale factor: {scale}")
    new_width = int(img1_gray.shape[1] * scale)
    new_height = int(img1_gray.shape[0] * scale)

    # Resize images for visualization
    img1_gray = cv2.resize(img1_gray, (new_width, new_height))
    img2_gray = cv2.resize(img2_gray, (new_width, new_height))

    # Step 1: Enhance contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization)
    logger.debug(f"Applying CLAHE with clip limits {clahe_clip_limit} and grid sizes {clahe_grid_size}")
    clahe_img1 = cv2.createCLAHE(
        clipLimit=clahe_clip_limit[0],
        tileGridSize=(clahe_grid_size[0], clahe_grid_size[0]),
    )
    img1_clahe = clahe_img1.apply(img1_gray)

    clahe_img2 = cv2.createCLAHE(
        clipLimit=clahe_clip_limit[1],
        tileGridSize=(clahe_grid_size[1], clahe_grid_size[1]),
    )
    img2_clahe = clahe_img2.apply(img2_gray)

    # Step 2: Apply Gaussian blur to reduce noise
    logger.debug(f"Applying Gaussian blur with kernel sizes {blur_kernel_size} and sigma {blur_sigma}")
    img1_blur = cv2.GaussianBlur(
        img1_clahe, (blur_kernel_size[0], blur_kernel_size[0]), blur_sigma[0]
    )
    img2_blur = cv2.GaussianBlur(
        img2_clahe, (blur_kernel_size[1], blur_kernel_size[1]), blur_sigma[1]
    )

    # Step 3: Edge Detection using Canny
    logger.debug(f"Detecting edges with thresholds {canny_low} and {canny_high}")
    edges1 = cv2.Canny(img1_blur, canny_low[0], canny_high[0])
    edges2 = cv2.Canny(img2_blur, canny_low[1], canny_high[1])

    # Optional: Dilate edges to make them more visible
    if edge_dilate_size[0] > 0:
        logger.debug(f"Dilating edges with kernel sizes {edge_dilate_size} and iterations {edge_dilate_iterations}")
        kernel = np.ones((edge_dilate_size[0], edge_dilate_size[0]), np.uint8)
        edges1 = cv2.dilate(edges1, kernel, iterations=edge_dilate_iterations[0])
    if edge_dilate_size[1] > 0:
        kernel = np.ones((edge_dilate_size[1], edge_dilate_size[1]), np.uint8)
        edges2 = cv2.dilate(edges2, kernel, iterations=edge_dilate_iterations[1])

    # Create edge visualization (Red/Blue/Green for image1/image2/overlap)
    logger.debug("Creating edge comparison visualization")
    edge_compare = np.zeros((new_height, new_width, 3), dtype=np.uint8)
    edge_compare[edges1 > 0] = [0, 0, 255]  # Red for first image
    edge_compare[edges2 > 0] = [255, 0, 0]  # Blue for second image
    overlap = (edges1 > 0) & (edges2 > 0)
    edge_compare[overlap] = [0, 255, 0]  # Green for overlap

    # Step 4: Calculate gradients using Sobel operator
    logger.debug(f"Computing gradients with Sobel kernel size {sobel_kernel_size}")
    sobelx1 = cv2.Sobel(img1_blur, cv2.CV_64F, 1, 0, ksize=sobel_kernel_size)
    sobely1 = cv2.Sobel(img1_blur, cv2.CV_64F, 0, 1, ksize=sobel_kernel_size)
    sobelx2 = cv2.Sobel(img2_blur, cv2.CV_64F, 1, 0, ksize=sobel_kernel_size)
    sobely2 = cv2.Sobel(img2_blur, cv2.CV_64F, 0, 1, ksize=sobel_kernel_size)

    # Compute gradient magnitudes
    mag1 = np.sqrt(sobelx1**2 + sobely1**2)
    mag2 = np.sqrt(sobelx2**2 + sobely2**2)

    # Normalize and threshold gradients based on percentile
    logger.debug(f"Thresholding gradients at {grad_threshold_percentile} percentile")
    mag1_vis = cv2.normalize(mag1, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    mag2_vis = cv2.normalize(mag2, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    grad_threshold = np.percentile(mag1_vis, grad_threshold_percentile)

    # Create gradient comparison visualization
    logger.debug("Creating gradient comparison visualization")
    grad_compare = np.zeros((new_height, new_width, 3), dtype=np.uint8)
    grad_compare[mag1_vis > grad_threshold] = [0, 0, 255]  # Red for first image
    grad_compare[mag2_vis > grad_threshold] = [255, 0, 0]  # Blue for second image
    grad_overlap = (mag1_vis > grad_threshold) & (mag2_vis > grad_threshold)
    grad_compare[grad_overlap] = [0, 255, 0]  # Green for overlap

    # Step 5: Create visualization grid with preprocessed images
    logger.debug("Assembling final visualization grid")
    row1 = np.hstack(
        [
            cv2.cvtColor(img1_clahe, cv2.COLOR_GRAY2BGR),
            cv2.cvtColor(img2_clahe, cv2.COLOR_GRAY2BGR),
        ]
    )
    row2 = np.hstack([edge_compare, grad_compare])
    row3 = np.hstack(
        [
            cv2.cvtColor(mag1_vis, cv2.COLOR_GRAY2BGR),
            cv2.cvtColor(mag2_vis, cv2.COLOR_GRAY2BGR),
        ]
    )

    visualization = np.vstack([row1, row2, row3])

    # Add labels to the visualization
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8 * scale
    color = (255, 255, 255)
    thickness = max(1, int(2 * scale))
    height = visualization.shape[0]

    cv2.putText(
        visualization, "Enhanced Images", (10, 30), font, font_scale, color, thickness
    )
    cv2.putText(
        visualization,
        "Edge & Gradient Comparison (Red/Blue/Green)",
        (10, height // 3 + 30),
        font,
        font_scale,
        color,
        thickness,
    )
    cv2.putText(
        visualization,
        "Gradient Magnitudes",
        (10, 2 * height // 3 + 30),
        font,
        font_scale,
        color,
        thickness,
    )

    # Display the visualization
    logger.info("Displaying similarity visualization")
    cv2.namedWindow("Similarity Components", cv2.WINDOW_NORMAL)
    cv2.resizeWindow(
        "Similarity Components", visualization.shape[1], visualization.shape[0]
    )
    cv2.imshow("Similarity Components", visualization)
    cv2.waitKey(1)

    # Step 6: Compute similarity metrics
    logger.debug("Computing similarity metrics")
    edge_score = np.mean(cv2.absdiff(edges1, edges2))
    gradient_score = np.mean(np.abs(mag1 - mag2))
    overlap_score = np.sum(overlap) / (np.sum(edges1 > 0) + np.sum(edges2 > 0) + 1e-6)
    
    logger.info(f"Computed metrics - Edge score: {edge_score:.4f}, Gradient score: {gradient_score:.4f}, Overlap score: {overlap_score:.4f}")

    return {
        "edge_score": edge_score,
        "gradient_score": gradient_score,
        "overlap_score": overlap_score,
    }


if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_args()
    frame_number = args.frame_number
    dataset = args.dataset
    
    logger.info(f"Processing frame {frame_number} from dataset '{dataset}'")

    # Define the image size (height, width)
    image_size = (1088, 2048)
    logger.info(f"Using image size: {image_size[1]}x{image_size[0]}")

    # Define camera intrinsic matrix
    camera_matrix = np.array(
        [
            [1.15734935e03, 0.00000000e00, 1.02914674e03],
            [0.00000000e00, 1.15627765e03, 5.27741541e02],
            [0.00000000e00, 0.00000000e00, 1.00000000e00],
        ]
    )

    # Define distortion coefficients
    dist_coeffs = np.array(
        [
            [
                -0.4894732329144252,      # k1 - reduce radial distortion
                0.32992541053980134,      # k2
                -0.0010032472743986569,   # p1 - keep tangential distortion
                -0.00014656349448021337,  # p2
                -0.14193195002328468 * 0.1,  # k3 - reduce higher-order radial distortion
            ]
        ]
    )

    # Log camera intrinsics
    logger.info("Camera Intrinsics:")
    logger.info(f"Focal length X: {camera_matrix[0,0]:.2f}")
    logger.info(f"Focal length Y: {camera_matrix[1,1]:.2f}")
    logger.info(f"Principal point X: {camera_matrix[0,2]:.2f}")
    logger.info(f"Principal point Y: {camera_matrix[1,2]:.2f}")

    # Construct file paths using the dataset parameter
    # Point cloud path
    point_cloud_path = (
        args.point_cloud
        if args.point_cloud
        else f"./raw_data/point_clouds/{dataset}/transformed/pose_{frame_number}_transformed.ply"
    )
    
    # Pose file path
    pose_file_path = (
        args.pose_file
        if args.pose_file
        else f"./raw_data/endoscope/{dataset}/frame{frame_number}/frame{frame_number}.yml"
    )
    
    # Real image path
    real_image_path = f"./raw_data/endoscope/{dataset}/frame{frame_number}/frame{frame_number}.png"
    
    # Verify file paths exist
    for path in [point_cloud_path, pose_file_path, real_image_path]:
        if not os.path.exists(path):
            logger.error(f"File not found: {path}")
            raise FileNotFoundError(f"Could not find the required file: {path}")
 
    # Load the point cloud
    logger.info(f"Loading point cloud from: {point_cloud_path}")
    pcd = o3d.io.read_point_cloud(point_cloud_path)
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors) if pcd.has_colors() else None
    logger.info(f"Loaded point cloud with {len(points)} points")

    # Load camera pose using the pose file and initial transformation parameters
    logger.info(f"Loading camera pose from: {pose_file_path}")
    logger.info(f"Using initial rotation: {args.initial_rotation} degrees")
    logger.info(f"Using initial translation: {args.initial_translation} mm")
    
    camera_pose = load_pose(
        pose_file_path,
        rotation_deg=args.initial_rotation,
        trans_offset_mm=args.initial_translation,
    )
    logger.info("Camera pose loaded and adjusted with initial parameters")

    # Project the point cloud to create a 2D image
    logger.info("Projecting point cloud onto the image plane...")
    projected_image = project_point_cloud(
        points, colors, camera_matrix, dist_coeffs, camera_pose, image_size
    )
    logger.info("Point cloud projection complete")
    
    # Load the real endoscope image for comparison
    logger.info(f"Loading real endoscope image from: {real_image_path}")
    real_image = cv2.imread(real_image_path)

    if real_image is None:
        logger.error(f"Could not read image file: {real_image_path}")
        raise FileNotFoundError(f"Could not read image file: {real_image_path}")

    # Resize real image if needed to match the target size
    if real_image.shape[:2] != image_size:
        logger.info(f"Resizing real image from {real_image.shape[:2]} to {image_size}")
        real_image = cv2.resize(real_image, (image_size[1], image_size[0]))

    # Visualize similarity between the real and projected images
    logger.info("Analyzing similarity between real and projected images...")
    scores = visualize_similarity_components(
        real_image,
        projected_image,
        blur_kernel_size=(9, 9),             # Smaller blur kernel
        blur_sigma=(2.0, 2.0),               # Lighter blur
        clahe_clip_limit=(2.0, 3.0),         # More contrast enhancement
        clahe_grid_size=(16, 16),            # Contrast enhancement grid size
        canny_low=(50, 100),                 # Lower thresholds
        canny_high=(80, 220),                # Higher thresholds
        edge_dilate_size=(3, 3),             # Edge dilation size
        edge_dilate_iterations=(2, 2),       # Number of dilations
        grad_threshold_percentile=85,        # Less selective gradient threshold
    )

    # Print similarity scores
    print("\nImage Similarity Scores:")
    print(f"Edge score: {scores['edge_score']:.4f} (lower is better)")
    print(f"Gradient score: {scores['gradient_score']:.4f} (lower is better)")
    print(f"Overlap score: {scores['overlap_score']:.4f} (higher is better)")

    # Wait for key press to close
    print("\nPress any key to exit...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    logger.info("Visualization closed. Processing complete.")
