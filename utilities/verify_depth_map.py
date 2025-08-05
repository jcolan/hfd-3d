"""
Depth Map Visualization and Verification Tool
===========================================

Description:
    This script provides an interactive tool to visualize 16-bit depth maps and
    optionally reconstruct a 3D point cloud from a single depth map. This allows
    for a comprehensive visual and geometric verification of the final data products.

Usage:
    # To visualize a depth map with an interactive depth-on-hover feature
    python utilities/verify_depth_map.py --dataset <name> --frame <num>

    # To reconstruct and save a point cloud from a depth map
    python utilities/verify_depth_map.py --dataset <name> --frame <num> --reconstruct

"""
import cv2
import numpy as np
import argparse
import glob
import os
import open3d as o3d

def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Visualize a 16-bit depth map and optionally reconstruct a point cloud.")
    parser.add_argument("--dataset", type=str, required=True, help="The name of the dataset (e.g., 'dataset1').")
    parser.add_argument("--frame", type=int, required=True, help="The frame number to visualize or reconstruct.")
    parser.add_argument("--quality", type=str, help="Optional quality suffix (e.g., 'uhd', 'debug').")
    parser.add_argument("--reconstruct", action="store_true", help="If set, reconstruct and save the point cloud.")
    parser.add_argument("--output", type=str, help="Optional output path for the reconstructed point cloud.")
    parser.add_argument("--no-visualize", action="store_true", help="Do not visualize the reconstructed point cloud.")
    return parser.parse_args()

def find_depth_map(dataset, frame, quality=None):
    """Finds the depth map file."""
    base_path = f"raw_data/endoscope/{dataset}/frame{frame}"
    if not os.path.isdir(base_path):
        print(f"Error: Directory not found: {base_path}")
        return None
    if quality:
        search_pattern = f"{base_path}/frame{frame}_depth_*_{quality}.png"
    else:
        search_pattern = f"{base_path}/frame{frame}_depth*.png"
    files = glob.glob(search_pattern)
    if not files:
        print("No depth map found for the specified criteria.")
        return None
    return files[0]

def reconstruct_point_cloud(depth_map, camera_matrix, dist_coeffs):
    """Reconstructs a point cloud from a depth map."""
    h, w = depth_map.shape
    points = []
    # Create a grid of pixel coordinates
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    u = u.flatten()
    v = v.flatten()
    z = depth_map.flatten() / 65535.0 * 300.0  # Convert to mm

    # Remove points with no depth
    valid = z > 0
    u, v, z = u[valid], v[valid], z[valid]

    # Back-project to 3D
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]

    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    points = np.vstack((x, y, z)).T

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd

def mouse_callback(event, x, y, flags, param):
    """Handles mouse events to display depth information."""
    depth_map = param['depth_map']
    display_image = param['display'].copy()
    h, w = depth_map.shape
    display_h, display_w = display_image.shape[:2]
    orig_x = int(x * (w / display_w))
    orig_y = int(y * (h / display_h))
    if 0 <= orig_y < h and 0 <= orig_x < w:
        depth_value = depth_map[orig_y, orig_x]
        depth_mm = depth_value * (300.0 / 65535.0)
        text = f"Depth: {depth_mm:.2f} mm"
        cv2.putText(display_image, text, (x + 15, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imshow("Depth Map Viewer", display_image)

def main():
    """Main function."""
    args = parse_args()
    depth_map_path = find_depth_map(args.dataset, args.frame, args.quality)
    if not depth_map_path:
        return

    depth_map = cv2.imread(depth_map_path, cv2.IMREAD_UNCHANGED)
    if depth_map is None:
        print(f"Error: Failed to load the depth map from '{depth_map_path}'.")
        return

    if args.reconstruct:
        camera_matrix = np.array([[1.15734935e03, 0.00000000e00, 1.02914674e03],
                                  [0.00000000e00, 1.15627765e03, 5.27741541e02],
                                  [0.00000000e00, 0.00000000e00, 1.00000000e00]])
        dist_coeffs = np.array([[-0.4894732329144252, 0.32992541053980134, -0.0010032472743986569, -0.00014656349448021337, -0.14193195002328468 * 0.1]])
        
        print("Reconstructing point cloud...")
        pcd = reconstruct_point_cloud(depth_map, camera_matrix, dist_coeffs)
        
        if args.output:
            output_path = args.output
        else:
            output_path = os.path.join(os.path.dirname(depth_map_path), f"frame{args.frame}_reconstructed.ply")
            
        o3d.io.write_point_cloud(output_path, pcd)
        print(f"Reconstructed point cloud saved to: {output_path}")

        if not args.no_visualize:
            print("Visualizing reconstructed point cloud...")
            o3d.visualization.draw_geometries([pcd])
    else:
        # Normalize for visualization and apply a colormap
        depth_display = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        depth_colored = cv2.applyColorMap(depth_display, cv2.COLORMAP_JET)

        # Resize for better screen fit if necessary
        h, w = depth_map.shape
        max_display_size = 1000
        scale = min(max_display_size / w, max_display_size / h)
        if scale < 1:
            display_size = (int(w * scale), int(h * scale))
            depth_colored = cv2.resize(depth_colored, display_size, interpolation=cv2.INTER_AREA)
        
        # Add informational text
        info_text = [
            f"File: {os.path.basename(depth_map_path)}",
            f"Frame: {args.frame}, Dataset: {args.dataset}",
            "Depth Range: 0 - 300 mm",
            "Hover to see depth values. Press any key to exit."
        ]
        
        for i, text in enumerate(info_text):
            cv2.putText(depth_colored, text, (10, 20 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(depth_colored, text, (10, 20 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # Setup window and mouse callback
        cv2.namedWindow("Depth Map Viewer")
        callback_params = {'depth_map': depth_map, 'display': depth_colored}
        cv2.setMouseCallback("Depth Map Viewer", mouse_callback, callback_params)
        
        cv2.imshow("Depth Map Viewer", depth_colored)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()