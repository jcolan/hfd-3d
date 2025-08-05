"""
Simple Point Cloud Viewer
=========================

Description:
    A simple command-line utility to quickly load and visualize a 3D point cloud
    from a .ply file using the Open3D library.

Usage:
    python utilities/view_cloud.py <path_to_point_cloud.ply>

"""
import open3d as o3d
import argparse
import sys
import os

def view_point_cloud(file_path):
    """Loads and displays a single point cloud file."""
    print(f"Attempting to load point cloud from: {file_path}")
    
    if not os.path.exists(file_path):
        print(f"Error: File not found at '{file_path}'", file=sys.stderr)
        sys.exit(1)
        
    try:
        pcd = o3d.io.read_point_cloud(file_path)
        if not pcd.has_points():
            print(f"Error: The file '{file_path}' was loaded, but it contains no points.", file=sys.stderr)
            sys.exit(1)
            
        print("Successfully loaded point cloud. Displaying...")
        print("Press 'Q' in the window to close.")
        
        o3d.visualization.draw_geometries([pcd])
        
        print("Visualization window closed.")

    except Exception as e:
        print(f"An error occurred while trying to read or display the file: {e}", file=sys.stderr)
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(
        description="Load and visualize a 3D point cloud from a .ply file.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        'file_path', 
        type=str, 
        help="The absolute or relative path to the .ply file to be visualized."
    )
    args = parser.parse_args()
    
    view_point_cloud(args.file_path)

if __name__ == "__main__":
    main()
