"""
Split Clean Dataset for Machine Learning
========================================

Description:
    This script prepares the final, clean dataset for use in a machine learning
    pipeline. It takes the data from the `depth_data` directory and splits it
    into training, validation, and test sets based on predefined dataset IDs.

    The script creates a new directory structure suitable for use with common
    deep learning frameworks like PyTorch, organizing the images and poses into
    separate subdirectories within each split (train/val/test).

Usage:
    python ml_tools/1_split_for_training.py --source_dir <path_to_depth_data> --dest_dir <output_directory>

"""
import os
import shutil
import argparse
import glob
from tqdm import tqdm

def create_dataset_split(source_dir, dest_dir):
    """
    Splits the cleaned dataset into train, validation, and test sets
    with a structure suitable for PyTorch training.
    """
    # --- Configuration ---
    VAL_IDS = [5, 11, 20, 24]
    TEST_IDS = [6, 9, 12, 21, 27]
    ALL_DATASETS = range(1, 28)

    print("Starting dataset split...")
    print(f"Source directory: {source_dir}")
    print(f"Destination directory: {dest_dir}")

    # --- Create destination directories ---
    for split in ['train', 'val', 'test']:
        for data_type in ['images', 'poses']:
            os.makedirs(os.path.join(dest_dir, split, data_type), exist_ok=True)

    # --- Process and split data ---
    for i in tqdm(ALL_DATASETS, desc="Processing datasets"):
        dataset_name = f"dataset{i}"
        
        # Determine which split this dataset belongs to
        if i in VAL_IDS:
            split_name = 'val'
        elif i in TEST_IDS:
            split_name = 'test'
        else:
            split_name = 'train'
            
        print(f"\nProcessing {dataset_name} -> {split_name} split")

        source_dataset_path = os.path.join(source_dir, 'endoscope', dataset_name)
        if not os.path.isdir(source_dataset_path):
            print(f"  Warning: Source directory not found, skipping: {source_dataset_path}")
            continue

        # Find all frame directories (e.g., frame0, frame1, ...)
        frame_dirs = glob.glob(os.path.join(source_dataset_path, 'frame*'))

        for frame_dir in tqdm(frame_dirs, desc=f"  Frames in {dataset_name}", leave=False):
            frame_name = os.path.basename(frame_dir) # e.g., 'frame123'
            
            # Define source paths
            source_img_path = os.path.join(frame_dir, f"{frame_name}.png")
            source_pose_path = os.path.join(frame_dir, f"{frame_name}.yml")

            # Define destination paths with new, unique names
            dest_img_path = os.path.join(dest_dir, split_name, 'images', f"{dataset_name}_{frame_name}.png")
            dest_pose_path = os.path.join(dest_dir, split_name, 'poses', f"{dataset_name}_{frame_name}.yml")

            # Copy files if they exist
            if os.path.exists(source_img_path):
                shutil.copy(source_img_path, dest_img_path)
            else:
                print(f"    Warning: Image not found: {source_img_path}")

            if os.path.exists(source_pose_path):
                shutil.copy(source_pose_path, dest_pose_path)
            else:
                print(f"    Warning: Pose file not found: {source_pose_path}")

    print("\nDataset splitting complete!")
    print(f"Data is ready for training in: {dest_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare dataset for PyTorch by splitting into train/val/test.")
    parser.add_argument("--source_dir", type=str, default="depth_data",
                        help="Path to the source 'depth_data' directory.")
    parser.add_argument("--dest_dir", type=str, default="pytorch_dataset",
                        help="Path to the destination directory for the split dataset.")
    
    args = parser.parse_args()
    
    create_dataset_split(args.source_dir, args.dest_dir)
