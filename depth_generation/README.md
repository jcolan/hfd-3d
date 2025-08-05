# Depth Generation Pipeline

This directory contains the core scripts for generating depth maps and other data products by aligning the master point cloud with a sequence of endoscopic video frames.

## Pipeline Steps

The scripts are designed to be run in numerical order:

### 1. `1_find_initial_offset.py`

*   **Purpose**: To interactively find a global transformation (rotation and translation) that provides a coarse alignment between the master point cloud and the endoscope's coordinate system. This is a crucial first step before fine-tuning.
*   **Usage**:
    ```bash
    python depth_generation/1_find_initial_offset.py --dataset <dataset_name> --frame <frame_number_to_use_for_alignment>
    ```

### 2. `2_apply_initial_offset.py`

*   **Purpose**: To apply the global offset found in the previous step to the camera poses of all frames in the dataset, creating `_initial.yml` pose files.
*   **Usage**:
    ```bash
    python depth_generation/2_apply_initial_offset.py --dataset <dataset_name>
    ```

### 3. `3_optimize_poses.py`

*   **Purpose**: To automatically and precisely refine the camera pose for each individual frame. It uses a coarse-to-fine grid search to maximize the edge-based similarity between the real endoscope image and the view projected from the point cloud. This step is computationally intensive but critical for accuracy.
*   **Usage**:
    ```bash
    python depth_generation/3_optimize_poses.py --dataset <dataset_name> --frame_start <start_frame> --frame_end <end_frame>
    ```

### 4. `4_generate_depth_maps.py`

*   **Purpose**: Using the final, optimized poses, this script projects the master point cloud to generate the primary data products: the 16-bit depth map and the corresponding projected color view.
*   **Usage**:
    ```bash
    python depth_generation/4_generate_depth_maps.py --dataset <dataset_name> --frame_start <start_frame> --frame_end <end_frame> [options]
    ```
*   **Helper Script**: `4_run_depth_generation.sh` is provided to easily run this step for a range of datasets.

### 5. `5_crop_final_images.py`

*   **Purpose**: To crop the original color images, the newly generated depth maps, and the projected views into a centered square. This removes the circular endoscopic vignette and prepares the data for use in standard computer vision models.
*   **Usage**:
    ```bash
    python depth_generation/5_crop_final_images.py --dataset <dataset_name> --frame_start <start_frame> --frame_end <end_frame>
    ```

### 6. `6_build_clean_dataset.py`

*   **Purpose**: The final step of the generation pipeline. It gathers all the final, cropped data products (RGB images, depth maps, projected views, poses) and organizes them into a simple, clean directory structure under `/depth_data`.
*   **Usage**:
    ```bash
    python depth_generation/6_build_clean_dataset.py --dataset <dataset_name> --frame_start <start_frame> --frame_end <end_frame>
    ```

## Output

The final output of this pipeline is a complete, clean dataset located in the `/depth_data` directory. This dataset is ready for the next stages: **Validation** and **ML Preparation**.
