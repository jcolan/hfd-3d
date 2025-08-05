# Point Cloud Processing Pipeline

This directory contains the scripts for the first major stage of the data generation pipeline: processing and merging multiple raw point clouds into a single, high-quality master point cloud for a scene.

## Pipeline Steps

The scripts are designed to be run in numerical order:

### 1. `1_generate_initial_poses.py`

*   **Purpose**: To calculate the initial transformation poses for each raw point cloud. It uses a reference pose from the robot's coordinate system and a new desired reference pose to establish a common frame of reference for all clouds.
*   **Usage**:
    ```bash
    python point_cloud/1_generate_initial_poses.py --dataset <dataset_name> --num_poses <number_of_poses>
    ```

### 2. `2_refine_pose_manually.py`

*   **Purpose**: To provide an interactive 3D environment for manually refining the alignment between two point clouds. This is crucial for correcting any drift or inaccuracies from the initial automated poses. You align a "movable" cloud against a static "base" cloud.
*   **Usage**:
    ```bash
    python point_cloud/2_refine_pose_manually.py --dataset <dataset_name> --base_pose_num <base_cloud_index> --align_pose_num <cloud_to_align_index>
    ```

### 3. `3_apply_poses_to_clouds.py`

*   **Purpose**: To apply the final, refined transformation matrices to their corresponding raw point clouds. It intelligently selects the best available pose, prioritizing manually refined poses (`_refined.yaml`) over initial ones (`_initial.yaml`).
*   **Usage**:
    ```bash
    python point_cloud/3_apply_poses_to_clouds.py --dataset <dataset_name>
    ```

### 4. `4_merge_transformed_clouds.py`

*   **Purpose**: The final step in this stage. It takes all the transformed point clouds and merges them into a single, dense master point cloud using the Colored ICP (Iterative Closest Point) algorithm. This ensures a highly accurate and detailed representation of the scene.
*   **Usage**:
    ```bash
    python point_cloud/4_merge_transformed_clouds.py --dataset <dataset_name> --quality <draft|standard|fine>
    ```

## Output

The final output of this pipeline is a single merged point cloud (e.g., `merged_cloud_standard.ply`) located in `raw_data/point_clouds/<dataset_name>/merged/`. This master cloud is the primary input for the next major stage of the project: **Depth Generation**.
