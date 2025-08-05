# Utility Scripts

This directory contains various helper and utility scripts for debugging, visualization, and fixing issues with the dataset.

## Scripts

### `projection.py`

*   **Purpose**: This is a core library module, not a standalone script. It contains the `project_point_cloud` function, which is the fundamental engine for projecting the 3D master point cloud into a 2D image plane. This function is imported and used by numerous scripts in the `depth_generation` and `validation` pipelines.

### `fix_incomplete_dataset.py`

*   **Purpose**: To scan a dataset for frames that are missing their final depth map or projected color image and generate only the missing files. This is highly efficient for fixing corrupted or incomplete runs without reprocessing the entire dataset.
*   **Usage**:
    ```bash
    python utilities/fix_incomplete_dataset.py --dataset <dataset_name> --total_frames <num_frames>
    ```

### `verify_depth_map.py`

*   **Purpose**: An interactive tool to inspect a final 16-bit depth map. It can also reconstruct a 3D point cloud from a single depth map, which is useful for verifying the integrity of the stored depth data.
*   **Usage**:
    ```bash
    # Visualize a depth map
    python utilities/verify_depth_map.py --dataset <dataset_name> --frame <frame_number>

    # Reconstruct a point cloud from the depth map
    python utilities/verify_depth_map.py --dataset <dataset_name> --frame <frame_number> --reconstruct
    ```

### `view_cloud.py`

*   **Purpose**: A simple command-line utility to quickly load and visualize any `.ply` point cloud file.
*   **Usage**:
    ```bash
    python utilities/view_cloud.py <path_to_point_cloud.ply>
    ```

### `visualize_similarity_components.py`

*   **Purpose**: A powerful debugging tool to visualize the components used in the automated pose optimization process. It shows a side-by-side comparison of edge maps, gradients, and other features for the real and projected images, helping to diagnose alignment issues.
*   **Usage**:
    ```bash
    python utilities/visualize_similarity_components.py <frame_number> --dataset <dataset_name>
    ```
