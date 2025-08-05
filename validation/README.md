# Dataset Validation Tools

This directory contains scripts to perform both quantitative and qualitative validation of the generated dataset, ensuring its quality, accuracy, and integrity.

## Scripts

### `run_quality_validation.py`

*   **Purpose**: To perform a comprehensive, quantitative analysis of the dataset quality. It calculates a variety of metrics and generates plots and a summary report.
*   **Key Metrics**:
    *   **Geometric Fidelity**: Planarity error of the master point cloud.
    *   **Alignment Accuracy**: Automated keypoint reprojection error and Gradient-based Normalized Cross-Correlation (NCC) between RGB and depth images.
    *   **Depth Quality**: The density (completeness) of the depth maps.
    *   **Statistics**: Depth value distribution histograms.
*   **Usage**:
    ```bash
    # Run all tests on the 'ex-vivo' group of datasets
    python validation/run_quality_validation.py --dataset_root ./depth_data --dataset_group ex-vivo

    # Run only the reprojection test on dataset 2
    python validation/run_quality_validation.py --dataset_root ./depth_data --dataset_group 2 --tests reprojection
    ```

### `verify_clean_data.py`

*   **Purpose**: To provide a visual, qualitative tool to verify the final alignment. For a given frame, it loads the final data, regenerates the projected view and depth map on-the-fly from the master cloud and pose, and displays them side-by-side for comparison.
*   **Usage**:
    ```bash
    python validation/verify_clean_data.py --dataset <dataset_name> --frame_start <start_frame> --frame_end <end_frame>
    ```

### `run_verification.sh`

*   **Purpose**: A simple shell script to run the `verify_clean_data.py` script across a range of dataset numbers.
*   **Usage**:
    ```bash
    ./validation/run_verification.sh <start_dataset_number> <end_dataset_number>
    ```

## Output

The `run_quality_validation.py` script saves its reports and plots to the `/results` directory, organized by timestamp.
