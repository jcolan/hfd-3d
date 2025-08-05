# Machine Learning Tools

This directory contains scripts for preparing the final, clean dataset for use in machine learning pipelines.

## Scripts

### 1. `1_split_for_training.py`

*   **Purpose**: To take the final datasets from the `depth_data` directory and split them into training, validation, and test sets. It organizes the data into a structure that is commonly used by deep learning frameworks like PyTorch, with separate folders for images and poses within each split.
*   **Configuration**: The script contains hardcoded lists (`VAL_IDS`, `TEST_IDS`) that define which datasets belong to the validation and test sets. All other datasets are assigned to the training set.
*   **Usage**:
    ```bash
    python ml_tools/1_split_for_training.py --source_dir <path_to_depth_data> --dest_dir <output_directory_for_split_data>
    ```

## Output

The output is a new directory (e.g., `pytorch_dataset`) with the following structure:

```
<dest_dir>/
├── train/
│   ├── images/
│   └── poses/
├── val/
│   ├── images/
│   └── poses/
└── test/
    ├── images/
    └── poses/
```
