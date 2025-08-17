# Monocular Depth Estimation Data Generation Pipeline

This project provides a comprehensive suite of tools to generate high-quality depth data for monocular depth estimation tasks. The pipeline takes multiple registered point clouds and a sequence of endoscopic video frames, and produces a clean, organized dataset containing:

*   Original RGB images (cropped)
*   16-bit depth maps (aligned and cropped)
*   Projected color views from the point cloud (cropped)
*   Precise camera poses for each frame

## Data Access

The project's datasets are available for download. The `depth_data` contains the final processed output of the pipeline, while the `raw_data` contains the original source files.

* **Processed Dataset (`depth_data`)**: This contains the final, clean dataset including cropped RGB images, 16-bit depth maps, projected views, and camera poses.
    * **[Download `depth_data` (~65GB)]([depth_data](https://doi.org/10.7910/DVN/NQFSWQ))**

* **Raw Source Data (`raw_data`)**: This contains the original endoscope video sequences and multiple point clouds used to generate the final dataset.
    * **[Download `raw_data` (~250GB)]([raw_data](https://drive.google.com/file/d/1KVapf6vsxKZ0lsokw6-pBskeOA3ezrZC))**
    * *Note: Access to the raw data may require permission or a Google account.*
      
## Directory Structure

The project is organized into several directories, each responsible for a specific stage of the data generation pipeline:

*   **/raw_data**: Contains the raw input data (endoscope images and point clouds).
*   **/depth_data**: The final output directory for the clean, organized dataset. Its structure is as follows:

```
depth_data/
├── endoscope/
│   ├── dataset1/
│   │   ├── frame0/
│   │   │   ├── frame0_depth.png
│   │   │   ├── frame0_projected.png
│   │   │   └── frame0.yml
│   │   ├── frame1/
│   │   │   └── ...
│   │   └── ...
│   ├── dataset2/
│   │   └── ...
│   └── ...
└── point_clouds/
    ├── dataset1/
    │   └── merged/
    │       └── point_cloud.ply
    ├── dataset2/
    │   └── ...
    └── ...
```

*   **/point\_cloud**: Scripts for processing and merging multiple raw point clouds into a single, dense, and accurate master cloud.
*   **/depth\_generation**: The core pipeline for generating depth maps and projected views by aligning the master point cloud with endoscopic images.
*   **/ml\_tools**: Scripts for preparing the final dataset for machine learning tasks, such as splitting into training, validation, and test sets.
*   **/validation**: Tools for quantitatively and qualitatively validating the quality and accuracy of the generated dataset.
*   **/utilities**: Miscellaneous helper scripts for tasks like fixing incomplete datasets or visualizing data.
*   **/results**: Output directory for validation reports and figures.

## Core Pipeline Overview

The data generation process follows these main steps. Each step corresponds to a directory containing the necessary scripts.

### Step 1: Point Cloud Merging (`point_cloud/`)

The first stage involves taking multiple raw point clouds (captured from different viewpoints) and merging them into a single, coherent master point cloud for the scene.

1.  **Generate Initial Poses** (`1_generate_initial_poses.py`): Calculate initial transformation poses for each point cloud based on robot data.
2.  **Refine Poses Manually** (`2_refine_pose_manually.py`): An interactive 3D tool to manually adjust and refine the alignment between point clouds.
3.  **Apply Poses** (`3_apply_poses_to_clouds.py`): Apply the final (refined) transformations to each raw point cloud.
4.  **Merge Clouds** (`4_merge_transformed_clouds.py`): Merge all transformed point clouds into a single master cloud using Colored ICP for high accuracy.

### Step 2: Depth Data Generation (`depth_generation/`)

This stage uses the master point cloud and the raw endoscope video frames to generate the final depth data.

1.  **Find Initial Offset** (`1_find_initial_offset.py`): Interactively find a global transformation offset to coarsely align the master point cloud with the endoscope's coordinate system.
2.  **Apply Initial Offset** (`2_apply_initial_offset.py`): Apply the global offset to the camera poses of all frames.
3.  **Optimize Poses** (`3_optimize_poses.py`): Automatically and precisely refine the camera pose for each frame by maximizing the edge similarity between the real image and the projected point cloud view.
4.  **Generate Depth Maps** (`4_generate_depth_maps.py`): Use the final, optimized poses to project the master point cloud and generate the 16-bit depth maps and projected color views. The `4_run_depth_generation.sh` script is provided to run this for multiple datasets.
5.  **Crop Final Images** (`5_crop_final_images.py`): Crop the original color images, depth maps, and projected views to a centered square to prepare them for the final dataset.
6.  **Build Clean Dataset** (`6_build_clean_dataset.py`): Copy all the final data products into the `/depth_data` directory with a simple, organized file structure.

### Step 3: Dataset Preparation for ML (`ml_tools/`)

Once the clean dataset is generated, these tools prepare it for use in a machine learning pipeline.

1.  **Split for Training** (`1_split_for_training.py`): Split the datasets within `/depth_data` into `train`, `val`, and `test` sets, organizing them into a structure suitable for frameworks like PyTorch.

## General Usage

It is recommended to run the scripts in the order described in the pipeline overview. Each script includes command-line arguments to specify the dataset and other parameters. Use the `--help` flag for detailed options.

**Example: Running a step**
```bash
python depth_generation/4_generate_depth_maps.py --dataset dataset6 --frame_start 0 --frame_end 511 --quality fine
```

For more detailed instructions, refer to the `README.md` file inside each subdirectory.

## Dataset Composition

The dataset is composed of 27 different samples, organized into the following categories:

| ID | Category                   | Sample                           |
|----|----------------------------|----------------------------------|
| 1  | Basic Phantoms             | Rigid Disc                       |
| 2  | Basic Phantoms             | Suturing Phantom 1               |
| 3  | Basic Phantoms             | Suturing Phantom 2               |
| 4  | Basic Phantoms             | Suturing Phantom 3               |
| 5  | Basic Phantoms             | Suturing Phantom 4               |
| 6  | Basic Phantoms             | Peeling Task Phantom             |
| 7  | Advanced Task Phantoms     | Stomach Phantom                  |
| 8  | Advanced Task Phantoms     | Anastomosis Phantom              |
| 9  | Advanced Task Phantoms     | Cutting Task Simulator           |
| 10 | Porous Surface Phantoms    | Planar Foam                      |
| 11 | Porous Surface Phantoms    | Convoluted Foam 1                |
| 12 | Porous Surface Phantoms    | Convoluted Foam 2                |
| 13 | Fabric Phantoms            | Fabric Phantom Yellow 1          |
| 14 | Fabric Phantoms            | Fabric Phantom Yellow 2          |
| 15 | Fabric Phantoms            | Fabric Phantom Yellow 3          |
| 16 | Fabric Phantoms            | Fabric Phantom Red 1             |
| 17 | Fabric Phantoms            | Fabric Phantom Red 2             |
| 18 | Fabric Phantoms            | Fabric Phantom Red 3             |
| 19 | Fabric Phantoms            | Fabric Phantom Mix 1             |
| 20 | Fabric Phantoms            | Fabric Phantom Mix 2             |
| 21 | Fabric Phantoms            | Fabric Phantom Mix 3             |
| 22 | Ex-Vivo                    | Ex-Vivo Avian Tissue 1           |
| 23 | Ex-Vivo                    | Ex-Vivo Avian Tissue 2           |
| 24 | Ex-Vivo                    | Ex-Vivo Avian Tissue 3           |
| 25 | Ex-Vivo                    | Ex-Vivo Bovine/Porcine Tissue 1  |
| 26 | Ex-Vivo                    | Ex-Vivo Bovine/Porcine Tissue 2  |
| 27 | Ex-Vivo                    | Ex-Vivo Bovine/Porcine Tissue 3  |
