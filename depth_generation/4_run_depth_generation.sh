#!/bin/bash

# This script runs the depth map and projected view generation for a range of datasets.
# Usage: ./4_run_depth_generation.sh <start_dataset_number> <end_dataset_number> [pose_type]

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <start_dataset_number> <end_dataset_number> [pose_type]"
    echo "pose_type (optional): 'initial' or 'optimized'. Defaults to 'optimized'."
    exit 1
fi

START=$1
END=$2
POSE_TYPE=${3:-initial}

for i in $(seq $START $END)
do
   dataset_name="dataset$i"
   echo "--- Processing $dataset_name with $POSE_TYPE poses ---"
   
   # You can customize the parameters below for each run
   python depth_generation/4_generate_depth_maps.py \
       --dataset "$dataset_name" \
       --frame_start 0 \
       --frame_end 511 \
       --quality "fine" \
       --point_size 0 \
       --near_clip 30 \
       --remove_outliers \
       --fill_holes \
       --hole_fill_kernel_size 4 \
       --pose_type "$POSE_TYPE"
    #    --no_smooth \

   echo "--- Finished processing $dataset_name ---"
done

echo "All datasets processed."

