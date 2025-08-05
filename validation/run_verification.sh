#!/bin/bash

# This script runs the clean data verification for a range of datasets.
# Usage: ./validation/run_verification.sh <start_dataset_number> <end_dataset_number>

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <start_dataset_number> <end_dataset_number>"
    exit 1
fi

START=$1
END=$2

for i in $(seq $START $END)
do
   dataset_name="dataset$i"
   echo "--- Verifying $dataset_name ---"
   
   python validation/verify_clean_data.py \
       --dataset "$dataset_name" \
       --frame_start 0 \
       --frame_end 511

   echo "--- Finished verifying $dataset_name ---"
done

echo "All datasets verified."

