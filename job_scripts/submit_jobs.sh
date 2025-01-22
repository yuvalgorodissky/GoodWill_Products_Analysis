#!/bin/bash

# Define parameters
START_ITEM_ID=217000000
MAX_ITEMS=1600000
NUM_JOBS=30
ITEMS_PER_JOB=$((MAX_ITEMS / NUM_JOBS))

# Directory for job scripts and output
SCRIPT_DIR="/sise/eliorsu-group/yuvalgor/courses/Data-mining-in-Big-Data/job_scripts"
OUTPUT_DIR="/sise/eliorsu-group/yuvalgor/courses/Data-mining-in-Big-Data/datasets"

# Loop to create and submit jobs
for i in $(seq 0 $((NUM_JOBS - 1)))
do
    # Calculate job-specific starting item ID
    JOB_START_ITEM_ID=$((START_ITEM_ID + i * ITEMS_PER_JOB))

    # Create unique output file for each job
    OUTPUT_FILE="$OUTPUT_DIR/goodwill_items_job_${i}.csv"

    # Submit job using sbatch
    sbatch --export=START_ITEM_ID=${JOB_START_ITEM_ID},MAX_ITEMS=${ITEMS_PER_JOB},OUTPUT_FILE=${OUTPUT_FILE} \
           $SCRIPT_DIR/run_collector.sh

done
