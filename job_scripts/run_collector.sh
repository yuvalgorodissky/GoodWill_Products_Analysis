#!/bin/bash

################################################################################################
### sbatch configuration parameters must start with #SBATCH and must precede any other commands.
### To ignore, just add another # - like so: ##SBATCH
################################################################################################

#SBATCH --partition main			### specify partition name where to run a job. short: 7 days limit; gtx1080: 7 days; debug: 2 hours limit and 1 job at a time
#SBATCH --time 7-00:00:00			### limit the time of job running. Make sure it is not greater than the partition time limit!! Format: D-H:MM:SS
#SBATCH --job-name yuvalgor
#SBATCH --output /sise/eliorsu-group/yuvalgor/courses/Data-mining-in-Big-Data/job_scripts/out/collector_items-job-%J.out			### output log for running job - %J for job number
#SBATCH --cpus-per-task=4      ### number of CPUs allocated to your job

# Note: the following 5 lines are commented out
##SBATCH --mail-user=user@post.bgu.ac.il	### user's email for sending job status messages
##SBATCH --mail-type=ALL			### conditions for sending the email. ALL,BEGIN,END,FAIL, REQUEU, NONE
##SBATCH --mem=60G				### ammount of RAM memory


###############  Following lines will be executed by a compute node    ########################

### Print some data to output file ###
echo `date`
echo -e "\nSLURM_JOBID:\t\t" $SLURM_JOBID
echo -e "SLURM_JOB_NODELIST:\t" $SLURM_JOB_NODELIST "\n\n"

### Start your code below ####
module load anaconda				### load anaconda module (must be present when working with conda environments)
source activate /sise/eliorsu-group/yuvalgor/.conda/envs/yuvalgor_env				### activate a conda environment, replace my_env with your conda environment
which python					### print python path to output file
### Read arguments passed from sbatch
START_ITEM_ID="$START_ITEM_ID"
MAX_ITEMS="$MAX_ITEMS"
OUTPUT_FILE="$OUTPUT_FILE"

echo "START_ITEM_ID: ${START_ITEM_ID}"
echo "MAX_ITEMS: ${MAX_ITEMS}"
echo "OUTPUT_FILE: ${OUTPUT_FILE}"

### Execute Python script with arguments ###
python /sise/eliorsu-group/yuvalgor/courses/Data-mining-in-Big-Data/src/collect_data.py \
        --start_item_id "${START_ITEM_ID}" \
        --max_items "${MAX_ITEMS}" \
        --output_file "${OUTPUT_FILE}"
