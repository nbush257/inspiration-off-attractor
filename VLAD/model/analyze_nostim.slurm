#!/bin/bash

#SBATCH --partition=cpu-core-sponsored
#SBATCH --account=cpu-medullary-sponsored
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --array=0-18
#SBATCH --mail-user=nicholas.bush@seattlechildrens.org
#SBATCH --mail-type=ALL
#SBATCH --job-name=analyze_rec_no_stim

# ^ Resource request
#------------------------------------------------------------------------------
# v Execution script
source activate iblenv
python analyze_rec_no_stim_fit.py $SLURM_ARRAY_TASK_ID 