#!/bin/bash

#SBATCH --partition=cpu-core-sponsored
#SBATCH --account=cpu-medullary-sponsored
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --array=0-22
#SBATCH --mail-user=nicholas.bush@seattlechildrens.org
#SBATCH --mail-type=ALL
#SBATCH --job-name=analyze_single_rec

# ^ Resource request
#------------------------------------------------------------------------------
# v Execution script
source activate iblenv
python analyze_single_rec.py single $SLURM_ARRAY_TASK_ID -n