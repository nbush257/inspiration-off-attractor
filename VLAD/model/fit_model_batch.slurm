#!/bin/bash

#SBATCH --partition=cpu-core-sponsored
#SBATCH --account=cpu-medullary-sponsored
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=256G
#SBATCH --time=48:00:00
#SBATCH --array=0-22
#SBATCH --mail-user=nicholas.bush@seattlechildrens.org
#SBATCH --mail-type=ALL
#SBATCH --job-name=fit_rslds

# ^ Resource request
#------------------------------------------------------------------------------
# v Execution script
source activate iblenv
python fit_model.py $SLURM_ARRAY_TASK_ID