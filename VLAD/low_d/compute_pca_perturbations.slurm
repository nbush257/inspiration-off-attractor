#!/bin/bash

#SBATCH --partition=cpu-core-sponsored
#SBATCH --account=cpu-medullary-sponsored
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --mail-user=nicholas.bush@seattlechildrens.org
#SBATCH --mail-type=ALL
#SBATCH --job-name=compute_pca_perturbations
low_d_dir=$(realpath "${SLURM_SUBMIT_DIR:-$0}")
echo "Running from directory: $low_d_dir"

source activate iblenv
python compute_pca_perturbations.py
echo "DONE"