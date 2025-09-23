#!/bin/bash

#SBATCH --partition=cpu-core-sponsored
#SBATCH --account=cpu-medullary-sponsored
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --mail-user=nicholas.bush@seattlechildrens.org
#SBATCH --mail-type=ALL
#SBATCH --job-name=make_all_figs_VLAD

#------------------------------------------------------------------------------
# This script submits a job to generate many of the figures for the VLAD manuscript.
# It assumes:
#       that the raw data repository is correct and referenced in the utils.py file
#       that intermediate data have already been generated


source activate iblenv
VLAD_DIR=$(realpath "${SLURM_SUBMIT_DIR:-$0}")
echo "Running from directory: $VLAD_DIR"

# Physiology
echo "Making physiology figures"
cd $VLAD_DIR/physiology
python plot_physiology.py
python plot_viral_quantification.py

# Single cell
echo "Making single cell figures"
cd $VLAD_DIR/singlecell
python plot_singlecell.py

#Low D (PCA) analyses
echo "Making PCA figures"
cd $VLAD_DIR/low_d
python plot_pca_examples.py
python plot_pca_perturbations.py
python plot_pca_speed_example.py
python plot_videos.py

#Hering Breuer
echo "Making Hering-Breuer figures"
cd $VLAD_DIR/hering_breuer
python plot_hering_breuer.py

# RSLDS Model (only summary figs, no examples at this point)
echo "Making RSLDS model figures"
cd $VLAD_DIR/model
python plot_summary_nostim_figures.py
python plot_stim_field_alignments.py

cd $VLAD_DIR
echo "Pushing to RSS"
sh push_all_to_rss.sh
echo "All done!"
