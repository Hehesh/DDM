#!/bin/bash
#SBATCH --job-name=fit_addm_batch
#SBATCH --output=logs/fit_addm_batch_%j.out
#SBATCH --error=logs/fit_addm_batch_%j.err
#SBATCH --time=6:00:00
#SBATCH --cpus-per-task=48
#SBATCH --mem=20G
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --partition=main
#SBATCH --mail-user=bchien37@cmc.edu
#SBATCH --mail-type=BEGIN,END,FAIL

# Load Conda
source /hopper/home/bchien37/miniconda3/etc/profile.d/conda.sh
conda activate DDM

# Limit NumPy thread usage
export OMP_NUM_THREADS=48

# Run the model fitting script
python fit_addm_batch.py \
  "/hopper/groups/enkavilab/users/bchien/filteredsamples" \
  1 \
  0.1

