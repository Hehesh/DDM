#!/bin/bash
#SBATCH --job-name=create_all_trials_csv_job.sh
#SBATCH --output=logs/create_all_trials_csv_%j.out
#SBATCH --error=logs/create_all_trials_csv_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --partition=main
#SBATCH --mail-user=bchien37@cmc.edu
#SBATCH --mail-type=BEGIN,END,FAIL

source /hopper/home/bchien37/miniconda3/etc/profile.d/conda.sh
conda activate DDM

python fixations.py \
  '/hopper/groups/enkavilab/users/bchien/eum2023_data_code/data/joint' \
  '.' \
  $1