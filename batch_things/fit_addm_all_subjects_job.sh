#!/bin/bash
#SBATCH --job-name=fit_addm_all_subjects
#SBATCH --output=logs/fit_addm_all_subjects_%j.out
#SBATCH --error=logs/fit_addm_all_subjects_%j.err
#SBATCH --time=6:00:00
#SBATCH --cpus-per-task=48
#SBATCH --mem=20G
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --partition=main
#SBATCH --mail-user=bchien37@cmc.edu
#SBATCH --mail-type=BEGIN,END,FAIL

# Activate conda environment
source /hopper/home/bchien37/miniconda3/etc/profile.d/conda.sh
conda activate DDM

# include CPU's
export OMP_NUM_THREADS=48

# Data settings
DATA_DIR="/Users/braydenchien/Desktop/Enkavilab/DDM/filteredsamples"
BIN_SIZE=0.1

# Ensure logs directory exists (in case)
mkdir -p logs

# Main loop
for file in "$DATA_DIR"/pyddm_sample*.csv; do
    filename=$(basename "$file")

    sub_id=$(echo "$filename" | grep -oP '(?<=pyddm_sample)\d+(?=_)')
    visibility=$(echo "$filename" | grep -oP '(VISIBLE|HIDDEN)')

    if [[ -n "$sub_id" && -n "$visibility" ]]; then
        echo "Running subject $sub_id ($visibility)"
        python run_fit_script.py "$DATA_DIR" "$sub_id" "$visibility" "$BIN_SIZE"
    else
        echo "Skipping $filename — could not extract sub_id or visibility"
    fi
done
