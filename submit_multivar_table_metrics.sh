#!/bin/bash
#SBATCH --job-name=RVD_multivar_table
#SBATCH --output=slurm_multivar_table_%j.out
#SBATCH --error=slurm_multivar_table_%j.err
#SBATCH --gpus=1
#SBATCH --time=02:00:00

# GPU table extraction for the orange TP + all-meteorological-variables line.
# Before submission: activate a Python environment with requirements.txt installed.
# Choose your cluster partition with sbatch --partition=<partition> if needed.
set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
if [ ! -f "$PROJECT_ROOT/summarize_pcsdiff_table.py" ]; then
    echo "ERROR: run sbatch from the project root."
    exit 2
fi
cd "$PROJECT_ROOT"

COMMON_ARGS=(
    --pointer-file results/latest_multivar_china_nc.txt
    --method-label "TP + All Met (orange)"
    --output-name table_metrics_multivar_days_3_5_7_10.txt
    --latest-output-pointer results/latest_multivar_table_metrics.txt
)
if [ "$#" -gt 0 ]; then
    python -u summarize_pcsdiff_table.py "$1" "${COMMON_ARGS[@]}"
else
    python -u summarize_pcsdiff_table.py "${COMMON_ARGS[@]}"
fi
