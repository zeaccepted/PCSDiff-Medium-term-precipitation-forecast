#!/bin/bash
#SBATCH --job-name=RVD_table_metrics
#SBATCH --output=slurm_table_metrics_%j.out
#SBATCH --error=slurm_table_metrics_%j.err
#SBATCH --gpus=1
#SBATCH --time=02:00:00

# Read an existing PCSDiff China inference result and write only the table TXT.
# Before submission: activate a Python environment with requirements.txt installed.
# Choose your cluster partition with sbatch --partition=<partition> if needed.
set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
if [ ! -f "$PROJECT_ROOT/summarize_pcsdiff_table.py" ]; then
    echo "ERROR: run sbatch from the project root."
    exit 2
fi
cd "$PROJECT_ROOT"
if [ "$#" -gt 0 ]; then
    python -u summarize_pcsdiff_table.py "$1"
else
    python -u summarize_pcsdiff_table.py
fi
