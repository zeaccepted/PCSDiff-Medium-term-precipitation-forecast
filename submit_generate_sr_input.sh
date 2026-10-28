#!/bin/bash
#SBATCH --job-name=RVD_sr_input
#SBATCH --output=slurm_sr_input_%j.out
#SBATCH --error=slurm_sr_input_%j.err
#SBATCH --gpus=2
#SBATCH --time=24:00:00

# Before submission: activate a Python environment with requirements.txt installed.
# Choose your cluster partition with sbatch --partition=<partition> if needed.
set -euo pipefail
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
test -f "$PROJECT_ROOT/config_server.py" || { echo "Run this script from the project root"; exit 2; }
export RVD_PROJECT_ROOT="$PROJECT_ROOT"
cd "$PROJECT_ROOT"
python -u generate_correction_dataset.py
