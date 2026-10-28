#!/bin/bash
#SBATCH --job-name=RVD_sr_eval
#SBATCH --output=slurm_sr_eval_%j.out
#SBATCH --error=slurm_sr_eval_%j.err
#SBATCH --gpus=1
#SBATCH --time=72:00:00

# Before submission: activate a Python environment with requirements.txt installed.
# Choose your cluster partition with sbatch --partition=<partition> if needed.
set -euo pipefail
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
export RVD_PROJECT_ROOT="$PROJECT_ROOT"
cd "$PROJECT_ROOT"
POINTER="$PROJECT_ROOT/results/latest_sr_checkpoint.txt"
test -f "$POINTER" || { echo "First run submit_train_super_resolution.sh"; exit 2; }
export RVD_SR_CHECKPOINT
RVD_SR_CHECKPOINT=$(tr -d '\r\n' < "$POINTER")
test -f "$RVD_SR_CHECKPOINT" || { echo "Missing checkpoint: $RVD_SR_CHECKPOINT"; exit 2; }
python -u evaluate_super_resolution.py
