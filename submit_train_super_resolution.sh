#!/bin/bash
#SBATCH --job-name=RVD_sr_train
#SBATCH --output=slurm_sr_train_%j.out
#SBATCH --error=slurm_sr_train_%j.err
#SBATCH --gpus=1
#SBATCH --time=72:00:00

# Before submission: activate a Python environment with requirements.txt installed.
# Choose your cluster partition with sbatch --partition=<partition> if needed.
set -euo pipefail
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
export RVD_PROJECT_ROOT="$PROJECT_ROOT"
cd "$PROJECT_ROOT"
TRAIN_NC="$PROJECT_ROOT/results/super_resolution/correction_input/train/global_corrected.nc"
TEST_NC="$PROJECT_ROOT/results/super_resolution/correction_input/test/global_corrected.nc"
test -f "$TRAIN_NC" && test -f "$TEST_NC" || { echo "First run submit_generate_sr_input.sh"; exit 2; }
python -u train_super_resolution.py
CKPT="$PROJECT_ROOT/results/super_resolution/best_model_v3.pt"
test -f "$CKPT" || { echo "SR checkpoint was not created"; exit 1; }
printf '%s\n' "$CKPT" > "$PROJECT_ROOT/results/latest_sr_checkpoint.txt"
