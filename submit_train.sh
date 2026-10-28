#!/bin/bash
#SBATCH --job-name=RVD_train
#SBATCH --output=slurm_train_%j.out
#SBATCH --error=slurm_train_%j.err
#SBATCH --gpus=2
#SBATCH --time=72:00:00

# 仅训练并保存checkpoint；不执行推理或绘图。
# Before submission: activate a Python environment with requirements.txt installed.
# Choose your cluster partition with sbatch --partition=<partition> if needed.
set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
if [ ! -f "$PROJECT_ROOT/config_server.py" ]; then
    echo "ERROR: run sbatch from the project root; config_server.py not found in: $PROJECT_ROOT"
    exit 2
fi
export RVD_PROJECT_ROOT="$PROJECT_ROOT"
cd "$PROJECT_ROOT"
RESULTS_ROOT="$PROJECT_ROOT/results"

if [ -z "${RVD_RUN_NUMBER:-}" ]; then
    export RVD_RUN_NUMBER=$(python -c "import config_server as c; print(c.run_number)" 2>/dev/null | grep -oE '^[0-9]+$' | head -1)
fi
if [ -z "${RVD_RUN_NUMBER:-}" ]; then
    echo "ERROR: failed to allocate RVD_RUN_NUMBER"
    exit 1
fi

export RVD_USE_TP_LOSS=1
export RVD_USE_FSS_LOSS=0
RESULT_RUN_DIR="$RESULTS_ROOT/${RVD_RUN_NUMBER}"
LATEST_TRAINING_FILE="$RESULTS_ROOT/latest_training_checkpoint.txt"

echo "========== Training only =========="
echo "RVD_RUN_NUMBER=${RVD_RUN_NUMBER}"
python -u train_precipitation.py --ndevice 2

echo "========== Training artifacts =========="
FINAL_CHECKPOINT=$(find "$RESULT_RUN_DIR" -name '*.pt' -print | sort | tail -n 1)
if [ -z "$FINAL_CHECKPOINT" ]; then
    echo "ERROR: training ended but no checkpoint was found"
    exit 1
fi
if [ ! -f "$(dirname "$FINAL_CHECKPOINT")/normalization_params.npz" ]; then
    echo "ERROR: matching normalization_params.npz was not found"
    exit 1
fi
printf '%s\n' "$FINAL_CHECKPOINT" > "$LATEST_TRAINING_FILE"
echo "Final checkpoint: $FINAL_CHECKPOINT"
echo "Registered latest checkpoint: $LATEST_TRAINING_FILE"
echo "Next command: sbatch submit_infer.sh"
