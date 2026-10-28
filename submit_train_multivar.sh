#!/bin/bash
#SBATCH --job-name=RVD_multivar_train
#SBATCH --output=slurm_multivar_train_%j.out
#SBATCH --error=slurm_multivar_train_%j.err
#SBATCH --gpus=2
#SBATCH --time=72:00:00

# Before submission: activate a Python environment with requirements.txt installed.
# Choose your cluster partition with sbatch --partition=<partition> if needed.
set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
if [ ! -f "$PROJECT_ROOT/config_server.py" ]; then
    echo "ERROR: run 'sbatch submit_train_multivar.sh' from the project root."
    exit 2
fi
export RVD_PROJECT_ROOT="$PROJECT_ROOT"
cd "$PROJECT_ROOT"
RESULTS_ROOT="$PROJECT_ROOT/results"

# 该实验必须独占一个新 run，不能复用主模型目录。
export RVD_RUN_NUMBER=$(python -c "import os, sys; root=sys.argv[1]; os.makedirs(root, exist_ok=True); nums=[int(x) for x in os.listdir(root) if x.isdigit() and os.path.isdir(os.path.join(root, x))]; print(max(nums, default=0) + 1)" "$RESULTS_ROOT")
RUN_DIR="$RESULTS_ROOT/$RVD_RUN_NUMBER"
POINTER="$RESULTS_ROOT/latest_multivar_checkpoint.txt"
export RVD_USE_TP_LOSS=1
export RVD_USE_FSS_LOSS=0

echo "========== TP + 29 meteorological channels training =========="
echo "Run: $RUN_DIR"
python -u train_multivar.py --ndevice 2

CKPT=$(find "$RUN_DIR" -name '*.pt' -print | sort | tail -n 1)
if [ -z "$CKPT" ] || [ ! -f "$CKPT" ]; then
    echo "ERROR: multivariable training ended but no checkpoint was found"
    exit 1
fi
if [ ! -f "$(dirname "$CKPT")/aux_normalization_params.npz" ]; then
    echo "ERROR: auxiliary normalization file missing beside checkpoint"
    exit 1
fi
printf '%s\n' "$CKPT" > "$POINTER"
echo "Registered multivariable checkpoint: $CKPT"
