#!/bin/bash
#SBATCH --job-name=RVD_multivar_infer
#SBATCH --output=slurm_multivar_infer_%j.out
#SBATCH --error=slurm_multivar_infer_%j.err
#SBATCH --gpus=2
#SBATCH --time=24:00:00

# Before submission: activate a Python environment with requirements.txt installed.
# Choose your cluster partition with sbatch --partition=<partition> if needed.
set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
if [ ! -f "$PROJECT_ROOT/config_server.py" ]; then
    echo "ERROR: run 'sbatch submit_infer_multivar.sh' from the project root."
    exit 2
fi
export RVD_PROJECT_ROOT="$PROJECT_ROOT"
cd "$PROJECT_ROOT"
RESULTS_ROOT="$PROJECT_ROOT/results"
POINTER="$RESULTS_ROOT/latest_multivar_checkpoint.txt"
if [ ! -f "$POINTER" ]; then
    echo "ERROR: no multivariable checkpoint. First run submit_train_multivar.sh."
    exit 2
fi
CKPT=$(tr -d '\r\n' < "$POINTER")
if [ ! -f "$CKPT" ]; then
    echo "ERROR: checkpoint not found: $CKPT"
    exit 2
fi
RUN_DIR=$(dirname "$(dirname "$CKPT")")
export RVD_RUN_NUMBER=$(basename "$RUN_DIR")
OUT="$RUN_DIR/nc_china_multivar"
mkdir -p "$OUT"

RVD_MODEL_CHECKPOINT="$CKPT" \
RVD_NC_OUTPUT_DIR="$OUT" \
RVD_FORECAST_LEAD_TIME=10 \
python -u infer_multivar.py

NC_FILE="$OUT/global_corrected.nc"
if [ ! -f "$NC_FILE" ]; then
    echo "ERROR: multivariable inference did not create: $NC_FILE"
    exit 1
fi
printf '%s\n' "$NC_FILE" > "$RESULTS_ROOT/latest_multivar_china_nc.txt"
echo "Registered multivariable China NC: $NC_FILE"
