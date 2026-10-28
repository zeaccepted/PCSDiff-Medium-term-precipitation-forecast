#!/bin/bash
#SBATCH --job-name=RVD_infer
#SBATCH --output=slurm_infer_%j.out
#SBATCH --error=slurm_infer_%j.err
#SBATCH --gpus=2
#SBATCH --time=24:00:00

# 用已经训练好的checkpoint做推理；绝不训练。
# 默认：中国区域、最近一次训练checkpoint、10天预报、2张GPU。
# 可选高级用法：sbatch submit_infer.sh <china|global> <checkpoint.pt> <output_nc_dir> [lead_days]
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
LATEST_TRAINING_FILE="$RESULTS_ROOT/latest_training_checkpoint.txt"
LATEST_INFERENCE_FILE="$RESULTS_ROOT/latest_inference_china_nc.txt"
REGION="${1:-china}"
CHECKPOINT="${2:-}"
LEAD_DAYS="${4:-10}"
if [ -z "$CHECKPOINT" ]; then
    if [ ! -f "$LATEST_TRAINING_FILE" ]; then
        echo "ERROR: no registered trained checkpoint. Run: sbatch submit_train.sh"
        exit 2
    fi
    CHECKPOINT=$(tr -d '\r\n' < "$LATEST_TRAINING_FILE")
fi
if [ "$REGION" != "china" ] && [ "$REGION" != "global" ]; then
    echo "ERROR: region must be china or global"
    exit 2
fi
if [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: checkpoint not found: $CHECKPOINT"
    exit 2
fi

CHECKPOINT_DIR=$(dirname "$CHECKPOINT")
NORMALIZATION_PARAMS="$CHECKPOINT_DIR/normalization_params.npz"
if [ ! -f "$NORMALIZATION_PARAMS" ]; then
    echo "ERROR: matching normalization_params.npz not found: $NORMALIZATION_PARAMS"
    exit 2
fi

# checkpoint目录为 results/<run_number>/<model_name>/，由此复用原run编号，避免推理时新建训练目录。
RUN_DIR=$(dirname "$CHECKPOINT_DIR")
RUN_NAME=$(basename "$RUN_DIR")
if [[ "$RUN_NAME" =~ ^[0-9]+$ ]]; then
    export RVD_RUN_NUMBER="$RUN_NAME"
fi
OUTPUT_DIR="${3:-$RUN_DIR/nc_${REGION}}"
mkdir -p "$OUTPUT_DIR"

if [ "$REGION" = "china" ]; then
    INFER_SCRIPT="infer_global_correction.py"
else
    INFER_SCRIPT="infer_global_correction_global.py"
fi

echo "========== Inference only =========="
echo "Region:      $REGION"
echo "Checkpoint:  $CHECKPOINT"
echo "Norm params: $NORMALIZATION_PARAMS"
echo "Output dir:  $OUTPUT_DIR"
echo "Lead days:   $LEAD_DAYS"
RVD_MODEL_CHECKPOINT="$CHECKPOINT" \
RVD_NC_OUTPUT_DIR="$OUTPUT_DIR" \
RVD_FORECAST_LEAD_TIME="$LEAD_DAYS" \
python -u "$INFER_SCRIPT"

NC_FILE="$OUTPUT_DIR/global_corrected.nc"
if [ ! -f "$NC_FILE" ]; then
    echo "ERROR: inference ended but NC file was not created: $NC_FILE"
    exit 1
fi
if [ "$REGION" = "china" ]; then
    printf '%s\n' "$NC_FILE" > "$LATEST_INFERENCE_FILE"
    echo "Registered latest China NC: $LATEST_INFERENCE_FILE"
fi
echo "Inference completed: $NC_FILE"
echo "Next command: sbatch submit_plot_china.sh"
