#!/bin/bash
#SBATCH --job-name=RVD_plot
#SBATCH --output=slurm_plot_%j.out
#SBATCH --error=slurm_plot_%j.err
#SBATCH --gpus=2
#SBATCH --time=02:00:00

# 只读取已经存在的中国区域NC文件并重新绘图；不读取checkpoint，也不训练。
# 默认：最近一次中国区域推理产生的NC、同一实验目录下的plots_china、6个起报日期、2张GPU。
# 可选高级用法：sbatch submit_plot_china.sh <global_corrected.nc> <output_plot_dir> [max_dates]
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
LATEST_INFERENCE_FILE="$RESULTS_ROOT/latest_inference_china_nc.txt"
NC_FILE="${1:-}"
if [ -z "$NC_FILE" ]; then
    if [ ! -f "$LATEST_INFERENCE_FILE" ]; then
        echo "ERROR: no registered China NC. Run: sbatch submit_infer.sh"
        exit 2
    fi
    NC_FILE=$(tr -d '\r\n' < "$LATEST_INFERENCE_FILE")
fi
PLOT_DIR="${2:-$(dirname "$NC_FILE")/../plots_china}"
MAX_DATES="${3:-6}"
if [ ! -f "$NC_FILE" ]; then
    echo "ERROR: NC file not found: $NC_FILE"
    exit 2
fi

mkdir -p "$PLOT_DIR"
echo "========== Plot only =========="
echo "NC file:    $NC_FILE"
echo "Output dir: $PLOT_DIR"
python plot_metrics_by_lead_time.py "$NC_FILE" --output_dir "$PLOT_DIR" --china_region
python plot_results_main.py "$NC_FILE" --max_dates "$MAX_DATES" --output_dir "$PLOT_DIR" --region_only china
python plot_acc_map.py "$NC_FILE" --output_dir "$PLOT_DIR"
echo "Plots completed: $PLOT_DIR"
