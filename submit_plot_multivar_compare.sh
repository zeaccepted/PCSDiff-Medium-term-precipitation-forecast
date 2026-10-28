#!/bin/bash
#SBATCH --job-name=RVD_multivar_plot
#SBATCH --output=slurm_multivar_plot_%j.out
#SBATCH --error=slurm_multivar_plot_%j.err
#SBATCH --gpus=1
#SBATCH --time=02:00:00

# Before submission: activate a Python environment with requirements.txt installed.
# Choose your cluster partition with sbatch --partition=<partition> if needed.
set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
if [ ! -f "$PROJECT_ROOT/config_server.py" ]; then
    echo "ERROR: run 'sbatch submit_plot_multivar_compare.sh' from the project root."
    exit 2
fi
export RVD_PROJECT_ROOT="$PROJECT_ROOT"
cd "$PROJECT_ROOT"
RESULTS_ROOT="$PROJECT_ROOT/results"
BLUE_NC=$(tr -d '\r\n' < "$RESULTS_ROOT/latest_inference_china_nc.txt")
ORANGE_NC=$(tr -d '\r\n' < "$RESULTS_ROOT/latest_multivar_china_nc.txt")
if [ ! -f "$BLUE_NC" ] || [ ! -f "$ORANGE_NC" ]; then
    echo "ERROR: both baseline and multivariable China NC files are required"
    exit 2
fi
OUT="$(dirname "$ORANGE_NC")/../plots_multivar_compare"
# 默认从 7 月、8 月各均匀选择 8 个起报日期，方便查看降水活跃季的空间效果。
# 传入 --metrics-only 时只更新指标图，不重新生成空间案例图。
METRICS_ONLY=false
DATES_PER_MONTH="${1:-8}"
if [ "${1:-}" = "--metrics-only" ]; then
    METRICS_ONLY=true
fi
mkdir -p "$OUT"
python plot_metrics_by_lead_time.py "$BLUE_NC" --reference_nc "$ORANGE_NC" --output_dir "$OUT" --china_region
echo "Three-line metrics: $OUT/metrics_vs_leadtime_china.png"
echo "Paper-ready PDF: $OUT/metrics_vs_leadtime_china.pdf"
if [ "$METRICS_ONLY" = false ]; then
    python plot_results_main.py "$BLUE_NC" --reference_nc "$ORANGE_NC" --months 7 8 --dates_per_month "$DATES_PER_MONTH" --output_dir "$OUT" --region_only china
    echo "Four-column maps: $OUT/precip_comparison_all_leads_*_china.png"
fi
