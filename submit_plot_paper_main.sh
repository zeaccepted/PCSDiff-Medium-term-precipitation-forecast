#!/bin/bash
#SBATCH --job-name=RVD_paper_figs
#SBATCH --output=slurm_paper_figs_%j.out
#SBATCH --error=slurm_paper_figs_%j.err
#SBATCH --gpus=1
#SBATCH --time=02:00:00

# Before submission: activate a Python environment with requirements.txt installed.
# Choose your cluster partition with sbatch --partition=<partition> if needed.
set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
if [ ! -f "$PROJECT_ROOT/plot_paper_main_figures.py" ]; then
    echo "ERROR: run 'sbatch submit_plot_paper_main.sh' from the project root."
    exit 2
fi
export RVD_PROJECT_ROOT="$PROJECT_ROOT"
export HDF5_USE_FILE_LOCKING=FALSE
cd "$PROJECT_ROOT"

STAGE1_NC="$PROJECT_ROOT/results/super_resolution/correction_input/test/global_corrected.nc"
SR_NC="$PROJECT_ROOT/results/super_resolution/evaluation/super_resolution_test_v3.nc"
OUTPUT_DIR="$PROJECT_ROOT/results/paper_main_figures"
test -f "$STAGE1_NC" || { echo "ERROR: missing Stage-1 result: $STAGE1_NC"; exit 2; }
# Python validates the fixed v3 file and can fall back to a readable _job*.nc.

python -u plot_paper_main_figures.py \
    --stage1_nc "$STAGE1_NC" \
    --sr_nc "$SR_NC" \
    --output_dir "$OUTPUT_DIR" \
    --dates_per_month 4
echo "Paper main figures: $OUTPUT_DIR"
