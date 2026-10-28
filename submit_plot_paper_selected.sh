#!/bin/bash
#SBATCH --job-name=RVD_selected_figs
#SBATCH --output=slurm_selected_figs_%j.out
#SBATCH --error=slurm_selected_figs_%j.err
#SBATCH --gpus=1
#SBATCH --time=01:00:00

# Before submission: activate a Python environment with requirements.txt installed.
# Choose your cluster partition with sbatch --partition=<partition> if needed.
set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
if [ ! -f "$PROJECT_ROOT/plot_paper_selected_cases.py" ]; then
    echo "ERROR: run 'sbatch submit_plot_paper_selected.sh' from the project root."
    exit 2
fi
export RVD_PROJECT_ROOT="$PROJECT_ROOT"
export HDF5_USE_FILE_LOCKING=FALSE
cd "$PROJECT_ROOT"

python -u plot_paper_selected_cases.py \
    --stage1_nc "$PROJECT_ROOT/results/super_resolution/correction_input/test/global_corrected.nc" \
    --sr_nc "$PROJECT_ROOT/results/super_resolution/evaluation/super_resolution_test_v3.nc" \
    --output_dir "$PROJECT_ROOT/results/paper_selected_cases"
echo "Selected paper figures: $PROJECT_ROOT/results/paper_selected_cases"
