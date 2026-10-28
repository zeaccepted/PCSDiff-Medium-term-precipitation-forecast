"""Generate the exact orange-line low-resolution inputs for both SR splits."""
from __future__ import annotations

import os
import subprocess
import sys


def main():
    project_root = os.path.abspath(os.environ.get("RVD_PROJECT_ROOT", os.path.dirname(__file__)))
    pointer = os.path.join(project_root, "results", "latest_multivar_checkpoint.txt")
    if not os.path.isfile(pointer):
        raise FileNotFoundError("Missing results/latest_multivar_checkpoint.txt; train the multivariable model first")
    with open(pointer, encoding="utf-8") as handle:
        checkpoint = handle.read().strip()
    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(f"Multivariable checkpoint no longer exists: {checkpoint}")

    output_root = os.path.join(project_root, "results", "super_resolution", "correction_input")
    common = os.environ.copy()
    checkpoint_run = os.path.basename(os.path.dirname(os.path.dirname(checkpoint)))
    common.update({"RVD_PROJECT_ROOT": project_root, "RVD_MODEL_CHECKPOINT": checkpoint,
                   "RVD_RUN_NUMBER": checkpoint_run,
                   "RVD_FORECAST_LEAD_TIME": "10", "RVD_SAVE_ENSEMBLE_MEMBERS": "0"})
    for split in ("train", "test"):
        output_dir = os.path.join(output_root, split)
        os.makedirs(output_dir, exist_ok=True)
        env = common.copy()
        env.update({"RVD_INFERENCE_SPLIT": split, "RVD_NC_OUTPUT_DIR": output_dir})
        print(f"[SR input] generating {split}: {output_dir}", flush=True)
        subprocess.run([sys.executable, "-u", os.path.join(project_root, "infer_multivar.py")],
                       cwd=project_root, env=env, check=True)
        expected = os.path.join(output_dir, "global_corrected.nc")
        if not os.path.isfile(expected):
            raise RuntimeError(f"Inference finished without creating {expected}")
    print(f"[SR input] complete: {output_root}")


if __name__ == "__main__":
    main()
