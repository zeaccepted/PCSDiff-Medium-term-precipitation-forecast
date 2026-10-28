# PCSDiff: Diffusion-Based Bias Correction and Super Resolution Toward Practical Operational Medium-Term Precipitation Forecast

[Paper (arXiv)](https://arxiv.org/abs/2609.06942)

## Introduction

Medium-range precipitation forecasts are impaired by persistent systematic biases, lead-time-dependent error accumulation, and coarse spatial resolution, restricting their reliability for flood-drought risk assessment. Existing AI correction techniques lack dedicated modeling for multi-day dynamic bias evolution and proper meteorological constraints, often generating over-smoothed rainfall structures, and cannot meet operational deployment demands. This work introduces PCSDiff, a cascaded task-decoupled diffusion framework targeting 10-day precipitation bias correction and downscaling. To jointly counteract temporal error drifts and reconstruct physically plausible local precipitation details, PCSDiff integrates the Precipitation Intensity-aware Multi-branch Decoder (PIMD) module for dynamic multi-day error mitigation using synoptic-temporal features, followed by a two-phase conditional diffusion super-resolution module to restore fine-scale precipitation patterns. Evaluated against CMA-CRA observations over China after global-data training, PCSDiff cuts RMSE by 16.1% and lifts ACC by 13.9% relative to raw ECMWF forecasts at 3-10-day lead times, and consistently outperforms mainstream deep-learning baselines on both general and extreme-precipitation metrics. Benefiting from a streaming inference pipeline, our method achieves low-latency rolling forecasting for practical meteorological operations.

<img src="https://github.com/user-attachments/assets/260cdfa9-4a62-4b74-8812-bb4aa581d912" alt="Overview of the PCSDiff framework" width="100%" />

## Installation

The training scripts target Linux with Slurm and CUDA GPUs. Python 3.10 or newer is recommended. Install a PyTorch build compatible with your CUDA driver from the [official PyTorch instructions](https://pytorch.org/get-started/locally/), then install the remaining dependencies:

```bash
python -m pip install -r requirements.txt
```

Activate this Python environment before submitting jobs. The `submit_*.sh` files are Slurm wrappers. Select a partition for your cluster, for example `sbatch --partition=<partition> submit_train.sh`, and adjust their GPU and time requests as needed.

## Data preparation

Meteorological data are not distributed with this repository. Set `RVD_DATA_ROOT` to the parent directory containing the preprocessed NetCDF files:

```bash
export RVD_DATA_ROOT=/path/to/preprocessed_data
```

| Directory under `RVD_DATA_ROOT` | Expected contents |
| --- | --- |
| `ecmwf_tp_daily_noensmean/` | ECMWF S2S daily precipitation files matching `s2s_ecmf_YYYYMMDD_*_tp_daily.nc`; each forecast file must contain deterministic `tp` and ensemble variables `tp_2` through `tp_11`. |
| `CRA_truth_nc/` | Native-resolution CRA daily truth files named `CRA_truth_YYYYMMDD.nc`, containing `tp`. |
| `ecmwf_2t_nc/`, `ecmwf_msl_nc/`, `ecmwf_gh_nc/`, `ecmwf_q_nc/`, `ecmwf_sp_nc/`, `ecmwf_u_nc/`, `ecmwf_v_nc/`, `ecmwf_w_nc/` | Deterministic auxiliary forecasts for the multivariable experiment, named `s2s_ecmf_YYYYMMDD_<variable>.nc` and aligned with the precipitation forecast grid. |

The multivariable model uses 29 auxiliary channels: one each for `2t`, `msl`, and `sp`; `gh` at 850/700/500 hPa; `q` at 1000/925/850/700/500 hPa; and each of `u`, `v`, and `w` at 1000/925/850/700/500/300 hPa. `ttr` is not used. Coordinates may be named `lat`/`lon` or `latitude`/`longitude`. See `data/datasets/precipitation.py` and `data/datasets/multivar_precipitation.py` for the exact dimensions, units, and file-matching logic. Other data sources must be converted to these formats before use.

The temporal split is fixed: a training sample's last target date must be no later than **2021-12-31**, and test initialization dates start on **2022-01-01**. Training samples whose targets cross the boundary are excluded. Normalization statistics are computed from training data and reused for testing and inference.

## Training and evaluation

Run the commands from the repository root, waiting for each job to complete before submitting its dependent job. Generated checkpoints, NetCDF files, metrics, and figures are written under `results/`.

### Stage 1: precipitation-only baseline

```bash
sbatch submit_train.sh
sbatch submit_infer.sh
sbatch submit_plot_china.sh
```

Training records the checkpoint path in `results/latest_training_checkpoint.txt`. Inference uses the matching `normalization_params.npz` beside that checkpoint.

### Stage 1: multivariable correction

```bash
sbatch submit_train_multivar.sh
sbatch submit_infer_multivar.sh
sbatch submit_plot_multivar_compare.sh
```

This variant adds 29 meteorological channels to the 10 precipitation ensemble members. Its checkpoint must be kept together with both `normalization_params.npz` and `aux_normalization_params.npz`.

### Stage 2: super-resolution

```bash
sbatch submit_generate_sr_input.sh
sbatch submit_train_super_resolution.sh
sbatch submit_evaluate_super_resolution.sh
```

The first job produces the Stage 1 corrected inputs at `results/super_resolution/correction_input/{train,test}/global_corrected.nc`. Stage 2 uses the native-resolution CRA data in `RVD_DATA_ROOT/CRA_truth_nc/` as its target. Its model and evaluation outputs are saved under `results/super_resolution/`.

The default lead time is 10 days. Set `RVD_FORECAST_LEAD_TIME` to change it; `submit_infer.sh` also accepts region, checkpoint, output directory, and lead days as arguments. Model and training settings are in `config_server.py`. After generating results, additional figures and tables can be produced with `submit_plot_paper_main.sh`, `submit_plot_paper_selected.sh`, `submit_table_metrics.sh`, and `submit_multivar_table_metrics.sh`.

## Repository structure

- `data/datasets/`: sample matching, temporal split, normalization, and data loading.
- `modules/`: correction model, trainer, and super-resolution model.
- `train_*.py` and `infer_*.py`: training and inference entry points.
- `evaluate_super_resolution.py`, `plot_*.py`, and `summarize_pcsdiff_table.py`: evaluation and visualization.
- `config_server.py`: model, training, and output configuration.

Data, trained weights, generated results, and the manuscript PDF are excluded from the repository.

## License

The code in this repository is released under the [MIT License](LICENSE).

## Citation

If you use this work, please cite the [PCSDiff preprint](https://arxiv.org/abs/2609.06942):

```bibtex
@misc{sun2026pcsdiff,
  title={PCSDiff: Diffusion-Based Bias Correction and Super Resolution Toward Practical Operational Medium-Term Precipitation Forecast},
  author={Sun, Yuze and Wang, Shiyi and Pan, Jiancheng and Wang, Die and Prein, Andreas F. and Luo, Wentao and Jiang, Linhan and Wu, Jie and Zhang, Quan and Huang, Xiaomeng},
  year={2026},
  eprint={2609.06942},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2609.06942}
}
```

## Contact

For questions or suggestions, contact [Yuze Sun](mailto:syz23@mails.tsinghua.edu.cn) or open a GitHub issue.
