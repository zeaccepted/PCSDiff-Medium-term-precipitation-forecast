"""独立推理 TP(10)+29 个气象辅助变量模型，复用主工程 NC 输出口径。"""
import os

import numpy as np
import torch

import config_server as config
import infer_global_correction as baseline_inference
from data.load_dataset_multivar import load_multivar_dataset
from modules.deterministic_correction import DeterministicCorrection

N_INPUT_CHANNELS = 39


def _checkpoint_path():
    path = os.environ.get("RVD_MODEL_CHECKPOINT", "")
    if not path or not os.path.isfile(path):
        raise FileNotFoundError("Set RVD_MODEL_CHECKPOINT to the multivariable .pt checkpoint")
    return path


def load_multivar_model():
    checkpoint_path = _checkpoint_path()
    model = DeterministicCorrection(
        simple_unet_dim=config.embed_dim, n_ensemble_channels=N_INPUT_CHANNELS,
        n_residual_channels=config.data_config["output_channel"], loss_type=config.loss_type,
        huber_delta=config.huber_delta, use_ensemble_spread=config.use_ensemble_spread,
        pred_mode=config.pred_mode, clip_noise=config.clip_noise, aux_loss=config.aux_loss,
        residual_blend_weight=config.residual_blend_weight, max_abs_residual_norm=config.max_abs_residual_norm,
        positive_residual_clip_value=config.positive_residual_clip_value,
        negative_residual_clip_value=config.negative_residual_clip_value,
        use_pimd=config.use_pimd, pimd_num_branches=config.pimd_num_branches,
        pimd_temperature=config.pimd_temperature, pimd_entropy_loss_weight=config.pimd_entropy_loss_weight,
    ).to(baseline_inference.DEVICE)
    checkpoint = torch.load(checkpoint_path, map_location=baseline_inference.DEVICE)
    state = checkpoint.get("ema", checkpoint.get("model", {}))
    state = {k[7:] if k.startswith("module.") else k: v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    model.eval()
    print(f"[多变量模型] checkpoint loaded; input channels={N_INPUT_CHANNELS}: {checkpoint_path}")
    return model


def load_multivar_data(data_config):
    stats_path = os.path.join(os.path.dirname(_checkpoint_path()), "aux_normalization_params.npz")
    if not os.path.isfile(stats_path):
        raise FileNotFoundError(f"Missing training-only auxiliary normalization file: {stats_path}")
    with np.load(stats_path) as data:
        stats = {key: data[key] for key in ("names", "min", "max")}
    return load_multivar_dataset(data_config, aux_normalization=stats)


if __name__ == "__main__":
    # 主推理脚本负责残差解码、NC 写入及指标所需字段；这里只替换模型与数据集工厂。
    baseline_inference.load_model = load_multivar_model
    baseline_inference.load_dataset = load_multivar_data
    baseline_inference.main()
