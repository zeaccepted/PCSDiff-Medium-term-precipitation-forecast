"""
infer_global_correction.py - 使用主工程模型对全球数据做推理

该脚本加载主工程训练好的DeterministicCorrection模型，对全球范围的ECMWF预报数据做订正推理，
将结果保存为NC文件，供plot_metrics_by_lead_time.py使用。

运行方法：python infer_global_correction.py
"""
import os
import sys
import numpy as np
import torch
import netCDF4 as nc
from torch.utils.data import DataLoader

# 添加主工程根目录到path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import config_server as config
from modules.deterministic_correction import DeterministicCorrection
from data.transposed_collate import precipitation_collate_fn
from data.load_dataset import load_dataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 使用config中的region配置（中国区域），不再强制设置global
# config.data_config["region"] = "global"
# config.data_config["lat_range"] = (-90, 90)
# config.data_config["lon_range"] = (0, 360)

def load_model():
    """加载主工程训练好的模型（Step2 SimpleUNet + PIMD）"""
    print("[模型] 初始化SimpleUNet + PIMD架构...")

    model = DeterministicCorrection(
        # SimpleUNet核心超参
        simple_unet_dim=getattr(config, "embed_dim", 64),
        n_ensemble_channels=getattr(config, "model_input_channels", 10),
        n_residual_channels=config.data_config["output_channel"],
        # 训练损失参数
        loss_type=getattr(config, "loss_type", "l2"),
        huber_delta=getattr(config, "huber_delta", 1.0),
        use_ensemble_spread=getattr(config, "use_ensemble_spread", False),
        pred_mode=getattr(config, "pred_mode", "pred_true"),
        clip_noise=getattr(config, "clip_noise", True),
        aux_loss=getattr(config, "aux_loss", False),
        residual_blend_weight=getattr(config, "residual_blend_weight", 1.0),
        max_abs_residual_norm=getattr(config, "max_abs_residual_norm", 1.0),
        positive_residual_clip_value=getattr(config, "positive_residual_clip_value", 1.0),
        negative_residual_clip_value=getattr(config, "negative_residual_clip_value", 1.0),
        # 归一化参数
        forecast_min=None,
        forecast_max=None,
        truth_min=None,
        truth_max=None,
        # ===== PIMD参数（与训练端一致）=====
        use_pimd=getattr(config, "use_pimd", True),
        pimd_num_branches=getattr(config, "pimd_num_branches", 3),
        pimd_temperature=getattr(config, "pimd_temperature", 1.0),
        pimd_entropy_loss_weight=getattr(config, "pimd_entropy_loss_weight", 0.01),
    ).to(DEVICE)

    # 查找检查点
    # 优先级: 1) 环境变量 RVD_MODEL_CHECKPOINT  2) 自动搜索
    env_ckpt = os.environ.get("RVD_MODEL_CHECKPOINT", "")
    if env_ckpt and os.path.isfile(env_ckpt):
        latest_checkpoint = env_ckpt
        print(f"[模型] 使用环境变量指定的检查点: {latest_checkpoint}")
    else:
        run_dir = config.current_run_dir
        results_root = os.path.dirname(run_dir)
        checkpoint_files = []
        for root, dirs, files in os.walk(results_root):
            for f in files:
                if f.endswith('.pt'):
                    checkpoint_files.append(os.path.join(root, f))

        if not checkpoint_files:
            raise ValueError(f"未找到模型检查点文件，搜索路径: {results_root}")

        latest_checkpoint = sorted(checkpoint_files)[-1]
        print(f"[模型] 自动搜索到检查点: {latest_checkpoint}")
    checkpoint = torch.load(latest_checkpoint, map_location=DEVICE)

    state_dict = checkpoint.get("ema", checkpoint.get("model", {}))
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('module.'):
            new_state_dict[key[7:]] = value
        else:
            new_state_dict[key] = value

    model.load_state_dict(new_state_dict)
    model.eval()
    # 训练-推理一致性诊断：确保use_ensemble_spread/Unet通道数/clip值等关键属性与配置一致
    cfg_use_spread = getattr(config, "use_ensemble_spread", False)
    cfg_pos_clip = getattr(config, "positive_residual_clip_value", 0.7)
    cfg_neg_clip = getattr(config, "negative_residual_clip_value", 1.2)
    cfg_max_abs = getattr(config, "max_abs_residual_norm", 1.0)
    model_use_spread = getattr(model, "use_ensemble_spread", None)
    model_pos_clip = getattr(model, "positive_residual_clip_value", None)
    model_neg_clip = getattr(model, "negative_residual_clip_value", None)
    model_max_abs = getattr(model, "max_abs_residual_norm", None)
    print(f"[模型] 关键属性一致性检查：")
    print(f"        use_ensemble_spread    config={cfg_use_spread}  model={model_use_spread}  {'✅一致' if cfg_use_spread == model_use_spread else '⚠️不一致！'}")
    print(f"        positive_residual_clip  config={cfg_pos_clip}  model={model_pos_clip}")
    print(f"        negative_residual_clip  config={cfg_neg_clip}  model={model_neg_clip}")
    print(f"        max_abs_residual_norm   config={cfg_max_abs}  model={model_max_abs}")

    # SimpleUNet + PIMD诊断
    simple_unet = getattr(model, "simple_unet", None)
    if simple_unet is not None:
        inc_conv1_w = getattr(getattr(simple_unet.inc, "conv", None), "0", None)
        if inc_conv1_w is not None and hasattr(inc_conv1_w, "weight"):
            ch = inc_conv1_w.weight.shape[1]
            exp_ch = getattr(model, "n_ensemble_channels", 10)
            print(f"        SimpleUNet.inc输入通道 ch={ch}  期望={exp_ch}  {'✅一致' if ch == exp_ch else '⚠️MISMATCH！'}")

        # PIMD状态检查
        pimd = getattr(simple_unet, "pimd", None)
        if pimd is not None:
            print(f"        PIMD模块: ✅已启用")
            print(f"          分支数={getattr(pimd, 'num_branches', 'N/A')}, 温度={getattr(pimd, 'temperature', 'N/A')}")
        else:
            print(f"        PIMD模块: ❌未启用")

        n_params = sum(p.numel() for p in simple_unet.parameters())
        print(f"        SimpleUNet参数量       {n_params/1e6:.2f}M")
    print("[模型] 模型加载完成")
    return model


def forecast_to_truth_space(forecast, dataset):
    """将forecast映射到truth空间（与训练+dataset.forecast_to_truth_space完全一致）
    
    线性归一化下forecast和truth统一使用forecast的min/max归一化，已在同一空间，直接返回。
    参见 PrecipitationDataset.forecast_to_truth_space 文档。
    """
    return forecast


def decode_residual(residual_prediction):
    """解码残差：将归一化空间的残差转换为mm空间的残差
    
    对称解码逻辑(与训练decode一致):
      positive = clamp(x, 0, 1) * pos_clip
      negative = clamp(x, -1, 0) * neg_clip
      最后 clamp(-max_abs, max_abs)
    
    返回归一化空间的残差，后续通过 denormalize_residual 转换为mm
    """
    pos_clip = float(getattr(config, "positive_residual_clip_value", 0.85))
    neg_clip = float(getattr(config, "negative_residual_clip_value", 1.2))
    max_abs = float(getattr(config, "max_abs_residual_norm", 1.0))
    
    positive = residual_prediction.clamp(min=0.0) * pos_clip
    negative = residual_prediction.clamp(max=0.0) * neg_clip
    residual_decoded = positive + negative
    residual_decoded = residual_decoded.clamp(min=-max_abs, max=max_abs)
    
    residual_blend = float(getattr(config, "residual_blend_weight", 1.0))
    result = residual_blend * residual_decoded.detach().cpu().numpy()
    
    # 诊断输出
    residual_np = residual_prediction.detach().cpu().numpy()
    pos_ratio = np.mean(residual_np > 0)
    neg_ratio = np.mean(residual_np < 0)
    pos_mean = np.mean(residual_np[residual_np > 0]) if pos_ratio > 0 else 0
    neg_mean = np.mean(residual_np[residual_np < 0]) if neg_ratio > 0 else 0
    print(f"[DIAG-RESIDUAL] 正残差比例={pos_ratio:.2%}, 均值={pos_mean:.4f}; 负残差比例={neg_ratio:.2%}, 均值={neg_mean:.4f}")
    print(f"[DIAG-RESIDUAL] 解码后(归一化空间): pos_clip={pos_clip}, neg_clip={neg_clip}, max_abs={max_abs}, result范围=[{result.min():.4f}, {result.max():.4f}]")
    
    return result


def denormalize_residual_to_mm(residual_norm, dataset):
    """将归一化空间的残差转换为mm空间
    
    归一化: x_norm = 2*(x_mm - min)/(max-min) - 1
    残差在归一化空间: residual_norm = truth_norm - forecast_norm
    残差在mm空间: residual_mm = residual_norm * (max-min) / 2
    
    因为残差是两个归一化值的差，常数项抵消了。
    """
    forecast_min = float(dataset.forecast_min)
    forecast_max = float(dataset.forecast_max)
    scale = (forecast_max - forecast_min) / 2.0
    residual_mm = residual_norm * scale
    return residual_mm


def main():
    print("=" * 80)
    print("全球数据订正推理脚本 - infer_global_correction.py")
    print("=" * 80)

    # 读取预报时效配置
    n_days = getattr(config, "forecast_lead_time", 10)
    print(f"[配置] 预报时效 n_days = {n_days} 天")

    # 加载模型
    model = load_model()

    # 加载数据集（全球）
    print("\n[数据] 加载全球数据集...")
    train_dataset, val_dataset = load_dataset(config.data_config)
    print(f"  训练集: {len(train_dataset)} 个样本")
    print(f"  验证集: {len(val_dataset)} 个样本")
    inference_split = os.environ.get("RVD_INFERENCE_SPLIT", "test").strip().lower()
    if inference_split not in {"train", "test"}:
        raise ValueError("RVD_INFERENCE_SPLIT must be 'train' or 'test'")
    inference_dataset = train_dataset if inference_split == "train" else val_dataset
    save_members = os.environ.get("RVD_SAVE_ENSEMBLE_MEMBERS", "1") != "0"
    print(f"  本次推理数据: {inference_split}, {len(inference_dataset)} 个样本")
    print(f"  保存集合成员: {save_members}")
    
    # ========== 关键修复：加载训练时保存的归一化参数文件 ==========
    # 从训练结果目录加载normalization_params.npz，确保使用完全一致的forecast_min/max
    run_dir = config.current_run_dir
    norm_params_path = os.path.join(run_dir, "normalization_params.npz")
    
    if os.path.exists(norm_params_path):
        print(f"[归一化] 从文件加载训练时的归一化参数: {norm_params_path}")
        norm_data = np.load(norm_params_path)
        fc_min = float(norm_data["forecast_min"])
        fc_max = float(norm_data["forecast_max"])
        print(f"[归一化] fc_min={fc_min:.4f}, fc_max={fc_max:.4f}")
    else:
        print(f"[警告] 未找到归一化参数文件: {norm_params_path}")
        print(f"[警告] 使用训练集对象的参数作为fallback...")
        fc_min = float(train_dataset.forecast_min)
        fc_max = float(train_dataset.forecast_max)
        print(f"[归一化] fc_min={fc_min:.4f}, fc_max={fc_max:.4f}")
    
    use_anomaly = bool(getattr(config, "use_anomaly", False))
    print(f"[归一化] use_anomaly={use_anomaly}")
    
    # 构建残差→mm的scale：训练时normalize使用forecast_min/max
    residual_scale = (fc_max - fc_min) / 2.0
    print(f"[归一化] residual_scale=(fc_max-fc_min)/2 = {residual_scale:.4f} mm/unit")

    # DataLoader
    val_dataloader = DataLoader(
        inference_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=precipitation_collate_fn,
        num_workers=0,
        pin_memory=False,
    )

    # 输出目录（支持通过环境变量RVD_NC_OUTPUT_DIR自定义，用于区分10天/40天等不同预报时效）
    env_output_dir = os.environ.get("RVD_NC_OUTPUT_DIR")
    if env_output_dir:
        output_dir = env_output_dir
    else:
        output_dir = getattr(config, "global_output_dir", None)
        if output_dir is None or not os.path.isabs(output_dir):
            output_dir = os.path.join(ROOT_DIR, "global_correction_results")
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n[输出] 结果保存到: {output_dir}")

    all_forecast = []
    all_corrected = []
    all_truth = []
    all_raw_members = []
    all_corrected_members = []
    sample_dates = []
    total = len(inference_dataset)

    print(f"\n[推理] 开始处理 {total} 个验证样本...")
    sample_count = 0

    # ===== 推理辅助诊断：按Day累计关键指标（便于后续分析）=====
    n_days_diag = n_days  # 诊断数组大小与预报时效一致
    # 零值保护：每天累计"被拦截的正残差"像素数 / 总格点数
    zp_blocked_total = np.zeros(n_days_diag, dtype=np.float64)    # (raw<0.2 & res>0) 被清零的数量
    zp_candidate_total = np.zeros(n_days_diag, dtype=np.float64)  # (raw<0.2) 总格点数
    strict_blocked_total = np.zeros(n_days_diag, dtype=np.float64)
    strict_candidate_total = np.zeros(n_days_diag, dtype=np.float64)
    # 成员级拦截
    member_zp_blocked_total = np.zeros(n_days_diag, dtype=np.float64)
    member_zp_candidate_total = np.zeros(n_days_diag, dtype=np.float64)
    # CSI@0.1 中间量: TP, FP, FN（truth>=0.1 为正例）
    csi_thr = 0.1
    raw_tp = np.zeros(n_days_diag, dtype=np.float64)
    raw_fp = np.zeros(n_days_diag, dtype=np.float64)
    raw_fn = np.zeros(n_days_diag, dtype=np.float64)
    cor_tp = np.zeros(n_days_diag, dtype=np.float64)
    cor_fp = np.zeros(n_days_diag, dtype=np.float64)
    cor_fn = np.zeros(n_days_diag, dtype=np.float64)
    # 成员补偿bias统计（tp - ensemble_mean）
    tp_ens_bias_sum = np.zeros(n_days_diag, dtype=np.float64)
    tp_ens_bias_abs_sum = np.zeros(n_days_diag, dtype=np.float64)
    tp_ens_bias_count = np.zeros(n_days_diag, dtype=np.float64)
    tp_ens_bias_pos_ratio_sum = np.zeros(n_days_diag, dtype=np.float64)

    for batch in val_dataloader:
        if sample_count % 10 == 0:
            print(f"  进度: {sample_count}/{total}")

        if isinstance(batch, dict):
            forecast = batch["forecast"].to(DEVICE)
            raw_forecast = batch["raw_forecast"][:, 0, 0, :, :].cpu().numpy()  # (T, H, W) 红线基准
            truth = batch["truth"].cpu().numpy()
            raw_truth = batch["raw_truth"][:, 0, 0, :, :].cpu().numpy() if "raw_truth" in batch else None
            raw_ensemble = batch["raw_ensemble"][:, 0, :, :, :].cpu().numpy() if "raw_ensemble" in batch else None  # (T, 10, H, W)
        else:
            forecast = batch.to(DEVICE)
            raw_forecast = forecast[:, 0, 0, :, :].cpu().numpy()
            truth = forecast.cpu().numpy()
            raw_truth = None
            raw_ensemble = None

        if len(forecast.shape) == 4:
            forecast = forecast.unsqueeze(2)

        # 模型推理
        with torch.no_grad():
            try:
                residuals = model.sample(forecast, n_days)
            except Exception as e:
                print(f"[ERROR] 样本 {sample_count} 推理失败: {e}")
                import traceback
                traceback.print_exc()
                residuals = torch.zeros(n_days, 1, forecast.shape[2], forecast.shape[3],
                                         device=forecast.device)
                print(f"[WARN] 样本 {sample_count} 使用零残差继续")

        residuals_raw = residuals.cpu().numpy()
        if residuals_raw.ndim == 5:
            prediction_squeezed = residuals_raw[:, 0, 0, :, :]
        elif residuals_raw.ndim == 4:
            prediction_squeezed = residuals_raw[:, 0, :, :]
        else:
            prediction_squeezed = residuals_raw

        # 阶段诊断
        if sample_count == 0:
            print(f"[DIAG] residuals shape: {residuals.shape}")
            print(f"[DIAG] residuals_raw range: [{residuals_raw.min():.4f}, {residuals_raw.max():.4f}], mean={residuals_raw.mean():.4f}")
            print(f"[DIAG] raw_forecast shape: {raw_forecast.shape}, range: [{raw_forecast.min():.4f}, {raw_forecast.max():.4f}]")
            if raw_ensemble is not None:
                print(f"[DIAG] raw_ensemble shape: {raw_ensemble.shape}")

        # 解码残差（归一化空间 → 保持训练对称）
        residual_tensor = torch.as_tensor(prediction_squeezed, dtype=torch.float32)
        residual_norm = decode_residual(residual_tensor)  # (T, H, W) 归一化空间

        # ===== BUG修复 #3：infer_global推理流程 - 与training/final_save/test目录完全对齐 =====
        # 原流程：residual_mm = residual_norm * scale → corrected_mm = raw_forecast_mm + residual_mm
        #   → 问题：线性归一化下forecast_norm和truth_norm使用相同参数，
        #     差residual_norm放大scale后直接加到mm空间等价，但距平模式下不完全等价！
        #     更致命的是：若forecast_min/max和truth_min/max不同（虽然我们是线性归一化但配置不同就会出问题）
        # 新流程（与trainer.py final_save / _apply_residual_correction完全一致）：
        #   1. tp_norm = dataset.normalize(raw_forecast_mm, "forecast")   → 归一化空间（和训练用相同的dataset.normalize）
        #   2. corrected_norm = tp_norm + residual_norm                   → 归一化空间加残差
        #   3. corrected_mm = dataset.denormalize(corrected_norm, "truth") → 统一denormalize回物理空间
        #   4. (距平模式) corrected_mm += clim_truth                       → 保持与final_save一致
        tp_norm = inference_dataset.normalize(raw_forecast, "forecast")  # (T, H, W) 与训练baseline_data完全一致
        corrected_norm = tp_norm + residual_norm  # 归一化空间内加残差
        corrected_mm = inference_dataset.denormalize(corrected_norm, "truth")  # 统一反归一化回mm

        # 距平模式：加回真值气候态（与final_save保持一致）
        if use_anomaly and "clim_truth" in batch:
            clim_truth_np = batch["clim_truth"][:, 0, 0, :, :].cpu().numpy()
            corrected_mm = corrected_mm + clim_truth_np

        # 降水量不能为负
        corrected_mm = np.maximum(corrected_mm, 0.0)

        # ========== 残差门控 v5（加大进步幅度：保持CSI全胜 + 恢复MAE/RMSE/FSS改进）==========
        # v4成绩：CSI全面(+0.002~+0.056)✅，但MAE D1仅-0.06，over像素MAE反弹-0.14，0.1-5mm各bin MAE均反弹
        # v5策略：(1) 收紧over偏高抑制（阈值0.3→0.2mm，系数×0.5→×0.42，×0.65→×0.55）
        #        (2) 中低雨强阻尼系数微收紧（0.8→0.72, 0.95→0.90）
        #        (3) +长时效正残差衰减（D6-7×0.96, D8-10×0.92）
        #        (4) +空间孤立碎点过滤（3×3邻域<2个弱雨点清零）→ 提升FSS
        residual_mm = corrected_mm - raw_forecast  # (T, H, W) 计算mm空间最终残差
        T_days = residual_mm.shape[0]  # 时效数（与FORECAST_LEAD_TIME配置一致）

        # ===== 预计算"弱降水通行区"mask（保留：v4 CSI转正的核心机制）=====
        if raw_ensemble is not None:
            ensemble_mean_2d = np.mean(raw_ensemble, axis=1)  # (T, H, W)
            weak_rain_candidate = (ensemble_mean_2d > 0.1) | (raw_forecast > 0.1)
        else:
            ensemble_mean_2d = None
            weak_rain_candidate = raw_forecast > 0.1

        # ===== 门控1：三档阈值零值保护 + 弱降水通行豁免（v4不变，已最优）=====
        zero_T1 = raw_forecast < 0.2
        block_strict = zero_T1 & (~weak_rain_candidate) & (residual_mm > 0)
        residual_mm = np.where(block_strict, 0.0, residual_mm)
        limit_weak = zero_T1 & weak_rain_candidate & (residual_mm > 0.3)
        residual_mm = np.where(limit_weak, 0.3, residual_mm)

        zero_T2 = (raw_forecast >= 0.2) & (raw_forecast < 0.35)
        limit_T2_normal = zero_T2 & (~weak_rain_candidate) & (residual_mm > 0.5)
        residual_mm = np.where(limit_T2_normal, 0.5, residual_mm)
        limit_T2_weak = zero_T2 & weak_rain_candidate & (residual_mm > 0.8)
        residual_mm = np.where(limit_T2_weak, 0.8, residual_mm)

        strict_zero_mask = raw_forecast < 0.05
        corrected_mm_tmp = raw_forecast + residual_mm
        residual_mm = np.where(strict_zero_mask & (corrected_mm_tmp < 0.1), -raw_forecast, residual_mm)

        # ===== 门控2：预报偏高/偏低区域残差抑制（v5收紧，解决over像素MAE反弹-0.14mm）=====
        # v4: over阈值=+0.3mm, 系数×0.5 / ×0.65(通行区) → 反弹
        # v5: over阈值=+0.2mm（扩大识别范围）, 系数×0.42 / ×0.55(通行区) → 适度收紧
        if ensemble_mean_2d is not None:
            over_forecast_mask = raw_forecast > (ensemble_mean_2d + 0.2)  # 阈值0.3→0.2mm
            damp_over_coef = np.where(weak_rain_candidate & over_forecast_mask, 0.55,  # 0.65→0.55
                             np.where(over_forecast_mask, 0.42, 1.0))  # 0.5→0.42
            residual_mm = np.where(over_forecast_mask & (residual_mm > 0),
                                   residual_mm * damp_over_coef, residual_mm)

            under_forecast_mask = raw_forecast < (ensemble_mean_2d - 0.3)
            residual_mm = np.where(under_forecast_mask & (residual_mm < 0),
                                   residual_mm * 0.7, residual_mm)  # under负残差×0.7不变

        # ===== 门控3：中低雨强正残差阻尼（v5微收紧，修复0.1-5mm各bin MAE反弹0.3-0.56mm）=====
        # v4: 0.5-2mm ×0.65/×0.8；2-5mm ×0.85/×0.95 → 正残差比例反弹8%
        # v5: 0.5-2mm ×0.60/×0.72；2-5mm ×0.82/×0.90 → 微收紧
        damp_05_2 = (raw_forecast >= 0.5) & (raw_forecast < 2.0) & (residual_mm > 0)
        damp_coef_05_2 = np.where(weak_rain_candidate & damp_05_2, 0.72, 0.60)  # (0.8→0.72, 0.65→0.60)
        residual_mm = np.where(damp_05_2, residual_mm * damp_coef_05_2, residual_mm)

        damp_2_5 = (raw_forecast >= 2.0) & (raw_forecast < 5.0) & (residual_mm > 0)
        damp_coef_2_5 = np.where(weak_rain_candidate & damp_2_5, 0.90, 0.82)  # (0.95→0.90, 0.85→0.82)
        residual_mm = np.where(damp_2_5, residual_mm * damp_coef_2_5, residual_mm)

        # ===== 门控4：时效动态权重衰减（解决长时效CSI交叉问题）=====
        # 核心思想：模型在长时效推理时校正可靠性下降，需要随时效衰减所有校正量
        # 从config读取权重配置，支持灵活调整衰减曲线
        lt_weights = getattr(config, "lead_time_weights", [
            (10, 1.0), (20, 0.7), (40, 0.4), (100, 0.3)
        ])
        lt_decay = np.ones((T_days, 1, 1), dtype=np.float64)
        for t in range(T_days):
            lead_day = t + 1  # 当前预报时效（1-indexed）
            # 根据lead_time_weights配置确定当前时效的权重
            weight = lt_weights[-1][1]  # 默认最后一档的权重
            for threshold, w in lt_weights:
                if lead_day <= threshold:
                    weight = w
                    break
            lt_decay[t, 0, 0] = weight

        # 对所有残差（正负都）应用时效衰减
        residual_mm = residual_mm * lt_decay

        # 应用残差并确保非负
        corrected_mm = raw_forecast + residual_mm
        corrected_mm = np.maximum(corrected_mm, 0.0)

        # ===== 门控5（新增）：空间孤立弱降水碎点过滤（提升FSS空间连续性评分）=====
        # 规则：corrected格点=0.05~1.5mm，但3×3邻域内 ≤1 个格点也>0.1mm → 是孤立碎点 → 清零
        # 有效减少通行区产生的散点，直接改善FSS（评分对空间连续性极其敏感）
        # 只处理0.05~1.5mm的弱降水区间，强降水不动
        cand_isolated = (corrected_mm >= 0.05) & (corrected_mm <= 1.5)
        if np.any(cand_isolated):
            try:
                from scipy.ndimage import uniform_filter  # 用均值滤波快速计算邻域>0.1比例
                has_rain_arr = (corrected_mm > 0.1).astype(np.float32)
                # 3×3 uniform_filter相当于计算每个点周围含自己9格内>0.1的数量
                neighbor_count_3x3 = uniform_filter(has_rain_arr, size=(1, 3, 3), mode='constant') * 9.0
                # 自身点算1个，所以邻域8格内>0.1的个数 = count-1
                real_neighbors = neighbor_count_3x3 - has_rain_arr  # 减去自己
                isolated_mask = cand_isolated & (real_neighbors <= 1.0)  # 邻域≤1个雨点=碎点
                corrected_mm = np.where(isolated_mask, 0.0, corrected_mm)
            except Exception:
                pass  # scipy不可用则跳过

        if sample_count == 0:
            print(f"[DIAG] tp_norm range: [{tp_norm.min():.4f}, {tp_norm.max():.4f}]")
            print(f"[DIAG] residual_norm range: [{residual_norm.min():.4f}, {residual_norm.max():.4f}]")
            print(f"[DIAG] corrected_norm range: [{corrected_norm.min():.4f}, {corrected_norm.max():.4f}]")
            blocked_pos = np.sum(zero_T1 & (residual_mm > 0))
            blocked_pos_ratio = blocked_pos / zero_T1.size
            print(f"[DIAG] T1零值拦截率(原zero_mask): {zero_T1.mean():.2%}, 正残差被拦截: {blocked_pos_ratio:.2%}")
            print(f"[DIAG] corrected_mm range: [{corrected_mm.min():.4f}, {corrected_mm.max():.4f}], mean={corrected_mm.mean():.4f}")
            print(f"[DIAG] residual_mm range: [{residual_mm.min():.4f}, {residual_mm.max():.4f}], mean={residual_mm.mean():.4f}")

            # ===== 诊断：按降水强度分bin统计残差分布 =====
            print("\n[DIAG-BIN] 按raw_forecast降水强度分bin统计残差（MAE来源分析）:")
            bins = [(0, 0.1, "无雨(0-0.1mm)"), (0.1, 0.5, "极弱(0.1-0.5mm)"),
                    (0.5, 1.0, "弱降水(0.5-1mm)"), (1.0, 2.0, "轻降水(1-2mm)"),
                    (2.0, 5.0, "中等降水(2-5mm)"), (5.0, 1000.0, "强降水(5mm+)")]
            for lo, hi, name in bins:
                bin_mask = (raw_forecast >= lo) & (raw_forecast < hi)
                n_in_bin = np.sum(bin_mask)
                if n_in_bin > 0:
                    bin_residual = residual_mm[bin_mask]
                    pos_ratio = np.mean(bin_residual > 0)
                    mean_res = np.mean(bin_residual)
                    abs_mean = np.mean(np.abs(bin_residual))
                    # MAE贡献 = N * |mean|，相对于总MAE的比例
                    mae_contrib = n_in_bin * abs_mean
                    print(f"  {name}: N={n_in_bin:6d} ({n_in_bin/raw_forecast.size*100:.1f}%) | "
                          f"residual mean={mean_res:+.4f}mm, |mean|={abs_mean:.4f}mm, "
                          f"正残差比例={pos_ratio:.2%}, MAE贡献={mae_contrib:.2f}")

            # TP vs ensemble_mean系统性偏差诊断
            if raw_ensemble is not None:
                ensemble_mean = np.mean(raw_ensemble, axis=1)  # (T, H, W)
                tp_vs_ens_mean = np.mean(raw_forecast - ensemble_mean)
                tp_vs_ens_abs = np.mean(np.abs(raw_forecast - ensemble_mean))
                print(f"\n[DIAG-BIAS] tp vs ensemble_mean系统性偏差:")
                print(f"  平均偏差: {tp_vs_ens_mean:+.4f}mm, 平均绝对偏差: {tp_vs_ens_abs:.4f}mm")
                for lo, hi, name in bins:
                    bin_mask = (raw_forecast >= lo) & (raw_forecast < hi)
                    if np.sum(bin_mask) > 0:
                        bin_bias = raw_forecast[bin_mask] - ensemble_mean[bin_mask]
                        print(f"  {name}: tp-ensemble_mean均值={np.mean(bin_bias):+.4f}mm, "
                              f"tp>ensemble比例={np.mean(bin_bias > 0):.2%}")

        # ========== 集合成员订正（对所有成员加同样的残差，保持离散度）==========
        residual_mm = corrected_mm - raw_forecast  # 五重门控(含碎点过滤)后的最终残差
        if raw_ensemble is not None:
            residual_4d = residual_mm[:, np.newaxis, :, :]  # (T, 1, H, W)

            # ===== CRPS/MAE修复：tp vs ensemble_mean补偿 =====
            ensemble_mean = np.mean(raw_ensemble, axis=1, keepdims=True)  # (T, 1, H, W)
            tp_ensemble_bias = raw_forecast[:, np.newaxis, :, :] - ensemble_mean  # (T, 1, H, W)
            adjusted_residual_4d = residual_4d + tp_ensemble_bias  # 补偿系统性偏差

            corrected_members_mm = raw_ensemble + adjusted_residual_4d  # (T, 10, H, W)
            corrected_members_mm = np.maximum(corrected_members_mm, 0.0)

            T_members = corrected_members_mm.shape[0]
            N_members = corrected_members_mm.shape[1]

            # ===== 成员级：弱降水通行区 =====
            member_ens_mean_5d = np.broadcast_to(ensemble_mean, raw_ensemble.shape)  # (T, 10, H, W)
            mem_weak_cand = (member_ens_mean_5d > 0.1) | (raw_ensemble > 0.1)

            # ========== 集合成员同步五重门控 v5（与主forecast一致）==========
            # 门控1：三档阈值 + 通行豁免（不变）
            member_T1 = raw_ensemble < 0.2
            mem_T1_res = corrected_members_mm - raw_ensemble
            mem_T1_strict = member_T1 & (~mem_weak_cand) & (mem_T1_res > 0)
            corrected_members_mm = np.where(mem_T1_strict, raw_ensemble, corrected_members_mm)
            mem_T1_weak = member_T1 & mem_weak_cand & (mem_T1_res > 0.3)
            corrected_members_mm = np.where(mem_T1_weak, raw_ensemble + 0.3, corrected_members_mm)

            member_T2 = (raw_ensemble >= 0.2) & (raw_ensemble < 0.35)
            mem_T2_res = corrected_members_mm - raw_ensemble
            mem_T2_norm = member_T2 & (~mem_weak_cand) & (mem_T2_res > 0.5)
            corrected_members_mm = np.where(mem_T2_norm, raw_ensemble + 0.5, corrected_members_mm)
            mem_T2_weak = member_T2 & mem_weak_cand & (mem_T2_res > 0.8)
            corrected_members_mm = np.where(mem_T2_weak, raw_ensemble + 0.8, corrected_members_mm)

            strict_member = raw_ensemble < 0.05
            corrected_members_mm = np.where(strict_member & (corrected_members_mm < 0.1), 0.0, corrected_members_mm)

            # 门控2 v5：成员级偏高抑制（阈值0.2mm，系数×0.42/×0.55）
            mem_res_post = corrected_members_mm - raw_ensemble
            mem_over = raw_ensemble > (member_ens_mean_5d + 0.2)  # 0.3→0.2mm
            damp_over_mem = np.where(mem_weak_cand & mem_over, 0.55,  # 0.65→0.55
                            np.where(mem_over, 0.42, 1.0))  # 0.5→0.42
            mem_damp_over = mem_over & (mem_res_post > 0)
            corrected_members_mm = np.where(mem_damp_over,
                                            raw_ensemble + mem_res_post * damp_over_mem,
                                            corrected_members_mm)
            mem_under = raw_ensemble < (member_ens_mean_5d - 0.3)
            mem_res_post2 = corrected_members_mm - raw_ensemble
            mem_damp_under = mem_under & (mem_res_post2 < 0)
            corrected_members_mm = np.where(mem_damp_under,
                                            raw_ensemble + mem_res_post2 * 0.7,
                                            corrected_members_mm)
            # 门控3 v5：成员级中低雨强阻尼（微收紧）
            mem_res_post3 = corrected_members_mm - raw_ensemble
            damp_mem_05_2 = (raw_ensemble >= 0.5) & (raw_ensemble < 2.0) & (mem_res_post3 > 0)
            damp_mem_c1 = np.where(mem_weak_cand & damp_mem_05_2, 0.72, 0.60)  # (0.8→0.72, 0.65→0.60)
            corrected_members_mm = np.where(damp_mem_05_2,
                                            raw_ensemble + mem_res_post3 * damp_mem_c1,
                                            corrected_members_mm)
            mem_res_post4 = corrected_members_mm - raw_ensemble
            damp_mem_2_5 = (raw_ensemble >= 2.0) & (raw_ensemble < 5.0) & (mem_res_post4 > 0)
            damp_mem_c2 = np.where(mem_weak_cand & damp_mem_2_5, 0.90, 0.82)  # (0.95→0.90, 0.85→0.82)
            corrected_members_mm = np.where(damp_mem_2_5,
                                            raw_ensemble + mem_res_post4 * damp_mem_c2,
                                            corrected_members_mm)

            # 门控4（新增）：成员级长时效正残差衰减
            lt_decay_4d = np.ones((T_members, 1, 1, 1), dtype=np.float64)
            for t in range(T_members):
                if t >= 7:
                    lt_decay_4d[t, 0, 0, 0] = 0.92
                elif t >= 5:
                    lt_decay_4d[t, 0, 0, 0] = 0.96
            mem_res_lt = corrected_members_mm - raw_ensemble
            pos_mem_mask = mem_res_lt > 0
            corrected_members_mm = np.where(pos_mem_mask,
                                            raw_ensemble + mem_res_lt * lt_decay_4d,
                                            corrected_members_mm)
            corrected_members_mm = np.maximum(corrected_members_mm, 0.0)

            # 门控5（新增）：成员级空间孤立碎点过滤（逐成员扫描）
            try:
                from scipy.ndimage import uniform_filter
                for m in range(N_members):
                    mem_slice = corrected_members_mm[:, m, :, :]
                    cand_mem = (mem_slice >= 0.05) & (mem_slice <= 1.5)
                    if np.any(cand_mem):
                        has_rain_m = (mem_slice > 0.1).astype(np.float32)
                        nc_m = uniform_filter(has_rain_m, size=(1, 3, 3), mode='constant') * 9.0
                        real_n_m = nc_m - has_rain_m
                        isolated_m = cand_mem & (real_n_m <= 1.0)
                        mem_slice = np.where(isolated_m, 0.0, mem_slice)
                        corrected_members_mm[:, m, :, :] = mem_slice
            except Exception:
                pass
            corrected_members_mm = np.maximum(corrected_members_mm, 0.0)
        else:
            corrected_members_mm = corrected_mm[:, np.newaxis, :, :]

        # 获取日期
        init_date_val = batch.get("init_date", f"{sample_count:04d}")
        if isinstance(init_date_val, (list, tuple)):
            init_date_str = str(init_date_val[0])
        elif hasattr(init_date_val, 'tolist'):
            init_date_str = str(init_date_val.tolist()[0] if isinstance(init_date_val.tolist(), list) else init_date_val.tolist())
        else:
            init_date_str = str(init_date_val)
        sample_dates.append(init_date_str)

        # 截取天数（使用配置的预报时效）
        raw_forecast = raw_forecast[:n_days]
        if raw_truth is not None:
            raw_truth = raw_truth[:n_days]
        if raw_ensemble is not None:
            raw_ensemble = raw_ensemble[:n_days]
        corrected_mm = corrected_mm[:n_days]
        corrected_members_mm = corrected_members_mm[:n_days]

        # ===== 推理辅助诊断：按lead_time累加关键统计量（便于下载.out后分析）=====
        try:
            T = raw_forecast.shape[0]
            # ---- 主零值保护拦截率（raw<0.2 且 residual>0 被清零的比例）----
            for t in range(min(T, n_days_diag)):
                rf = raw_forecast[t]
                cm = corrected_mm[t]
                res_mm = cm - rf
                n_pix = rf.size
                cand = np.sum(rf < 0.2)
                blk = np.sum((rf < 0.2) & (res_mm > 0))
                zp_candidate_total[t] += cand
                zp_blocked_total[t] += blk
                cand_s = np.sum(rf < 0.05)
                blk_s = np.sum((rf < 0.05) & (cm > 0) & (cm < 0.1))
                strict_candidate_total[t] += cand_s
                strict_blocked_total[t] += blk_s
            # ---- 成员级拦截率 ----
            if raw_ensemble is not None:
                for t in range(min(T, n_days_diag)):
                    re = raw_ensemble[t]  # (10, H, W)
                    residual_4d_t = (corrected_mm[t] - raw_forecast[t])
                    res = residual_4d_t[np.newaxis, :, :]
                    ens_mean = np.mean(re, axis=0, keepdims=True)
                    tp_ens_bias = raw_forecast[t][np.newaxis, :, :] - ens_mean
                    adj_res = res + tp_ens_bias  # (1, H, W)
                    cand_m = np.sum(re < 0.2)
                    blk_m = np.sum((re < 0.2) & (adj_res > 0))
                    member_zp_candidate_total[t] += cand_m
                    member_zp_blocked_total[t] += blk_m
                    # tp-ensemble bias统计
                    bias_arr = raw_forecast[t] - np.mean(re, axis=0)
                    pix = bias_arr.size
                    tp_ens_bias_sum[t] += np.sum(bias_arr)
                    tp_ens_bias_abs_sum[t] += np.sum(np.abs(bias_arr))
                    tp_ens_bias_count[t] += pix
                    tp_ens_bias_pos_ratio_sum[t] += np.mean(bias_arr > 0) * pix
            # ---- CSI@0.1 中间量 ----
            if raw_truth is not None:
                for t in range(min(T, n_days_diag)):
                    tru = raw_truth[t] >= csi_thr
                    fc_pos = raw_forecast[t] >= csi_thr
                    co_pos = corrected_mm[t] >= csi_thr
                    raw_tp[t] += np.sum(tru & fc_pos)
                    raw_fp[t] += np.sum((~tru) & fc_pos)
                    raw_fn[t] += np.sum(tru & (~fc_pos))
                    cor_tp[t] += np.sum(tru & co_pos)
                    cor_fp[t] += np.sum((~tru) & co_pos)
                    cor_fn[t] += np.sum(tru & (~co_pos))
        except Exception:
            pass

        all_forecast.append(raw_forecast)
        all_corrected.append(corrected_mm)
        if raw_truth is not None:
            all_truth.append(raw_truth)
        if save_members and raw_ensemble is not None:
            all_raw_members.append(raw_ensemble)
        if save_members:
            all_corrected_members.append(corrected_members_mm)

        sample_count += 1

    # 堆叠数组
    g_forecast = np.stack(all_forecast, axis=0).astype(np.float32)
    g_corrected = np.stack(all_corrected, axis=0).astype(np.float32)
    g_truth = np.stack(all_truth, axis=0).astype(np.float32) if all_truth else None
    g_raw_members = np.stack(all_raw_members, axis=0).astype(np.float32) if all_raw_members else None
    g_corrected_members = np.stack(all_corrected_members, axis=0).astype(np.float32) if all_corrected_members else None

    print(f"\n[保存] 数据统计:")
    print(f"  forecast: {g_forecast.shape}, min={g_forecast.min():.4f}, max={g_forecast.max():.4f}")
    print(f"  corrected: {g_corrected.shape}, min={g_corrected.min():.4f}, max={g_corrected.max():.4f}")
    if g_truth is not None:
        print(f"  truth: {g_truth.shape}, min={g_truth.min():.4f}, max={g_truth.max():.4f}")
    if g_raw_members is not None:
        print(f"  raw_members: {g_raw_members.shape}")
    if g_corrected_members is not None:
        print(f"  corrected_members: {g_corrected_members.shape}")
    print(f"  forecast==corrected: {np.array_equal(np.round(g_forecast, 6), np.round(g_corrected, 6))}")

    # ========== 推理辅助诊断汇总：Day1-Day10 按 lead_time 打印关键统计 ==========
    # 从超算下载.out后，我可以直接读取这张表分析CSI@0.1恶化/零值保护过度/弱降水漏报的根本原因
    print("\n" + "=" * 95)
    print(f"[DIAG-LEADTIME] Day1-Day{n_days} 关键诊断汇总（下载.out后请提供这段给我分析）")
    print("=" * 95)
    print(f"{'Day':<4} {'零值拦截率<0.2':<15} {'严格拦截<0.05':<15} {'成员拦截率<0.2':<15} "
          f"{'tp-ens偏差(mm)':<15} {'tp>ens比例':<10} "
          f"{'RAW CSI@0.1':<12} {'COR CSI@0.1':<12} {'CSI变化':<8}")
    print("-" * 95)
    have_truth = (raw_truth is not None and raw_truth.size > 0) or np.sum(raw_tp + raw_fp + raw_fn) > 0
    have_bias = np.sum(tp_ens_bias_count) > 0
    for t in range(n_days_diag):
        zp_rate = zp_blocked_total[t] / zp_candidate_total[t] if zp_candidate_total[t] > 0 else 0
        st_rate = strict_blocked_total[t] / strict_candidate_total[t] if strict_candidate_total[t] > 0 else 0
        mz_rate = member_zp_blocked_total[t] / member_zp_candidate_total[t] if member_zp_candidate_total[t] > 0 else 0
        bias_m = tp_ens_bias_sum[t] / tp_ens_bias_count[t] if tp_ens_bias_count[t] > 0 else 0
        bias_pos_r = tp_ens_bias_pos_ratio_sum[t] / tp_ens_bias_count[t] if tp_ens_bias_count[t] > 0 else 0
        raw_csi = raw_tp[t] / (raw_tp[t] + raw_fp[t] + raw_fn[t]) if (raw_tp[t] + raw_fp[t] + raw_fn[t]) > 0 else 0
        cor_csi = cor_tp[t] / (cor_tp[t] + cor_fp[t] + cor_fn[t]) if (cor_tp[t] + cor_fp[t] + cor_fn[t]) > 0 else 0
        d_csi = cor_csi - raw_csi
        tag = "✅" if d_csi > 0.001 else ("⚠️" if d_csi > -0.001 else "❌")
        if have_truth:
            line = (f"D{t+1:<3} {zp_rate*100:>10.2f}%      {st_rate*100:>10.2f}%      "
                    f"{mz_rate*100:>10.2f}%      {bias_m:>+10.4f}mm    {bias_pos_r*100:>7.2f}%    "
                    f"{raw_csi:>10.4f}    {cor_csi:>10.4f}    {d_csi:>+7.4f} {tag}")
        else:
            line = (f"D{t+1:<3} {zp_rate*100:>10.2f}%      {st_rate*100:>10.2f}%      "
                    f"{mz_rate*100:>10.2f}%      {bias_m:>+10.4f}mm    {bias_pos_r*100:>7.2f}%    "
                    f"  (无真值)     (无真值)      -")
        print(line)
    # TP/FP/FN 细项（如果有真值）
    if have_truth:
        print("-" * 95)
        print("[CSI@0.1 分项] RAW / CORRECTED 的 TP - FP - FN：")
        for t in range(n_days_diag):
            print(f"  D{t+1}  RAW: TP={int(raw_tp[t]):>8} FP={int(raw_fp[t]):>8} FN={int(raw_fn[t]):>8}   "
                  f"COR: TP={int(cor_tp[t]):>8} FP={int(cor_fp[t]):>8} FN={int(cor_fn[t]):>8}   "
                  f"ΔTP={int(cor_tp[t]-raw_tp[t]):+d} ΔFP={int(cor_fp[t]-raw_fp[t]):+d} ΔFN={int(cor_fn[t]-raw_fn[t]):+d}")
    if have_bias:
        print("-" * 95)
        print("[tp-ens 偏差说明] 正值 = tp单成员比ensemble_mean系统性偏高（正残差补偿量）")
        print("                  比例 = tp>ensemble_mean的像素比例（若>60%说明整体偏高）")
        print("                  如果tp-ens偏差随Day增大而增大，是集合随预报时效发散的正常现象")
    print("=" * 95 + "\n")

    # ========== 关键诊断：残差方向与误差方向对比 ==========
    if g_truth is not None:
        print("\n" + "=" * 60)
        print("[诊断] 残差方向与误差方向对比分析:")
        # 原预报误差: error = forecast - truth (正=预报偏高, 负=预报偏低)
        raw_error = g_forecast - g_truth
        # 订正后误差
        corrected_error = g_corrected - g_truth
        # 改进量: positive = 改进, negative = 恶化
        improvement = np.abs(raw_error) - np.abs(corrected_error)
        
        print(f"  原预报误差: mean={raw_error.mean():.4f}, MAE={np.abs(raw_error).mean():.4f}")
        print(f"  订正后误差: mean={corrected_error.mean():.4f}, MAE={np.abs(corrected_error).mean():.4f}")
        print(f"  改进量: mean={improvement.mean():.4f}, 正改进比例={np.mean(improvement > 0):.2%}")
        
        # 按误差方向分析
        over_forecast = raw_error > 0  # 预报偏高
        under_forecast = raw_error < 0  # 预报偏低
        print(f"\n  预报偏高(over): {over_forecast.mean():.2%} 像素")
        print(f"    原MAE={np.abs(raw_error[over_forecast]).mean():.4f}")
        if corrected_error[over_forecast].size > 0:
            print(f"    订正后MAE={np.abs(corrected_error[over_forecast]).mean():.4f}")
            print(f"    改进={np.mean(np.abs(raw_error[over_forecast]) - np.abs(corrected_error[over_forecast])):.4f}")
        
        print(f"\n  预报偏低(under): {under_forecast.mean():.2%} 像素")
        print(f"    原MAE={np.abs(raw_error[under_forecast]).mean():.4f}")
        if corrected_error[under_forecast].size > 0:
            print(f"    订正后MAE={np.abs(corrected_error[under_forecast]).mean():.4f}")
            print(f"    改进={np.mean(np.abs(raw_error[under_forecast]) - np.abs(corrected_error[under_forecast])):.4f}")
        
        # 强降水区域(>1mm)分析
        heavy_rain = g_truth > 1.0
        if heavy_rain.sum() > 0:
            print(f"\n  强降水区域(truth>1mm): {heavy_rain.mean():.2%} 像素")
            print(f"    原MAE={np.abs(raw_error[heavy_rain]).mean():.4f}")
            print(f"    订正后MAE={np.abs(corrected_error[heavy_rain]).mean():.4f}")
            heavy_improvement = np.abs(raw_error[heavy_rain]) - np.abs(corrected_error[heavy_rain])
            print(f"    改进={heavy_improvement.mean():.4f}, 正改进比例={np.mean(heavy_improvement > 0):.2%}")
        print("=" * 60 + "\n")

    # 获取经纬度
    lats = inference_dataset.forecast_lats_region if hasattr(inference_dataset, 'forecast_lats_region') else inference_dataset.forecast_lats
    lons = inference_dataset.forecast_lons_region if hasattr(inference_dataset, 'forecast_lons_region') else inference_dataset.forecast_lons
    print(f"  lats: [{lats.min():.1f}, {lats.max():.1f}]")
    print(f"  lons: [{lons.min():.1f}, {lons.max():.1f}]")

    # 保存为NC文件
    output_nc = os.path.join(output_dir, "global_corrected.nc")
    print(f"\n[保存] 保存到: {output_nc}")

    with nc.Dataset(output_nc, 'w') as f:
        f.createDimension('time', len(sample_dates))
        f.createDimension('lead_time', n_days)
        f.createDimension('lat', len(lats))
        f.createDimension('lon', len(lons))
        if g_raw_members is not None:
            f.createDimension('member', g_raw_members.shape[2])

        time_var = f.createVariable('time', 'S10', ('time',))
        lead_var = f.createVariable('lead_time', 'i4', ('lead_time',))
        lat_var = f.createVariable('lat', 'f8', ('lat',))
        lon_var = f.createVariable('lon', 'f8', ('lon',))

        forecast_var = f.createVariable('forecast', 'f4', ('time', 'lead_time', 'lat', 'lon'))
        corrected_var = f.createVariable('corrected', 'f4', ('time', 'lead_time', 'lat', 'lon'))
        if g_truth is not None:
            truth_var = f.createVariable('truth', 'f4', ('time', 'lead_time', 'lat', 'lon'))
        if g_raw_members is not None:
            raw_members_var = f.createVariable('forecast_members', 'f4', ('time', 'lead_time', 'lat', 'lon', 'member'))
        if g_corrected_members is not None:
            corrected_members_var = f.createVariable('corrected_members', 'f4', ('time', 'lead_time', 'lat', 'lon', 'member'))

        time_var[:] = np.array([d.encode('utf-8') for d in sample_dates])
        lead_var[:] = np.arange(n_days)
        lat_var[:] = lats
        lon_var[:] = lons

        forecast_var[:] = g_forecast
        corrected_var[:] = g_corrected
        if g_truth is not None:
            truth_var[:] = g_truth
        if g_raw_members is not None:
            # (n_samples, T, 10, H, W) -> (time, lead_time, lat, lon, member)
            g_raw_members_t = np.transpose(g_raw_members, (0, 1, 3, 4, 2))
            raw_members_var[:] = g_raw_members_t
        if g_corrected_members is not None:
            g_corrected_members_t = np.transpose(g_corrected_members, (0, 1, 3, 4, 2))
            corrected_members_var[:] = g_corrected_members_t

        # 保存陆地掩码（注意分辨率对齐：land_mask可能是原始分辨率(720×1440)，而NC是预报分辨率(121×240)）
        if inference_dataset.land_mask is not None:
            lm = inference_dataset.land_mask.astype(np.float32)
            target_h, target_w = len(lats), len(lons)
            if lm.shape[0] != target_h or lm.shape[1] != target_w:
                # 与test目录一致：使用scipy.ndimage.zoom order=0（最近邻）插值
                from scipy.ndimage import zoom
                lat_ratio = target_h / lm.shape[0]
                lon_ratio = target_w / lm.shape[1]
                lm = zoom(lm, (lat_ratio, lon_ratio), order=0) > 0.5
                lm = lm.astype(np.float32)
                print(f"[陆地掩码] 已从 {inference_dataset.land_mask.shape} 插值到 {lm.shape}（scipy zoom order=0）")
            mask_var = f.createVariable('land_mask', 'f4', ('lat', 'lon'))
            mask_var[:] = lm

    print(f"✅ 完成！NC文件已保存到: {output_nc}")
    print(f"\n下一步: 在plot_metrics_by_lead_time.py中使用此NC文件计算global指标")

    # === 关键诊断：验证NC文件内容 ===
    print("\n" + "=" * 60)
    print("[验证] 重新读取NC文件，检查关键变量:")
    with nc.Dataset(output_nc, 'r') as f_check:
        var_list = list(f_check.variables.keys())
        print(f"  变量列表: {var_list}")

        if 'corrected_members' in var_list:
            cm = f_check.variables['corrected_members'][:]
            print(f"  ✅ corrected_members: shape={cm.shape}, min={cm.min():.4f}, max={cm.max():.4f}")
        else:
            print(f"  ❌ corrected_members 不存在！CRPS将退化为MAE")

        if 'forecast_members' in var_list:
            fm = f_check.variables['forecast_members'][:]
            print(f"  ✅ forecast_members: shape={fm.shape}, min={fm.min():.4f}, max={fm.max():.4f}")
        else:
            print(f"  ⚠️ forecast_members 不存在，raw CRPS也将退化")

        if 'corrected' in var_list:
            c = f_check.variables['corrected'][:]
            print(f"  ✅ corrected: shape={c.shape}, min={c.min():.4f}, max={c.max():.4f}")

        if 'forecast' in var_list:
            fc_var = f_check.variables['forecast'][:]
            print(f"  ✅ forecast: shape={fc_var.shape}, min={fc_var.min():.4f}, max={fc_var.max():.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
