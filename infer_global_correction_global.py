"""
infer_global_correction_global.py - 使用主工程模型对全球数据做推理

该脚本与infer_global_correction.py完全独立，专门用于全球区域推理。
使用与中国区域相同的模型和推理逻辑，但数据范围为全球。

运行方法：python infer_global_correction_global.py
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

# ========== 关键修改：设置为全球区域 ==========
ORIGINAL_REGION = config.data_config["region"]
ORIGINAL_LAT_RANGE = config.data_config["lat_range"]
ORIGINAL_LON_RANGE = config.data_config["lon_range"]

config.data_config["region"] = "global"
config.data_config["lat_range"] = (-90, 90)
config.data_config["lon_range"] = (0, 360)

print(f"[配置] 全球区域推理:")
print(f"  region: {config.data_config['region']}")
print(f"  lat_range: {config.data_config['lat_range']}")
print(f"  lon_range: {config.data_config['lon_range']}")


def load_model():
    """加载主工程训练好的模型（与中国区域完全相同的架构）"""
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
        # PIMD参数（与训练端一致）
        use_pimd=getattr(config, "use_pimd", True),
        pimd_num_branches=getattr(config, "pimd_num_branches", 3),
        pimd_temperature=getattr(config, "pimd_temperature", 1.0),
        pimd_entropy_loss_weight=getattr(config, "pimd_entropy_loss_weight", 0.01),
    ).to(DEVICE)

    # 查找检查点（与infer_global_correction.py完全一致，使用current_run_dir）
    # 优先级: 1) 环境变量 RVD_MODEL_CHECKPOINT  2) 自动搜索
    env_ckpt = os.environ.get("RVD_MODEL_CHECKPOINT", "")
    if env_ckpt and os.path.isfile(env_ckpt):
        latest_checkpoint = env_ckpt
        print(f"[模型] 使用环境变量指定的检查点: {latest_checkpoint}")
        # results_root 用于后续诊断，取checkpoint所在run目录的上一级
        results_root = os.path.dirname(os.path.dirname(latest_checkpoint))
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

    # PIMD状态检查
    simple_unet = getattr(model, "simple_unet", None)
    if simple_unet is not None:
        pimd = getattr(simple_unet, "pimd", None)
        if pimd is not None:
            print(f"[模型] PIMD模块: ✅已启用")
        else:
            print(f"[模型] PIMD模块: ❌未启用")
        n_params = sum(p.numel() for p in simple_unet.parameters())
        print(f"[模型] SimpleUNet参数量: {n_params/1e6:.2f}M")
    print("[模型] 模型加载完成")
    return model, results_root


def decode_residual(residual_prediction):
    """解码残差（与中国区域完全一致）"""
    pos_clip = float(getattr(config, "positive_residual_clip_value", 1.0))
    neg_clip = float(getattr(config, "negative_residual_clip_value", 1.0))
    max_abs = float(getattr(config, "max_abs_residual_norm", 1.0))
    
    positive = residual_prediction.clamp(min=0.0) * pos_clip
    negative = residual_prediction.clamp(max=0.0) * neg_clip
    residual_decoded = positive + negative
    residual_decoded = residual_decoded.clamp(min=-max_abs, max=max_abs)
    
    residual_blend = float(getattr(config, "residual_blend_weight", 1.0))
    result = residual_blend * residual_decoded.detach().cpu().numpy()
    return result


def main():
    print("=" * 80)
    print("全球数据订正推理脚本（全球区域版）")
    print("=" * 80)

    # 读取预报时效配置
    n_days = getattr(config, "forecast_lead_time", 10)
    print(f"[配置] 预报时效 n_days = {n_days} 天")

    # 加载模型
    model, results_root = load_model()

    # 加载全球数据集
    print("\n[数据] 加载全球数据集...")
    train_dataset, val_dataset = load_dataset(config.data_config)
    print(f"  训练集: {len(train_dataset)} 个样本")
    print(f"  验证集: {len(val_dataset)} 个样本")

    # 加载归一化参数
    # 从run_dir（results_root的子目录）查找normalization_params.npz
    run_dir = config.current_run_dir
    norm_params_path = os.path.join(run_dir, "normalization_params.npz")
    if os.path.exists(norm_params_path):
        print(f"[归一化] 从文件加载训练时的归一化参数: {norm_params_path}")
        norm_data = np.load(norm_params_path)
        fc_min = float(norm_data["forecast_min"])
        fc_max = float(norm_data["forecast_max"])
        print(f"[归一化] fc_min={fc_min:.4f}, fc_max={fc_max:.4f}")
    else:
        print(f"[警告] 未找到归一化参数文件，使用数据集对象的参数")
        fc_min = float(train_dataset.forecast_min)
        fc_max = float(train_dataset.forecast_max)

    use_anomaly = bool(getattr(config, "use_anomaly", False))
    residual_scale = (fc_max - fc_min) / 2.0

    # DataLoader
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=precipitation_collate_fn,
        num_workers=0,
        pin_memory=False,
    )

    # 输出目录（支持通过环境变量RVD_NC_OUTPUT_DIR自定义，用于区分10天/40天等不同预报时效）
    env_output_dir = os.environ.get("RVD_NC_OUTPUT_DIR")
    if env_output_dir:
        global_output_dir = env_output_dir
    else:
        global_output_dir = os.path.join(run_dir, "nc_global")
    os.makedirs(global_output_dir, exist_ok=True)
    print(f"\n[输出] 全球区域结果保存到: {global_output_dir}")

    all_forecast = []
    all_corrected = []
    all_truth = []
    all_raw_members = []
    all_corrected_members = []
    sample_dates = []
    total = len(val_dataset)

    print(f"\n[推理] 开始处理 {total} 个全球验证样本...")
    sample_count = 0

    for batch in val_dataloader:
        if sample_count % 10 == 0:
            print(f"  进度: {sample_count}/{total}")

        if isinstance(batch, dict):
            forecast = batch["forecast"].to(DEVICE)
            raw_forecast = batch["raw_forecast"][:, 0, 0, :, :].cpu().numpy()
            truth = batch["truth"].cpu().numpy()
            raw_truth = batch["raw_truth"][:, 0, 0, :, :].cpu().numpy() if "raw_truth" in batch else None
            raw_ensemble = batch["raw_ensemble"][:, 0, :, :, :].cpu().numpy() if "raw_ensemble" in batch else None
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
                residuals = torch.zeros(n_days, 1, forecast.shape[2], forecast.shape[3], device=forecast.device)

        residuals_raw = residuals.cpu().numpy()
        if residuals_raw.ndim == 5:
            prediction_squeezed = residuals_raw[:, 0, 0, :, :]
        elif residuals_raw.ndim == 4:
            prediction_squeezed = residuals_raw[:, 0, :, :]
        else:
            prediction_squeezed = residuals_raw

        # 解码残差
        residual_tensor = torch.as_tensor(prediction_squeezed, dtype=torch.float32)
        residual_norm = decode_residual(residual_tensor)

        # 与中国区域完全相同的订正流程
        tp_norm = val_dataset.normalize(raw_forecast, "forecast")
        corrected_norm = tp_norm + residual_norm
        corrected_mm = val_dataset.denormalize(corrected_norm, "truth")

        # 距平模式
        if use_anomaly and "clim_truth" in batch:
            clim_truth_np = batch["clim_truth"][:, 0, 0, :, :].cpu().numpy()
            corrected_mm = corrected_mm + clim_truth_np

        corrected_mm = np.maximum(corrected_mm, 0.0)

        # 残差门控（与中国区域完全相同的v5逻辑）
        residual_mm = corrected_mm - raw_forecast
        T_days = residual_mm.shape[0]

        # 预计算弱降水通行区mask
        if raw_ensemble is not None:
            ensemble_mean_2d = np.mean(raw_ensemble, axis=1)
            weak_rain_candidate = (ensemble_mean_2d > 0.1) | (raw_forecast > 0.1)
        else:
            ensemble_mean_2d = None
            weak_rain_candidate = raw_forecast > 0.1

        # 门控1：三档阈值零值保护
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

        # 门控2：预报偏高/偏低区域残差抑制
        if ensemble_mean_2d is not None:
            over_forecast_mask = raw_forecast > (ensemble_mean_2d + 0.2)
            damp_over_coef = np.where(weak_rain_candidate & over_forecast_mask, 0.55,
                             np.where(over_forecast_mask, 0.42, 1.0))
            residual_mm = np.where(over_forecast_mask & (residual_mm > 0),
                                   residual_mm * damp_over_coef, residual_mm)

            under_forecast_mask = raw_forecast < (ensemble_mean_2d - 0.3)
            residual_mm = np.where(under_forecast_mask & (residual_mm < 0),
                                   residual_mm * 0.7, residual_mm)

        # 门控3：中低雨强正残差阻尼
        damp_05_2 = (raw_forecast >= 0.5) & (raw_forecast < 2.0) & (residual_mm > 0)
        damp_coef_05_2 = np.where(weak_rain_candidate & damp_05_2, 0.72, 0.60)
        residual_mm = np.where(damp_05_2, residual_mm * damp_coef_05_2, residual_mm)

        damp_2_5 = (raw_forecast >= 2.0) & (raw_forecast < 5.0) & (residual_mm > 0)
        damp_coef_2_5 = np.where(weak_rain_candidate & damp_2_5, 0.90, 0.82)
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

        # 门控5：空间孤立碎点过滤
        cand_isolated = (corrected_mm >= 0.05) & (corrected_mm <= 1.5)
        if np.any(cand_isolated):
            try:
                from scipy.ndimage import uniform_filter
                has_rain_arr = (corrected_mm > 0.1).astype(np.float32)
                neighbor_count_3x3 = uniform_filter(has_rain_arr, size=(1, 3, 3), mode='constant') * 9.0
                real_neighbors = neighbor_count_3x3 - has_rain_arr
                isolated_mask = cand_isolated & (real_neighbors <= 1.0)
                corrected_mm = np.where(isolated_mask, 0.0, corrected_mm)
            except Exception:
                pass

        # 集合成员订正
        if raw_ensemble is not None:
            residual_4d = residual_mm[:, np.newaxis, :, :]
            ensemble_mean = np.mean(raw_ensemble, axis=1, keepdims=True)
            tp_ensemble_bias = raw_forecast[:, np.newaxis, :, :] - ensemble_mean
            adjusted_residual_4d = residual_4d + tp_ensemble_bias
            corrected_members_mm = raw_ensemble + adjusted_residual_4d
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

        all_forecast.append(raw_forecast)
        all_corrected.append(corrected_mm)
        if raw_truth is not None:
            all_truth.append(raw_truth)
        if raw_ensemble is not None:
            all_raw_members.append(raw_ensemble)
        all_corrected_members.append(corrected_members_mm)

        sample_count += 1

    # 堆叠数组
    g_forecast = np.stack(all_forecast, axis=0).astype(np.float32)
    g_corrected = np.stack(all_corrected, axis=0).astype(np.float32)
    g_truth = np.stack(all_truth, axis=0).astype(np.float32) if all_truth else None
    g_raw_members = np.stack(all_raw_members, axis=0).astype(np.float32) if all_raw_members else None
    g_corrected_members = np.stack(all_corrected_members, axis=0).astype(np.float32) if all_corrected_members else None

    print(f"\n[保存] 全球数据统计:")
    print(f"  forecast: {g_forecast.shape}, min={g_forecast.min():.4f}, max={g_forecast.max():.4f}")
    print(f"  corrected: {g_corrected.shape}, min={g_corrected.min():.4f}, max={g_corrected.max():.4f}")
    if g_truth is not None:
        print(f"  truth: {g_truth.shape}, min={g_truth.min():.4f}, max={g_truth.max():.4f}")
    if g_raw_members is not None:
        print(f"  raw_members: {g_raw_members.shape}")
    if g_corrected_members is not None:
        print(f"  corrected_members: {g_corrected_members.shape}")

    # 获取经纬度（全球）
    lats = val_dataset.forecast_lats_region if hasattr(val_dataset, 'forecast_lats_region') else val_dataset.forecast_lats
    lons = val_dataset.forecast_lons_region if hasattr(val_dataset, 'forecast_lons_region') else val_dataset.forecast_lons
    print(f"  lats: [{lats.min():.1f}, {lats.max():.1f}]")
    print(f"  lons: [{lons.min():.1f}, {lons.max():.1f}]")

    # 保存为NC文件（全球专用）
    output_nc = os.path.join(global_output_dir, "global_corrected.nc")
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
            g_raw_members_t = np.transpose(g_raw_members, (0, 1, 3, 4, 2))
            raw_members_var[:] = g_raw_members_t
        if g_corrected_members is not None:
            g_corrected_members_t = np.transpose(g_corrected_members, (0, 1, 3, 4, 2))
            corrected_members_var[:] = g_corrected_members_t

        # 保存陆地掩码
        if val_dataset.land_mask is not None:
            lm = val_dataset.land_mask.astype(np.float32)
            target_h, target_w = len(lats), len(lons)
            if lm.shape[0] != target_h or lm.shape[1] != target_w:
                from scipy.ndimage import zoom
                lat_ratio = target_h / lm.shape[0]
                lon_ratio = target_w / lm.shape[1]
                lm = zoom(lm, (lat_ratio, lon_ratio), order=0) > 0.5
                lm = lm.astype(np.float32)
                print(f"[陆地掩码] 已从 {val_dataset.land_mask.shape} 插值到 {lm.shape}")
            mask_var = f.createVariable('land_mask', 'f4', ('lat', 'lon'))
            mask_var[:] = lm

    print(f"✅ 完成！全球NC文件已保存到: {output_nc}")

    # 恢复原始配置
    config.data_config["region"] = ORIGINAL_REGION
    config.data_config["lat_range"] = ORIGINAL_LAT_RANGE
    config.data_config["lon_range"] = ORIGINAL_LON_RANGE
    print(f"[配置] 已恢复为中国区域配置")


if __name__ == "__main__":
    main()
