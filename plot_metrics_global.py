"""
plot_metrics_global.py - 全球区域指标计算与可视化

该脚本与plot_metrics_by_lead_time.py独立，专门用于全球区域指标计算。
支持：
1. 计算全球区域指标
2. 可选：对比中国区域指标
3. 使用与中国区域完全相同的指标计算方法

运行方法：
  python plot_metrics_global.py <全球NC文件>
  python plot_metrics_global.py <全球NC文件> --china_nc <中国NC文件>
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
import sys
import netCDF4 as nc
from scipy.ndimage import uniform_filter

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'axes.unicode_minus': False,
})

CSI_THRESHOLDS = [0.1, 5.0]
FSS_WINDOWS = [3, 9]
FSS_THRESHOLD = 1.0
TP_THRESHOLD = 1.0

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


def load_nc_data(filepath):
    """加载NC文件（支持全球和中国区域）"""
    f = nc.Dataset(filepath, 'r')
    print(f"[NC] Loading: {filepath}")
    print(f"[NC] Variables: {list(f.variables.keys())}")

    truth = f.variables['truth'][:] if 'truth' in f.variables else None
    forecast = f.variables['forecast'][:] if 'forecast' in f.variables else None
    corrected = f.variables['corrected'][:] if 'corrected' in f.variables else None
    lats = f.variables['lat'][:] if 'lat' in f.variables else None
    lons = f.variables['lon'][:] if 'lon' in f.variables else None

    land_mask = None
    if 'land_mask' in f.variables:
        lm_raw = f.variables['land_mask'][:]
        land_mask = lm_raw > 0.5

    sample_dates = None
    if 'time' in f.variables:
        sample_dates = []
        for i in range(len(f.variables['time'])):
            try:
                sample_dates.append(f.variables['time'][i].decode('utf-8'))
            except Exception:
                sample_dates.append(str(i))

    # 填充NaN
    if forecast is not None:
        forecast = np.nan_to_num(forecast, nan=0.0).astype(np.float32)
    if truth is not None:
        truth = np.nan_to_num(truth, nan=0.0).astype(np.float32)
    if corrected is not None:
        corrected = np.nan_to_num(corrected, nan=0.0).astype(np.float32)

    # 读取集合成员
    forecast_members = None
    if 'forecast_members' in f.variables:
        forecast_members = f.variables['forecast_members'][:].astype(np.float32)
        if forecast_members.ndim == 5 and forecast_members.shape[-1] <= 20:
            forecast_members = np.transpose(forecast_members, (0, 1, 4, 2, 3))

    corrected_members = None
    if 'corrected_members' in f.variables:
        corrected_members = f.variables['corrected_members'][:].astype(np.float32)
        if corrected_members.ndim == 5 and corrected_members.shape[-1] <= 20:
            corrected_members = np.transpose(corrected_members, (0, 1, 4, 2, 3))

    # 维度对齐检查
    if lats is not None and forecast is not None:
        h_dim, w_dim = forecast.shape[-2], forecast.shape[-1]
        lat_len, lon_len = len(lats), len(lons)
        if h_dim == lon_len and w_dim == lat_len:
            print("[NC] lat/lon 维度与数据不匹配，自动交换")
            forecast = np.swapaxes(forecast, -2, -1)
            if truth is not None:
                truth = np.swapaxes(truth, -2, -1)
            if corrected is not None:
                corrected = np.swapaxes(corrected, -2, -1)
            if forecast_members is not None:
                forecast_members = np.swapaxes(forecast_members, -2, -1)
            if corrected_members is not None:
                corrected_members = np.swapaxes(corrected_members, -2, -1)
            lats, lons = lons, lats
        # 对齐land_mask
        if land_mask is not None:
            lm_h, lm_w = land_mask.shape[-2], land_mask.shape[-1]
            if lm_h != lat_len or lm_w != lon_len:
                from scipy.ndimage import zoom
                lat_ratio = lat_len / lm_h
                lon_ratio = lon_len / lm_w
                land_mask = zoom(land_mask.astype(np.float32), (lat_ratio, lon_ratio), order=0) > 0.5

    f.close()
    print(f"[NC] Loaded: forecast={forecast.shape}, corrected={corrected.shape if corrected is not None else None}")
    print(f"[NC] forecast_members={forecast_members.shape if forecast_members is not None else None}")
    print(f"[NC] lats=[{lats.min():.2f}, {lats.max():.2f}], lons=[{lons.min():.2f}, {lons.max():.2f}]")
    return truth, forecast, corrected, lats, lons, land_mask, sample_dates, forecast_members, corrected_members


def compute_csi(obs, pred, threshold):
    """CSI at given threshold（与中国区域完全一致）"""
    mask = np.isfinite(obs) & np.isfinite(pred)
    o = obs[mask]
    p = pred[mask]
    o_rain = o >= threshold
    p_rain = p >= threshold
    H = int(np.sum(p_rain & o_rain))
    FA = int(np.sum(p_rain & ~o_rain))
    M = int(np.sum(~p_rain & o_rain))
    return H / (H + FA + M) if (H + FA + M) > 0 else 0.0


def compute_hard_fss(pred_2d, obs_2d, threshold, window_size):
    """FSS（与中国区域完全一致）"""
    f_bin = (pred_2d >= threshold).astype(np.float64)
    o_bin = (obs_2d >= threshold).astype(np.float64)
    Pf = uniform_filter(f_bin, size=window_size, mode='constant')
    Po = uniform_filter(o_bin, size=window_size, mode='constant')
    num = np.sum((Pf - Po) ** 2)
    den = np.sum(Pf ** 2) + np.sum(Po ** 2) + 1e-8
    return float(1.0 - num / den)


def compute_tp_from_contingency(obs, pred, threshold, land_mask=None):
    """TP score（与中国区域完全一致）"""
    if land_mask is not None:
        obs = obs[land_mask]
        pred = pred[land_mask]
    mask = np.isfinite(obs) & np.isfinite(pred)
    o = obs[mask]
    p = pred[mask]

    NA = int(np.sum((p >= threshold) & (o >= threshold)))
    NB = int(np.sum((p < threshold) & (o >= threshold)))
    NC = int(np.sum((p >= threshold) & (o < threshold)))
    ND = int(np.sum((p < threshold) & (o < threshold)))

    TS = NA / (NA + NB + NC) if (NA + NB + NC) > 0 else 0.0
    PC = (NA + ND) / (NA + NB + NC + ND) if (NA + NB + NC + ND) > 0 else 0.0

    return TS, PC


def compute_metrics(truth, forecast, corrected, lats, lons,
                    forecast_members=None, corrected_members=None,
                    lat_range=None, lon_range=None,
                    land_mask=None, use_land_only=True):
    """
    计算指标（与中国区域完全一致的计算方法）

    Args:
        lat_range: (lat_min, lat_max) 如果为None，使用全球
        lon_range: (lon_min, lon_max) 如果为None，使用全球
        use_land_only: 是否只计算陆地区域（中国区域使用True，全球可选择）
    """
    if truth.ndim == 3:
        truth = truth[np.newaxis, :, :, :]
        forecast = forecast[np.newaxis, :, :, :]
        if corrected is not None:
            corrected = corrected[np.newaxis, :, :, :]
        if forecast_members is not None and forecast_members.ndim == 4:
            forecast_members = forecast_members[np.newaxis, :, :, :, :]
        if corrected_members is not None and corrected_members.ndim == 4:
            corrected_members = corrected_members[np.newaxis, :, :, :, :]

    n_samples, T, H, W = truth.shape

    # 裁剪到指定区域（如果指定了范围）
    if lat_range is not None and lon_range is not None and lats is not None and lons is not None:
        lat_mask = (lats >= lat_range[0]) & (lats <= lat_range[1])
        lon_mask = (lons >= lon_range[0]) & (lons <= lon_range[1])
        truth = truth[:, :, lat_mask, :][:, :, :, lon_mask]
        forecast = forecast[:, :, lat_mask, :][:, :, :, lon_mask]
        if corrected is not None:
            corrected = corrected[:, :, lat_mask, :][:, :, :, lon_mask]
        if forecast_members is not None:
            forecast_members = forecast_members[:, :, :, lat_mask, :][:, :, :, :, lon_mask]
        if corrected_members is not None:
            corrected_members = corrected_members[:, :, :, lat_mask, :][:, :, :, :, lon_mask]
        if land_mask is not None:
            land_mask = land_mask[lat_mask, :][:, lon_mask]
        print(f"[区域裁剪] lat_range={lat_range}, lon_range={lon_range}")

    # 陆地掩码
    if use_land_only and land_mask is not None:
        land = land_mask.astype(bool)
    else:
        # 使用全部有效像素（排除NaN）
        if truth is not None:
            land = np.ones((H, W), dtype=bool)
            # 标记有数据的像素
            valid = np.isfinite(truth[0, 0])
            land = land & valid
        else:
            land = np.ones((H, W), dtype=bool)

    n_members = forecast_members.shape[2] if forecast_members is not None else 10

    print(f"\n[指标计算] 样本数={n_samples}, 时效={T}天, 成员数={n_members}")
    print(f"[指标计算] 数据shape: truth={truth.shape}, members={forecast_members.shape if forecast_members is not None else None}")
    print(f"[指标计算] 陆地占比: {np.mean(land):.2%}")

    # 逐成员累加器
    sum_rmse_f = np.zeros(T)
    sum_rmse_c = np.zeros(T)
    sum_mae_f = np.zeros(T)
    sum_mae_c = np.zeros(T)
    sum_acc_f = np.zeros(T)
    sum_acc_c = np.zeros(T)
    sum_ts_raw = np.zeros(T)
    sum_ts_corr = np.zeros(T)
    sum_pc_raw = np.zeros(T)
    sum_pc_corr = np.zeros(T)
    sum_csi_f = {th: np.zeros(T) for th in CSI_THRESHOLDS}
    sum_csi_c = {th: np.zeros(T) for th in CSI_THRESHOLDS}
    sum_fss_f = {ws: np.zeros(T) for ws in FSS_WINDOWS}
    sum_fss_c = {ws: np.zeros(T) for ws in FSS_WINDOWS}
    count = np.zeros(T, dtype=int)

    for si in range(n_samples):
        for t in range(T):
            obs_t = truth[si, t]
            obs_land = obs_t[land]
            o_anom = obs_land - obs_land.mean()
            obs_2d = obs_t.copy()
            obs_2d[~land] = 0.0

            # 红线(Raw Forecast)
            raw_day = forecast[si, t][np.newaxis, :, :]

            if corrected is not None:
                corr_day = corrected[si, t][np.newaxis, :, :]
            else:
                corr_day = raw_day

            # 对齐成员数
            M_raw = raw_day.shape[0]
            M_corr = corr_day.shape[0]
            if M_raw == 1 and M_corr > 1:
                raw_day = np.repeat(raw_day, M_corr, axis=0)
                M = M_corr
            elif M_corr == 1 and M_raw > 1:
                corr_day = np.repeat(corr_day, M_raw, axis=0)
                M = M_raw
            else:
                M = max(M_raw, M_corr)

            for mi in range(M):
                raw_t = raw_day[mi]
                corr_t = corr_day[mi]

                raw_land = raw_t[land]
                corr_land = corr_t[land]

                # RMSE
                sum_rmse_f[t] += np.sqrt(np.mean((raw_land - obs_land) ** 2))
                sum_rmse_c[t] += np.sqrt(np.mean((corr_land - obs_land) ** 2))

                # MAE
                sum_mae_f[t] += np.mean(np.abs(raw_land - obs_land))
                sum_mae_c[t] += np.mean(np.abs(corr_land - obs_land))

                # ACC
                f_anom = raw_land - raw_land.mean()
                denom = np.sqrt(max(np.sum(f_anom ** 2) * np.sum(o_anom ** 2), 1e-10)) + 1e-10
                sum_acc_f[t] += np.sum(f_anom * o_anom) / denom

                p_anom = corr_land - corr_land.mean()
                denom2 = np.sqrt(max(np.sum(p_anom ** 2) * np.sum(o_anom ** 2), 1e-10)) + 1e-10
                sum_acc_c[t] += np.sum(p_anom * o_anom) / denom2

                # TP
                ts_r, pc_r = compute_tp_from_contingency(obs_t, raw_t, TP_THRESHOLD, land)
                ts_c, pc_c = compute_tp_from_contingency(obs_t, corr_t, TP_THRESHOLD, land)
                sum_ts_raw[t] += ts_r
                sum_ts_corr[t] += ts_c
                sum_pc_raw[t] += pc_r
                sum_pc_corr[t] += pc_c

                # CSI
                for th in CSI_THRESHOLDS:
                    sum_csi_f[th][t] += compute_csi(obs_land, raw_land, th)
                    sum_csi_c[th][t] += compute_csi(obs_land, corr_land, th)

                # FSS
                raw_2d = raw_t.copy()
                raw_2d[~land] = 0.0
                corr_2d = corr_t.copy()
                corr_2d[~land] = 0.0
                for ws in FSS_WINDOWS:
                    sum_fss_f[ws][t] += compute_hard_fss(raw_2d, obs_2d, FSS_THRESHOLD, ws)
                    sum_fss_c[ws][t] += compute_hard_fss(corr_2d, obs_2d, FSS_THRESHOLD, ws)

            count[t] += M

        if (si + 1) % 5 == 0 or si == n_samples - 1:
            print(f"  Progress: {si + 1}/{n_samples}")

    # 求平均
    lead_times = np.arange(1, T + 1)
    rmse_f = sum_rmse_f / count
    rmse_c = sum_rmse_c / count
    mae_f = sum_mae_f / count
    mae_c = sum_mae_c / count
    acc_f = sum_acc_f / count
    acc_c = sum_acc_c / count
    csi_f = {th: sum_csi_f[th] / count for th in CSI_THRESHOLDS}
    csi_c = {th: sum_csi_c[th] / count for th in CSI_THRESHOLDS}
    fss_f = {ws: sum_fss_f[ws] / count for ws in FSS_WINDOWS}
    fss_c = {ws: sum_fss_c[ws] / count for ws in FSS_WINDOWS}

    # TP
    ts_raw_avg = sum_ts_raw / count
    ts_corr_avg = sum_ts_corr / count
    pc_raw_avg = sum_pc_raw / count
    pc_corr_avg = sum_pc_corr / count

    ts_S_raw = np.mean(ts_raw_avg)
    ts_S_corr = np.mean(ts_corr_avg)

    ts_syn_raw = (ts_raw_avg + ts_S_raw) / (1.0 + ts_S_raw)
    ts_syn_corr = (ts_corr_avg + ts_S_corr) / (1.0 + ts_S_corr)

    tp_f = (ts_syn_raw + pc_raw_avg) / 2.0
    tp_c = (ts_syn_corr + pc_corr_avg) / 2.0

    print("[指标计算] 完成")

    return {
        'rmse_f': rmse_f, 'rmse_c': rmse_c,
        'mae_f': mae_f, 'mae_c': mae_c,
        'acc_f': acc_f, 'acc_c': acc_c,
        'tp_f': tp_f, 'tp_c': tp_c,
        'csi_f': csi_f, 'csi_c': csi_c,
        'fss_f': fss_f, 'fss_c': fss_c,
        'lead_times': lead_times,
        'n_members_total': count,
    }


def plot_metrics(metrics, output_path, title_prefix="", n_samples=1, reference_metrics=None, reference_metrics2=None):
    """绘制2×4网格指标对比图（只保留红线Raw+蓝线Corrected+绿色Improvement区域）
    
    Args:
        metrics: 当前模型指标
        output_path: 输出路径
        title_prefix: 标题前缀
        n_samples: 样本数
        reference_metrics: (已弃用)保留用于接口兼容
        reference_metrics2: (已弃用)保留用于接口兼容
    """
    lead_times = metrics['lead_times']
    T = len(lead_times)

    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    fig.suptitle(
        f"Metrics vs Lead Time{(' (' + title_prefix + ')') if title_prefix else ''} "
        f"({n_samples} init dates, per-member avg)",
        fontsize=16, fontweight='bold'
    )

    rmse_f = metrics['rmse_f']
    rmse_c = metrics['rmse_c']
    mae_f = metrics['mae_f']
    mae_c = metrics['mae_c']
    acc_f = metrics['acc_f']
    acc_c = metrics['acc_c']
    tp_f = metrics['tp_f']
    tp_c = metrics['tp_c']
    csi_f = metrics['csi_f']
    csi_c = metrics['csi_c']
    fss_f = metrics['fss_f']
    fss_c = metrics['fss_c']

    RED = '#D03030'
    BLUE = '#2060C0'

    panels = [
        (axes[0, 0], "RMSE", "RMSE (mm/day)", rmse_f, rmse_c, None, False),
        (axes[0, 1], "MAE", "MAE (mm/day)", mae_f, mae_c, None, False),
        (axes[0, 2], "ACC", "ACC", acc_f, acc_c, None, True),
        (axes[0, 3], "TP", "TP", tp_f, tp_c, None, True),
        (axes[1, 0], f"CSI@{CSI_THRESHOLDS[0]}", "CSI", csi_f[CSI_THRESHOLDS[0]], csi_c[CSI_THRESHOLDS[0]], None, True),
        (axes[1, 1], f"CSI@{CSI_THRESHOLDS[1]}", "CSI", csi_f[CSI_THRESHOLDS[1]], csi_c[CSI_THRESHOLDS[1]], None, True),
        (axes[1, 2], f"FSS (w={FSS_WINDOWS[0]})", "FSS", fss_f[FSS_WINDOWS[0]], fss_c[FSS_WINDOWS[0]], None, True),
        (axes[1, 3], f"FSS (w={FSS_WINDOWS[1]})", "FSS", fss_f[FSS_WINDOWS[1]], fss_c[FSS_WINDOWS[1]], None, True),
    ]

    for ax, title, ylabel, raw_data, corr_data, ylim, higher_is_better in panels:
        # 红线: Raw Forecast (始终绘制)
        ax.plot(lead_times, raw_data, color=RED, linewidth=2,
                label='Raw Forecast', marker='o', markersize=2, linestyle='-')
        # 蓝线: TP Loss模型订正（实线）
        ax.plot(lead_times, corr_data, color=BLUE, linewidth=2,
                label='Corrected (TP Loss)', marker='s', markersize=2, linestyle='-')

        # Improvement region: 蓝线相比红线进步的区域（绿色填充保留）
        if higher_is_better:
            ax.fill_between(lead_times, raw_data, corr_data,
                            where=corr_data > raw_data, alpha=0.15, color='green',
                            label='Improvement (TP Loss)')
        else:
            ax.fill_between(lead_times, corr_data, raw_data,
                            where=corr_data < raw_data, alpha=0.15, color='green',
                            label='Improvement (TP Loss)')

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel("Lead Time (days)", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_xlim(lead_times[0], lead_times[-1])
        if ylim is not None:
            ax.set_ylim(ylim)
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved plot: {output_path}")


def save_csv(metrics, output_path, n_samples, region_name=""):
    """保存CSV并打印ALL平均"""
    import pandas as pd
    lead_times = metrics['lead_times']
    rows = []
    for t in range(len(lead_times)):
        row = {
            "lead_time": int(lead_times[t]),
            "rmse_f": round(float(metrics['rmse_f'][t]), 4),
            "rmse_c": round(float(metrics['rmse_c'][t]), 4),
            "mae_f": round(float(metrics['mae_f'][t]), 4),
            "mae_c": round(float(metrics['mae_c'][t]), 4),
            "acc_f": round(float(metrics['acc_f'][t]), 4),
            "acc_c": round(float(metrics['acc_c'][t]), 4),
            "tp_f": round(float(metrics['tp_f'][t]), 4),
            "tp_c": round(float(metrics['tp_c'][t]), 4),
        }
        for th in CSI_THRESHOLDS:
            row[f"csi_{th}_f"] = round(float(metrics['csi_f'][th][t]), 4)
            row[f"csi_{th}_c"] = round(float(metrics['csi_c'][th][t]), 4)
        for ws in FSS_WINDOWS:
            row[f"fss_w{ws}_f"] = round(float(metrics['fss_f'][ws][t]), 4)
            row[f"fss_w{ws}_c"] = round(float(metrics['fss_c'][ws][t]), 4)
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"Saved CSV: {output_path}")

    # 打印ALL平均
    print(f"\n{'='*60}")
    if region_name:
        print(f"  {region_name} - ALL Average (across all lead times, {n_samples} init dates)")
    else:
        print(f"  ALL Average (across all lead times, {n_samples} init dates)")
    print(f"{'='*60}")
    rmse_f = metrics['rmse_f']
    rmse_c = metrics['rmse_c']
    mae_f = metrics['mae_f']
    mae_c = metrics['mae_c']
    acc_f = metrics['acc_f']
    acc_c = metrics['acc_c']
    tp_f = metrics['tp_f']
    tp_c = metrics['tp_c']
    csi_f = metrics['csi_f']
    csi_c = metrics['csi_c']
    fss_f = metrics['fss_f']
    fss_c = metrics['fss_c']

    print(f"  RMSE:   raw={rmse_f.mean():.4f}  corr={rmse_c.mean():.4f}  "
          f"imp={(rmse_f.mean()-rmse_c.mean())/rmse_f.mean()*100:+.2f}%")
    print(f"  MAE:    raw={mae_f.mean():.4f}  corr={mae_c.mean():.4f}  "
          f"imp={(mae_f.mean()-mae_c.mean())/max(mae_f.mean(),1e-6)*100:+.2f}%")
    print(f"  ACC:    raw={acc_f.mean():.4f}  corr={acc_c.mean():.4f}  "
          f"imp={(acc_c.mean()-acc_f.mean())/max(abs(acc_f.mean()),1e-6)*100:+.2f}%")
    print(f"  TP:     raw={tp_f.mean():.4f}  corr={tp_c.mean():.4f}  "
          f"imp={(tp_c.mean()-tp_f.mean())/max(tp_f.mean(),1e-6)*100:+.2f}%")
    for th in CSI_THRESHOLDS:
        print(f"  CSI@{th}: raw={csi_f[th].mean():.4f}  corr={csi_c[th].mean():.4f}  "
              f"imp={(csi_c[th].mean()-csi_f[th].mean())/max(csi_f[th].mean(),1e-6)*100:+.2f}%")
    for ws in FSS_WINDOWS:
        print(f"  FSS_w{ws}: raw={fss_f[ws].mean():.4f}  corr={fss_c[ws].mean():.4f}  "
              f"imp={(fss_c[ws].mean()-fss_f[ws].mean())/max(fss_f[ws].mean(),1e-6)*100:+.2f}%")


def compare_metrics(china_metrics, global_metrics, output_path):
    """对比中国和全球区域指标"""
    print(f"\n{'='*80}")
    print("  中国区域 vs 全球区域 指标对比")
    print(f"{'='*80}")
    print(f"  {'指标':<15} {'中国区域':<20} {'全球区域':<20} {'差异':<15}")
    print(f"  {'-'*70}")

    metrics_names = [
        ('RMSE', 'rmse_f', 'rmse_c', False),
        ('MAE', 'mae_f', 'mae_c', False),
        ('ACC', 'acc_f', 'acc_c', True),
        ('TP', 'tp_f', 'tp_c', True),
    ]

    for name, f_key, c_key, higher_is_better in metrics_names:
        china_raw = np.mean(china_metrics[f_key])
        china_corr = np.mean(china_metrics[c_key])
        global_raw = np.mean(global_metrics[f_key])
        global_corr = np.mean(global_metrics[c_key])

        china_imp = (china_corr - china_raw) / china_raw * 100 if not higher_is_better else (china_corr - china_raw) / abs(china_raw) * 100
        global_imp = (global_corr - global_raw) / global_raw * 100 if not higher_is_better else (global_corr - global_raw) / abs(global_raw) * 100

        print(f"  {name:<15}")
        print(f"    原始:  CHN={china_raw:.4f}  GBL={global_raw:.4f}  diff={global_raw-china_raw:+.4f}")
        print(f"    订正:  CHN={china_corr:.4f}  GBL={global_corr:.4f}  diff={global_corr-china_corr:+.4f}")
        print(f"    改进率: CHN={china_imp:+.2f}%  GBL={global_imp:+.2f}%")
        print()

    # CSI对比
    for th in CSI_THRESHOLDS:
        china_csi = np.mean(china_metrics['csi_c'][th])
        global_csi = np.mean(global_metrics['csi_c'][th])
        china_imp_csi = (china_csi - np.mean(china_metrics['csi_f'][th])) / max(np.mean(china_metrics['csi_f'][th]), 1e-6) * 100
        global_imp_csi = (global_csi - np.mean(global_metrics['csi_f'][th])) / max(np.mean(global_metrics['csi_f'][th]), 1e-6) * 100
        print(f"  CSI@{th}: CHN={china_csi:.4f} ({china_imp_csi:+.2f}%)  GBL={global_csi:.4f} ({global_imp_csi:+.2f}%)")

    # FSS对比
    for ws in FSS_WINDOWS:
        china_fss = np.mean(china_metrics['fss_c'][ws])
        global_fss = np.mean(global_metrics['fss_c'][ws])
        china_imp_fss = (china_fss - np.mean(china_metrics['fss_f'][ws])) / max(np.mean(china_metrics['fss_f'][ws]), 1e-6) * 100
        global_imp_fss = (global_fss - np.mean(global_metrics['fss_f'][ws])) / max(np.mean(global_metrics['fss_f'][ws]), 1e-6) * 100
        print(f"  FSS_w{ws}: CHN={china_fss:.4f} ({china_imp_fss:+.2f}%)  GBL={global_fss:.4f} ({global_imp_fss:+.2f}%)")

    print(f"{'='*80}")


def main():
    parser = argparse.ArgumentParser(description='全球区域指标计算与对比')
    parser.add_argument('global_nc', type=str, help='全球区域NC文件路径')
    parser.add_argument('--reference_nc', type=str, default=None,
                        help='参考模型的NC文件路径（可选，用于在图上添加参考模型的订正线对比）')
    parser.add_argument('--reference_nc2', type=str, default=None,
                        help='第二参考模型的NC文件路径（可选，用于添加第四条线对比，如FSS模型）')
    parser.add_argument('--china_nc', type=str, default=None,
                        help='中国区域NC文件路径（可选，用于对比）')
    parser.add_argument('--max_days', type=int, default=0, help='最大预报天数（0表示使用数据中的全部天数）')
    parser.add_argument('--start_day', type=int, default=1, help='起始预报天数（1表示从Day1开始，11表示从Day11开始）')
    parser.add_argument('--lat_min', type=float, default=-90.0, help='全球区域纬度下限')
    parser.add_argument('--lat_max', type=float, default=90.0, help='全球区域纬度上限')
    parser.add_argument('--lon_min', type=float, default=0.0, help='全球区域经度下限')
    parser.add_argument('--lon_max', type=float, default=360.0, help='全球区域经度上限')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='统一输出目录（全球PNG+CSV放这里，对比结果只打印不重复绘图）')
    parser.add_argument('--use_land_only', action='store_true', default=True,
                        help='是否只计算陆地区域')
    parser.add_argument('--all_pixels', action='store_true',
                        help='计算所有有效像素（包括海洋）')
    parser.add_argument('--skip_china_plot', action='store_true', default=True,
                        help='（默认开启）不再重复绘制中国区域图，避免与plot_metrics_by_lead_time.py生成的metrics_vs_leadtime_china.png重复')
    parser.add_argument('--force_china_plot', action='store_true', default=False,
                        help='强制生成中国区域图（默认关闭，避免重复）')
    args = parser.parse_args()

    # 加载全球数据
    print("=" * 60)
    print("加载全球区域数据")
    print("=" * 60)
    g_truth, g_forecast, g_corrected, g_lats, g_lons, g_land_mask, g_dates, g_fmembers, g_cmembers = \
        load_nc_data(args.global_nc)

    if g_truth is None or g_forecast is None:
        print("错误：无法加载全球区域的truth或forecast数据！")
        sys.exit(1)

    if g_corrected is None:
        print("警告：corrected=None，使用forecast代替")
        g_corrected = g_forecast.copy()

    max_days = args.max_days

    # 截取天数（max_days=0表示使用全部天数）
    if max_days > 0:
        if g_truth.ndim == 4 and g_truth.shape[1] > max_days:
            g_truth = g_truth[:, :max_days]
        if g_forecast.shape[1] > max_days:
            g_forecast = g_forecast[:, :max_days]
        if g_corrected is not None and g_corrected.shape[1] > max_days:
            g_corrected = g_corrected[:, :max_days]
        if g_fmembers is not None and g_fmembers.shape[1] > max_days:
            g_fmembers = g_fmembers[:, :max_days]
        if g_cmembers is not None and g_cmembers.shape[1] > max_days:
            g_cmembers = g_cmembers[:, :max_days]

    # 起始天数截取（start_day=1表示从Day1开始，start_day=11表示从Day11开始）
    start_idx = args.start_day - 1
    if start_idx > 0:
        g_truth = g_truth[:, start_idx:]
        g_forecast = g_forecast[:, start_idx:]
        if g_corrected is not None:
            g_corrected = g_corrected[:, start_idx:]
        if g_fmembers is not None:
            g_fmembers = g_fmembers[:, start_idx:]
        if g_cmembers is not None:
            g_cmembers = g_cmembers[:, start_idx:]

    n_samples = g_truth.shape[0]
    T = g_truth.shape[1]

    print(f"\n=== 全球数据概览 ===")
    print(f"  truth: {g_truth.shape}")
    print(f"  forecast: {g_forecast.shape}")
    print(f"  corrected: {g_corrected.shape}")
    print(f"  forecast_members: {g_fmembers.shape if g_fmembers is not None else None}")
    print(f"  lats: [{g_lats.min():.2f}, {g_lats.max():.2f}], lons: [{g_lons.min():.2f}, {g_lons.max():.2f}]")
    print(f"  n_samples={n_samples}, T={T}")

    # 计算全球区域指标
    print("\n" + "=" * 60)
    print("计算全球区域指标")
    print("=" * 60)
    use_land = not args.all_pixels
    global_metrics = compute_metrics(
        g_truth, g_forecast, g_corrected, g_lats, g_lons,
        forecast_members=g_fmembers,
        corrected_members=g_cmembers,
        lat_range=(args.lat_min, args.lat_max),
        lon_range=(args.lon_min, args.lon_max),
        land_mask=g_land_mask,
        use_land_only=use_land,
    )

    # 计算参考模型指标
    reference_metrics = None
    if args.reference_nc is not None:
        print(f"\n=== 加载参考模型全球NC: {args.reference_nc} ===")
        rg_truth, rg_forecast, rg_corrected, rg_lats, rg_lons, rg_land_mask, rg_dates, rg_fmembers, rg_cmembers = \
            load_nc_data(args.reference_nc)

        if rg_truth is not None and rg_forecast is not None:
            if rg_corrected is None:
                rg_corrected = rg_forecast.copy()

            # 同步截取
            if max_days > 0:
                if rg_truth.ndim == 4 and rg_truth.shape[1] > max_days:
                    rg_truth = rg_truth[:, :max_days]
                if rg_forecast.shape[1] > max_days:
                    rg_forecast = rg_forecast[:, :max_days]
                if rg_corrected is not None and rg_corrected.shape[1] > max_days:
                    rg_corrected = rg_corrected[:, :max_days]
                if rg_fmembers is not None and rg_fmembers.shape[1] > max_days:
                    rg_fmembers = rg_fmembers[:, :max_days]
                if rg_cmembers is not None and rg_cmembers.shape[1] > max_days:
                    rg_cmembers = rg_cmembers[:, :max_days]

            rg_start_idx = args.start_day - 1
            if rg_start_idx > 0:
                rg_truth = rg_truth[:, rg_start_idx:]
                rg_forecast = rg_forecast[:, rg_start_idx:]
                if rg_corrected is not None:
                    rg_corrected = rg_corrected[:, rg_start_idx:]
                if rg_fmembers is not None:
                    rg_fmembers = rg_fmembers[:, rg_start_idx:]
                if rg_cmembers is not None:
                    rg_cmembers = rg_cmembers[:, rg_start_idx:]

            reference_metrics = compute_metrics(
                rg_truth, rg_forecast, rg_corrected, rg_lats, rg_lons,
                forecast_members=rg_fmembers,
                corrected_members=rg_cmembers,
                lat_range=(args.lat_min, args.lat_max),
                lon_range=(args.lon_min, args.lon_max),
                land_mask=rg_land_mask,
                use_land_only=use_land,
            )
            print(f"[参考模型] 全球指标计算完成: TP_c={reference_metrics['tp_c'].mean():.4f}")

    # 计算第二参考模型指标（FSS模型）
    reference_metrics2 = None
    if args.reference_nc2 is not None:
        print(f"\n=== 加载第二参考模型全球NC: {args.reference_nc2} ===")
        rg2_truth, rg2_forecast, rg2_corrected, rg2_lats, rg2_lons, rg2_land_mask, rg2_dates, rg2_fmembers, rg2_cmembers = \
            load_nc_data(args.reference_nc2)

        if rg2_truth is not None and rg2_forecast is not None:
            if rg2_corrected is None:
                rg2_corrected = rg2_forecast.copy()

            # 同步截取
            if max_days > 0:
                if rg2_truth.ndim == 4 and rg2_truth.shape[1] > max_days:
                    rg2_truth = rg2_truth[:, :max_days]
                if rg2_forecast.shape[1] > max_days:
                    rg2_forecast = rg2_forecast[:, :max_days]
                if rg2_corrected is not None and rg2_corrected.shape[1] > max_days:
                    rg2_corrected = rg2_corrected[:, :max_days]
                if rg2_fmembers is not None and rg2_fmembers.shape[1] > max_days:
                    rg2_fmembers = rg2_fmembers[:, :max_days]
                if rg2_cmembers is not None and rg2_cmembers.shape[1] > max_days:
                    rg2_cmembers = rg2_cmembers[:, :max_days]

            rg2_start_idx = args.start_day - 1
            if rg2_start_idx > 0:
                rg2_truth = rg2_truth[:, rg2_start_idx:]
                rg2_forecast = rg2_forecast[:, rg2_start_idx:]
                if rg2_corrected is not None:
                    rg2_corrected = rg2_corrected[:, rg2_start_idx:]
                if rg2_fmembers is not None:
                    rg2_fmembers = rg2_fmembers[:, rg2_start_idx:]
                if rg2_cmembers is not None:
                    rg2_cmembers = rg2_cmembers[:, rg2_start_idx:]

            reference_metrics2 = compute_metrics(
                rg2_truth, rg2_forecast, rg2_corrected, rg2_lats, rg2_lons,
                forecast_members=rg2_fmembers,
                corrected_members=rg2_cmembers,
                lat_range=(args.lat_min, args.lat_max),
                lon_range=(args.lon_min, args.lon_max),
                land_mask=rg2_land_mask,
                use_land_only=use_land,
            )
            print(f"[第二参考模型] 全球指标计算完成: TP_c={reference_metrics2['tp_c'].mean():.4f}")

    # 修正lead_times，使X轴显示真实天数（如Day 11-40而非Day 1-30）
    if args.start_day > 1:
        n_lt = len(global_metrics['lead_times'])
        global_metrics['lead_times'] = np.arange(args.start_day, args.start_day + n_lt)

    # 输出目录
    if args.output_dir:
        output_dir = args.output_dir
    else:
        nc_dir = os.path.dirname(os.path.abspath(args.global_nc))
        output_dir = os.path.join(os.path.dirname(nc_dir), 'plots')
    os.makedirs(output_dir, exist_ok=True)

    # 绘制全球区域图
    global_plot_path = os.path.join(output_dir, 'metrics_vs_leadtime_global.png')
    plot_metrics(global_metrics, global_plot_path, title_prefix="Global Region", n_samples=n_samples,
                 reference_metrics=reference_metrics,
                 reference_metrics2=reference_metrics2)

    # 保存全球CSV
    global_csv_path = os.path.join(output_dir, 'metrics_vs_leadtime_global.csv')
    save_csv(global_metrics, global_csv_path, n_samples, region_name="全球区域")

    # 保存参考模型CSV
    if reference_metrics is not None:
        ref_csv_path = os.path.join(output_dir, 'metrics_vs_leadtime_global_reference.csv')
        save_csv(reference_metrics, ref_csv_path, n_samples, region_name="全球区域(参考)")

    # 保存第二参考模型CSV
    if reference_metrics2 is not None:
        ref2_csv_path = os.path.join(output_dir, 'metrics_vs_leadtime_global_reference2.csv')
        save_csv(reference_metrics2, ref2_csv_path, n_samples, region_name="全球区域(FSS模型)")

    # 如果有中国NC文件，计算中国区域指标并对比（默认不重复绘图，只打印和保存对比CSV）
    china_metrics = None
    generate_china_plot = (not args.skip_china_plot) or args.force_china_plot
    if args.china_nc:
        print("\n" + "=" * 60)
        print("计算中国区域指标（用于对比）")
        print("=" * 60)
        c_truth, c_forecast, c_corrected, c_lats, c_lons, c_land_mask, c_dates, c_fmembers, c_cmembers = \
            load_nc_data(args.china_nc)

        if c_truth is not None and c_forecast is not None:
            if c_corrected is None:
                c_corrected = c_forecast.copy()

            # 截取天数（max_days=0表示使用全部天数）
            if max_days > 0:
                if c_truth.ndim == 4 and c_truth.shape[1] > max_days:
                    c_truth = c_truth[:, :max_days]
                if c_forecast.shape[1] > max_days:
                    c_forecast = c_forecast[:, :max_days]
                if c_corrected is not None and c_corrected.shape[1] > max_days:
                    c_corrected = c_corrected[:, :max_days]
                if c_fmembers is not None and c_fmembers.shape[1] > max_days:
                    c_fmembers = c_fmembers[:, :max_days]
                if c_cmembers is not None and c_cmembers.shape[1] > max_days:
                    c_cmembers = c_cmembers[:, :max_days]

            # 起始天数截取
            c_start_idx = args.start_day - 1
            if c_start_idx > 0:
                c_truth = c_truth[:, c_start_idx:]
                c_forecast = c_forecast[:, c_start_idx:]
                if c_corrected is not None:
                    c_corrected = c_corrected[:, c_start_idx:]
                if c_fmembers is not None:
                    c_fmembers = c_fmembers[:, c_start_idx:]
                if c_cmembers is not None:
                    c_cmembers = c_cmembers[:, c_start_idx:]

            china_n = c_truth.shape[0]
            china_metrics = compute_metrics(
                c_truth, c_forecast, c_corrected, c_lats, c_lons,
                forecast_members=c_fmembers,
                corrected_members=c_cmembers,
                lat_range=(15, 55),
                lon_range=(70, 140),
                land_mask=c_land_mask,
                use_land_only=True,
            )

            # 修正中国区域lead_times
            if args.start_day > 1:
                n_lt_c = len(china_metrics['lead_times'])
                china_metrics['lead_times'] = np.arange(args.start_day, args.start_day + n_lt_c)

            # 【默认】不再生成重复的中国区域图，避免与 plot_metrics_by_lead_time.py 生成的 metrics_vs_leadtime_china.png 重复
            # 用户观察到 metrics_vs_leadtime_china_from_global.png 和 metrics_vs_leadtime_china.png 完全相同 → 确实是重复产物
            if generate_china_plot:
                china_plot_path = os.path.join(output_dir, 'metrics_vs_leadtime_china_from_global.png')
                plot_metrics(china_metrics, china_plot_path, title_prefix="China Region", n_samples=china_n)
                print(f"已生成中国区域图（强制模式）: {china_plot_path}")
            else:
                print("ℹ️ 已跳过中国区域图绘制（与plot_metrics_by_lead_time.py生成的metrics_vs_leadtime_china.png内容一致，避免重复）")
                print("ℹ️ 如需强制绘制，请加 --force_china_plot 参数")

            # 保存中国区域对比CSV（始终保存，便于二次处理）
            china_csv_path = os.path.join(output_dir, 'metrics_vs_leadtime_china_comparison.csv')
            save_csv(china_metrics, china_csv_path, china_n, region_name="中国区域(对比用)")

            # 终端打印对比（核心目标：用户最想要的是中国vs全球的指标差异，直接在日志里看）
            compare_metrics(china_metrics, global_metrics, output_dir)

    print("\n完成。")
    print(f"\n输出文件（所有结果集中在 plots/ 目录）:")
    print(f"  全球区域图:    {global_plot_path}")
    print(f"  全球区域CSV:   {global_csv_path}")
    if china_metrics is not None:
        print(f"  中国对比CSV:   {os.path.join(output_dir, 'metrics_vs_leadtime_china_comparison.csv')}")
        print(f"  （中国区域图已复用 plot_metrics_by_lead_time.py 生成的 metrics_vs_leadtime_china.png，不再重复生成）")


if __name__ == "__main__":
    main()