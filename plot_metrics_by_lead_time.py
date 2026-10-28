import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
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
    """加载NC文件，支持新格式(time, lead_time, lat, lon)"""
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
        # NC存储: (time, lead_time, lat, lon, member) → (time, lead_time, 10, lat, lon)
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
        # 对齐land_mask — 与test目录一致：使用scipy zoom order=0
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
    """CSI at given threshold — 与test/plot_results.py完全一致：使用 >= """
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
    """FSS — 与test/plot_results.py完全一致：使用 >= """
    f_bin = (pred_2d >= threshold).astype(np.float64)
    o_bin = (obs_2d >= threshold).astype(np.float64)
    Pf = uniform_filter(f_bin, size=window_size, mode='constant')
    Po = uniform_filter(o_bin, size=window_size, mode='constant')
    num = np.sum((Pf - Po) ** 2)
    den = np.sum(Pf ** 2) + np.sum(Po ** 2) + 1e-8
    return float(1.0 - num / den)


def compute_tp_from_contingency(obs, pred, threshold, land_mask=None):
    """TP (Threat Performance) score based on contingency table.
    
    与test/plot_results.py完全一致。
    
    Contingency table:
      NA: forecast>=thr AND obs>=thr  (hit)
      NB: forecast<thr AND obs>=thr  (miss)
      NC: forecast>=thr AND obs<thr  (false alarm)
      ND: forecast<thr AND obs<thr   (correct negative)
    
    TS_j = NA / (NA + NB + NC)
    PC_j = (NA + ND) / (NA + NB + NC + ND)
    """
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


def compute_metrics_per_member(truth, forecast, corrected, lats, lons,
                               forecast_members=None, corrected_members=None,
                               lat_range=(15, 55), lon_range=(70, 140),
                               land_mask=None):
    """逐成员计算所有指标（与plot_metrics_vs_leadtime.py一致）

    Args:
        truth: (n_samples, T, H, W)
        forecast: (n_samples, T, H, W) — ensemble mean (unused for per-member; members used instead)
        corrected: (n_samples, T, H, W)
        forecast_members: (n_samples, T, M, H, W) — M=10 members
        corrected_members: (n_samples, T, M, H, W)
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

    # 裁剪到中国区域
    if lats is not None and lons is not None:
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

    # 陆地掩码
    if land_mask is not None:
        land = land_mask.astype(bool)
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

            # 红线(Raw Forecast)：始终使用forecast(即tp确定性预报)，与test目录一致
            raw_day = forecast[si, t][np.newaxis, :, :]  # (1, H, W) - tp确定性预报

            if corrected is not None:
                corr_day = corrected[si, t][np.newaxis, :, :]  # (1, H, W) - 订正后tp预报
            else:
                corr_day = raw_day

            # 对齐成员数：确保raw_day和corr_day成员数一致
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

                # TP @ 1mm threshold (per member)
                ts_r, pc_r = compute_tp_from_contingency(obs_t, raw_t, TP_THRESHOLD, land)
                ts_c, pc_c = compute_tp_from_contingency(obs_t, corr_t, TP_THRESHOLD, land)
                sum_ts_raw[t] += ts_r
                sum_ts_corr[t] += ts_c
                sum_pc_raw[t] += pc_r
                sum_pc_corr[t] += pc_c

                # CSI @ multiple thresholds
                for th in CSI_THRESHOLDS:
                    sum_csi_f[th][t] += compute_csi(obs_land, raw_land, th)
                    sum_csi_c[th][t] += compute_csi(obs_land, corr_land, th)

                # FSS (2D field, ocean set to 0)
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

    # TP: 先对每个成员求平均，再按test/plot_results.py公式聚合
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
    """绘制 Stage 1 三方案的中国区域指标图。
    
    Args:
        metrics: 当前模型指标
        output_path: 输出路径
        title_prefix: 标题前缀
        n_samples: 样本数
        reference_metrics: 可选多变量模型指标，绘制为橙线
        reference_metrics2: (已弃用)保留用于接口兼容
    """
    lead_times = metrics['lead_times']
    T = len(lead_times)

    # 图用于论文正文：不设置整图总标题，保留各指标子图标题。
    fig, axes = plt.subplots(2, 4, figsize=(20, 9))

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
    ORANGE = '#E07A24'

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
        # 红线：未经订正的 ECMWF 确定性预报。
        ax.plot(lead_times, raw_data, color=RED, linewidth=2,
                label='ECMWF Forecast', marker='o', markersize=2, linestyle='-')
        # 蓝线：只使用 TP 输入的 Stage 1 订正。
        ax.plot(lead_times, corr_data, color=BLUE, linewidth=2,
                label='Corrected (TP-only input)', marker='s', markersize=2, linestyle='-')

        multivar_data = None
        if reference_metrics is not None:
            multivar_data = {
                "RMSE": reference_metrics['rmse_c'],
                "MAE": reference_metrics['mae_c'],
                "ACC": reference_metrics['acc_c'],
                "TP": reference_metrics['tp_c'],
                f"CSI@{CSI_THRESHOLDS[0]}": reference_metrics['csi_c'][CSI_THRESHOLDS[0]],
                f"CSI@{CSI_THRESHOLDS[1]}": reference_metrics['csi_c'][CSI_THRESHOLDS[1]],
                f"FSS (w={FSS_WINDOWS[0]})": reference_metrics['fss_c'][FSS_WINDOWS[0]],
                f"FSS (w={FSS_WINDOWS[1]})": reference_metrics['fss_c'][FSS_WINDOWS[1]],
            }[title]
            ax.plot(lead_times, multivar_data, color=ORANGE, linewidth=2,
                    label='Corrected (Full-atmospheric inputs)', marker='^', markersize=3, linestyle='-')

        # 与 ECMWF 相比的增益：若存在全气象变量方案，则逐时效选蓝/橙中
        # 指标更优者作为阴影边界；否则使用蓝线。误差指标越低越好，其余越高越好。
        if multivar_data is None:
            best_corrected = corr_data
        elif higher_is_better:
            best_corrected = np.maximum(corr_data, multivar_data)
        else:
            best_corrected = np.minimum(corr_data, multivar_data)

        if higher_is_better:
            ax.fill_between(lead_times, raw_data, best_corrected,
                            where=best_corrected > raw_data, alpha=0.16, color='#69A85A',
                            label='Improvement')
        else:
            ax.fill_between(lead_times, best_corrected, raw_data,
                            where=best_corrected < raw_data, alpha=0.16, color='#69A85A',
                            label='Improvement')

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel("Lead Time (days)", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_xlim(lead_times[0], lead_times[-1])
        if ylim is not None:
            ax.set_ylim(ylim)
        ax.legend(fontsize=9, loc='best')
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=10)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    pdf_path = os.path.splitext(output_path)[0] + '.pdf'
    fig.savefig(pdf_path, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved plot: {output_path}")
    print(f"Saved PDF: {pdf_path}")


def save_csv(metrics, output_path, n_samples):
    """保存CSV"""
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


def main():
    parser = argparse.ArgumentParser(description='指标随预报时效变化曲线图')
    parser.add_argument('nc_file', type=str, help='nc文件路径')
    parser.add_argument('--reference_nc', type=str, default=None,
                        help='参考模型的NC文件路径（可选，用于在图上添加参考模型的订正线对比）')
    parser.add_argument('--reference_nc2', type=str, default=None,
                        help='第二参考模型的NC文件路径（可选，用于添加第四条线对比，如FSS模型）')
    parser.add_argument('--max_days', type=int, default=0, help='最大预报天数（0表示使用数据中的全部天数）')
    parser.add_argument('--start_day', type=int, default=1, help='起始预报天数（1表示从Day1开始，11表示从Day11开始）')
    parser.add_argument('--lat_min', type=float, default=15.0)
    parser.add_argument('--lat_max', type=float, default=55.0)
    parser.add_argument('--lon_min', type=float, default=70.0)
    parser.add_argument('--lon_max', type=float, default=140.0)
    parser.add_argument('--china_output', type=str, default=None,
                        help='中国区域图输出路径')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='统一输出目录（PNG+CSV全部放这里，目录名=plots时使用这个）')
    parser.add_argument('--china_region', action='store_true', default=False,
                        help='显式标记为中国区域计算，强制使用 lat=(15,55), lon=(70,140)')
    args = parser.parse_args()

    # --china_region 覆盖 lat/lon 为中国区域
    if args.china_region:
        args.lat_min, args.lat_max = 15.0, 55.0
        args.lon_min, args.lon_max = 70.0, 140.0

    # 加载数据
    truth, forecast, corrected, lats, lons, land_mask, sample_dates, forecast_members, corrected_members = \
        load_nc_data(args.nc_file)

    if truth is None or forecast is None:
        print("错误：无法加载truth或forecast数据！")
        sys.exit(1)

    if corrected is None:
        print("警告：corrected=None，使用forecast代替")
        corrected = forecast.copy()

    max_days = args.max_days

    # 截取天数（max_days=0表示使用全部天数）
    if max_days > 0:
        if truth.ndim == 4:
            if truth.shape[1] > max_days:
                truth = truth[:, :max_days]
            if forecast.shape[1] > max_days:
                forecast = forecast[:, :max_days]
            if corrected is not None and corrected.shape[1] > max_days:
                corrected = corrected[:, :max_days]
            if forecast_members is not None and forecast_members.shape[1] > max_days:
                forecast_members = forecast_members[:, :max_days]
            if corrected_members is not None and corrected_members.shape[1] > max_days:
                corrected_members = corrected_members[:, :max_days]
        else:
            if truth.shape[0] > max_days:
                truth = truth[:max_days]

    # 起始天数截取（start_day=1表示从Day1开始，start_day=11表示从Day11开始）
    start_idx = args.start_day - 1
    if start_idx > 0:
        if truth.ndim == 4:
            truth = truth[:, start_idx:]
            forecast = forecast[:, start_idx:]
            if corrected is not None:
                corrected = corrected[:, start_idx:]
            if forecast_members is not None:
                forecast_members = forecast_members[:, start_idx:]
            if corrected_members is not None:
                corrected_members = corrected_members[:, start_idx:]
        else:
            truth = truth[start_idx:]
            forecast = forecast[start_idx:]
            if corrected is not None:
                corrected = corrected[start_idx:]

    if truth.ndim == 4:
        n_samples = truth.shape[0]
        T = truth.shape[1]
    else:
        n_samples = 1
        T = truth.shape[0]

    print(f"\n=== 数据概览 ===")
    print(f"  truth: {truth.shape}")
    print(f"  forecast: {forecast.shape}")
    print(f"  corrected: {corrected.shape}")
    print(f"  forecast_members: {forecast_members.shape if forecast_members is not None else None}")
    print(f"  lats: [{lats.min():.2f}, {lats.max():.2f}], lons: [{lons.min():.2f}, {lons.max():.2f}]")
    print(f"  n_samples={n_samples}, T={T}")

    # 计算区域指标
    region_name = "China Region" if args.china_region else "Custom Region"
    print(f"\n=== 计算{region_name}指标（逐成员） ===")
    china_metrics = compute_metrics_per_member(
        truth, forecast, corrected, lats, lons,
        forecast_members=forecast_members,
        corrected_members=corrected_members,
        lat_range=(args.lat_min, args.lat_max),
        lon_range=(args.lon_min, args.lon_max),
        land_mask=land_mask,
    )

    # 计算参考模型指标（用于第三条约对比线）
    reference_metrics = None
    if args.reference_nc is not None:
        print(f"\n=== 加载参考模型NC: {args.reference_nc} ===")
        ref_truth, ref_forecast, ref_corrected, ref_lats, ref_lons, ref_land_mask, ref_dates, ref_fmembers, ref_cmembers = \
            load_nc_data(args.reference_nc)

        if ref_truth is not None and ref_forecast is not None:
            if ref_corrected is None:
                ref_corrected = ref_forecast.copy()

            # 同步截取天数
            ref_max_days = args.max_days
            if ref_max_days > 0:
                if ref_truth.ndim == 4 and ref_truth.shape[1] > ref_max_days:
                    ref_truth = ref_truth[:, :ref_max_days]
                if ref_forecast.shape[1] > ref_max_days:
                    ref_forecast = ref_forecast[:, :ref_max_days]
                if ref_corrected is not None and ref_corrected.shape[1] > ref_max_days:
                    ref_corrected = ref_corrected[:, :ref_max_days]
                if ref_fmembers is not None and ref_fmembers.shape[1] > ref_max_days:
                    ref_fmembers = ref_fmembers[:, :ref_max_days]
                if ref_cmembers is not None and ref_cmembers.shape[1] > ref_max_days:
                    ref_cmembers = ref_cmembers[:, :ref_max_days]

            ref_start_idx = args.start_day - 1
            if ref_start_idx > 0:
                ref_truth = ref_truth[:, ref_start_idx:]
                ref_forecast = ref_forecast[:, ref_start_idx:]
                if ref_corrected is not None:
                    ref_corrected = ref_corrected[:, ref_start_idx:]
                if ref_fmembers is not None:
                    ref_fmembers = ref_fmembers[:, ref_start_idx:]
                if ref_cmembers is not None:
                    ref_cmembers = ref_cmembers[:, ref_start_idx:]

            print(f"[参考模型] 数据shape: truth={ref_truth.shape}, corrected={ref_corrected.shape}")
            if sample_dates is not None and ref_dates is not None and list(sample_dates) != list(ref_dates):
                raise ValueError("参考NC与主NC的起报日期不一致，禁止生成不公平的多变量对比图")
            if ref_truth.shape != truth.shape or ref_forecast.shape != forecast.shape:
                raise ValueError("参考NC与主NC的数据维度不一致，禁止生成不公平的多变量对比图")
            if not (np.allclose(ref_lats, lats) and np.allclose(ref_lons, lons)):
                raise ValueError("参考NC与主NC的网格不一致，禁止生成不公平的多变量对比图")

            reference_metrics = compute_metrics_per_member(
                ref_truth, ref_forecast, ref_corrected, ref_lats, ref_lons,
                forecast_members=ref_fmembers,
                corrected_members=ref_cmembers,
                lat_range=(args.lat_min, args.lat_max),
                lon_range=(args.lon_min, args.lon_max),
                land_mask=ref_land_mask,
            )

            # 修正参考模型的lead_times
            if args.start_day > 1:
                n_lt_ref = len(reference_metrics['lead_times'])
                reference_metrics['lead_times'] = np.arange(args.start_day, args.start_day + n_lt_ref)

            print(f"[参考模型] 指标计算完成: TP_c={reference_metrics['tp_c'].mean():.4f}")

    # 计算第二参考模型指标（FSS模型，用于第四条橙色对比线）
    reference_metrics2 = None
    if args.reference_nc2 is not None:
        print(f"\n=== 加载第二参考模型NC: {args.reference_nc2} ===")
        ref2_truth, ref2_forecast, ref2_corrected, ref2_lats, ref2_lons, ref2_land_mask, ref2_dates, ref2_fmembers, ref2_cmembers = \
            load_nc_data(args.reference_nc2)

        if ref2_truth is not None and ref2_forecast is not None:
            if ref2_corrected is None:
                ref2_corrected = ref2_forecast.copy()

            # 同步截取天数
            ref2_max_days = args.max_days
            if ref2_max_days > 0:
                if ref2_truth.ndim == 4 and ref2_truth.shape[1] > ref2_max_days:
                    ref2_truth = ref2_truth[:, :ref2_max_days]
                if ref2_forecast.shape[1] > ref2_max_days:
                    ref2_forecast = ref2_forecast[:, :ref2_max_days]
                if ref2_corrected is not None and ref2_corrected.shape[1] > ref2_max_days:
                    ref2_corrected = ref2_corrected[:, :ref2_max_days]
                if ref2_fmembers is not None and ref2_fmembers.shape[1] > ref2_max_days:
                    ref2_fmembers = ref2_fmembers[:, :ref2_max_days]
                if ref2_cmembers is not None and ref2_cmembers.shape[1] > ref2_max_days:
                    ref2_cmembers = ref2_cmembers[:, :ref2_max_days]

            ref2_start_idx = args.start_day - 1
            if ref2_start_idx > 0:
                ref2_truth = ref2_truth[:, ref2_start_idx:]
                ref2_forecast = ref2_forecast[:, ref2_start_idx:]
                if ref2_corrected is not None:
                    ref2_corrected = ref2_corrected[:, ref2_start_idx:]
                if ref2_fmembers is not None:
                    ref2_fmembers = ref2_fmembers[:, ref2_start_idx:]
                if ref2_cmembers is not None:
                    ref2_cmembers = ref2_cmembers[:, ref2_start_idx:]

            print(f"[第二参考模型] 数据shape: truth={ref2_truth.shape}, corrected={ref2_corrected.shape}")

            reference_metrics2 = compute_metrics_per_member(
                ref2_truth, ref2_forecast, ref2_corrected, ref2_lats, ref2_lons,
                forecast_members=ref2_fmembers,
                corrected_members=ref2_cmembers,
                lat_range=(args.lat_min, args.lat_max),
                lon_range=(args.lon_min, args.lon_max),
                land_mask=ref2_land_mask,
            )

            # 修正第二参考模型的lead_times
            if args.start_day > 1:
                n_lt_ref2 = len(reference_metrics2['lead_times'])
                reference_metrics2['lead_times'] = np.arange(args.start_day, args.start_day + n_lt_ref2)

            print(f"[第二参考模型] 指标计算完成: TP_c={reference_metrics2['tp_c'].mean():.4f}")

    # 修正lead_times，使X轴显示真实天数（如Day 11-40而非Day 1-30）
    if args.start_day > 1:
        n_lt = len(china_metrics['lead_times'])
        china_metrics['lead_times'] = np.arange(args.start_day, args.start_day + n_lt)

    # 输出路径：--output_dir 优先级最高
    if args.output_dir:
        output_dir = args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        china_plot_path = os.path.join(output_dir, 'metrics_vs_leadtime_china.png')
    elif args.china_output:
        china_plot_path = args.china_output
    else:
        nc_dir = os.path.dirname(os.path.abspath(args.nc_file))
        plots_dir = os.path.join(os.path.dirname(nc_dir), 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        china_plot_path = os.path.join(plots_dir, 'metrics_vs_leadtime_china.png')

    # 绘制图
    plot_metrics(china_metrics, china_plot_path,
                 title_prefix=region_name, n_samples=n_samples,
                 reference_metrics=reference_metrics,
                 reference_metrics2=reference_metrics2)

    # 保存CSV
    csv_dir = os.path.dirname(china_plot_path)
    save_csv(china_metrics, os.path.join(csv_dir, 'metrics_vs_leadtime_china.csv'), n_samples)

    # 保存参考模型的CSV
    if reference_metrics is not None:
        save_csv(reference_metrics, os.path.join(csv_dir, 'metrics_vs_leadtime_china_reference.csv'), n_samples)

    # 保存第二参考模型的CSV
    if reference_metrics2 is not None:
        save_csv(reference_metrics2, os.path.join(csv_dir, 'metrics_vs_leadtime_china_reference2.csv'), n_samples)

    print("\n完成。")


if __name__ == "__main__":
    main()
