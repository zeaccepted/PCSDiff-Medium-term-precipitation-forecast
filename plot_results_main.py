"""
plot_results_main.py - 主工程可视化脚本（与test目录plot_results.py风格完全一致）

特性：
- 不连续降水颜色映射（小值更精细的色阶）
- 误差图使用连续发散色标 RdBu_r，色条范围固定 -4 到 4
- 生成中国区域 (15-55°N, 70-140°E) 和全球区域分开的图片
- 海陆边界线叠加显示
- 统一字体和标签

用法: python plot_results_main.py [nc_path] [--max_dates 6] [--output_dir plots]
"""

import os
import sys
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import rcParams
from matplotlib.path import Path
import netCDF4 as nc

rcParams['font.family'] = 'DejaVu Sans'
rcParams['font.size'] = 10
rcParams['axes.titlesize'] = 10
rcParams['axes.labelsize'] = 9
rcParams['xtick.labelsize'] = 8
rcParams['ytick.labelsize'] = 8
rcParams['legend.fontsize'] = 9
rcParams['figure.titlesize'] = 14
rcParams['lines.linewidth'] = 1.2

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# 全国降水业务图范围：完整覆盖中国大陆、海南、台湾及近海。
CHINA_LAT_RANGE = (15.0, 55.0)
CHINA_LON_RANGE = (70.0, 140.0)


def find_latest_nc():
    """自动查找最新的NC结果文件"""
    nc_files = sorted(glob.glob(
        os.path.join(ROOT_DIR, 'results', '*', 'global_correction_results', 'global_corrected.nc')
    ))
    if nc_files:
        return nc_files[-1]
    fallback = sorted(glob.glob(
        os.path.join(ROOT_DIR, 'results', '*', '*', 'nc_results', 'final_prediction_results.nc')
    ))
    if fallback:
        return fallback[-1]
    return None


def load_nc_file(nc_path):
    """加载NC文件"""
    print(f"Loading NC file: {nc_path}")
    with nc.Dataset(nc_path, 'r') as f:
        forecast = np.array(f.variables['forecast'][:, :, :, :], dtype=np.float32)
        corrected = np.array(f.variables['corrected'][:, :, :, :], dtype=np.float32)
        truth = np.array(f.variables['truth'][:, :, :, :], dtype=np.float32)
        lats = np.array(f.variables['lat'][:], dtype=np.float64)
        lons = np.array(f.variables['lon'][:], dtype=np.float64)
        time = f.variables['time'][:]
        lead_time = np.array(f.variables['lead_time'][:], dtype=np.int32)

        land_mask = None
        if 'land_mask' in f.variables:
            land_mask_raw = np.array(f.variables['land_mask'][:, :], dtype=np.float32)
            land_mask = land_mask_raw > 0.5

    time_str = []
    for t in time:
        try:
            time_str.append(t.decode('utf-8'))
        except (UnicodeDecodeError, AttributeError):
            time_str.append(str(t))
    return forecast, corrected, truth, lats, lons, time_str, lead_time, land_mask


def create_china_land_mask(lats, lons):
    """基于固定地理多边形创建中国区域陆地掩码（不依赖数据NaN）"""
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    points = np.column_stack((lon_grid.ravel(), lat_grid.ravel()))

    china_land_polygon = [
        (70, 40), (73, 45), (78, 48), (82, 49), (88, 48),
        (92, 50), (96, 52), (105, 53), (115, 53), (120, 53),
        (125, 53), (130, 48), (134, 48), (135, 45), (131, 43),
        (130, 40), (124, 40), (122, 39), (121, 38), (120, 37),
        (121, 35), (120, 32), (121, 30), (120, 27), (118, 24),
        (115, 22), (110, 21), (108, 21), (108, 18), (105, 22),
        (97, 28), (95, 29), (90, 28), (85, 28), (80, 30),
        (75, 37), (70, 40),
    ]
    path = Path(china_land_polygon)
    mask = path.contains_points(points)
    return mask.reshape(len(lats), len(lons))


def create_precip_cmap():
    """全国降水业务分级配色：白/绿/蓝/品红/深红。"""
    colors_list = [
        '#FFFFFF',   # 0-10 mm (includes no precipitation)
        '#36B83B',   # 10-25 mm
        '#5AA6DF',   # 25-50 mm
        '#1236CF',   # 50-100 mm
        '#D017D9',   # 100-250 mm
    ]
    # 0-10 mm均为白色，以突出中到强降水带。
    bounds = [0.0, 10.0, 25.0, 50.0, 100.0, 250.0]
    cmap = colors.ListedColormap(colors_list)
    cmap.set_over('#8B003A')  # >=250 mm
    cmap.set_bad('#FFFFFF')
    norm = colors.BoundaryNorm(bounds, cmap.N)
    return cmap, norm, bounds


def crop_to_china(data, lats, lons):
    """裁剪数据到中国区域 (15-55°N, 70-140°E)"""
    lat_mask = (lats >= CHINA_LAT_RANGE[0]) & (lats <= CHINA_LAT_RANGE[1])
    lon_mask = (lons >= CHINA_LON_RANGE[0]) & (lons <= CHINA_LON_RANGE[1])
    if data.ndim == 2:
        return data[np.ix_(lat_mask, lon_mask)], lats[lat_mask], lons[lon_mask]
    elif data.ndim == 3:
        return data[:, np.ix_(lat_mask, lon_mask)][0], lats[lat_mask], lons[lon_mask]
    elif data.ndim == 4:
        return data[:, :, np.ix_(lat_mask, lon_mask)].reshape(data.shape[0], data.shape[1], lat_mask.sum(), lon_mask.sum()), lats[lat_mask], lons[lon_mask]
    return data, lats, lons


def plot_all_leads(forecast, corrected, truth, lats, lons, date_str, output_dir,
                   land_mask=None, region_suffix='', region_title='', start_day=1,
                   reference_corrected=None, reference_corrected2=None):
    """画所有预报时效的降水对比图。
    
    Args:
        reference_corrected: 可选的全气象变量订正结果（第4列，橙线模型）
        reference_corrected2: (已弃用)保留用于接口兼容
    """
    n_leads = forecast.shape[0]
    # 无 reference 时完全保留原图的三列布局；有 reference 时只追加第4列。
    n_cols = 4 if reference_corrected is not None else 3

    cmap, norm, bounds = create_precip_cmap()
    norm = colors.BoundaryNorm(bounds, cmap.N)

    fig, axes = plt.subplots(n_leads, n_cols, figsize=(4 * n_cols, 2.8 * n_leads))
    if n_leads == 1:
        axes = axes[np.newaxis, :]
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    lon_step = max(1, len(lons) // 6)
    lat_step = max(1, len(lats) // 6)

    for i in range(n_leads):
        for j in range(n_cols):
            ax = axes[i, j]
            if j == 0:
                data = forecast[i]
            elif j == 1:
                data = corrected[i]
            elif j == 2:
                data = truth[i]
            elif j == 3:
                data = reference_corrected[i]
            else:
                data = truth[i]

            # 0-10 mm与海洋NaN均以白色显示，以突出中到强降水区域。
            im = ax.pcolormesh(lons, lats, data, cmap=cmap, norm=norm,
                               shading='auto', vmin=None, vmax=None)

            if land_mask is not None:
                try:
                    ax.contour(lons, lats, land_mask.astype(np.float32),
                               levels=[0.5], colors='black', linewidths=0.8)
                except Exception:
                    pass

            ax.set_xlim(lons.min(), lons.max())
            ax.set_ylim(lats.min(), lats.max())

            if j == 0:
                ax.set_ylabel(f'Day {i + start_day}', fontsize=10, fontweight='bold', rotation=0,
                              labelpad=40, va='center')

            if i == n_leads - 1:
                ax.set_xticks(lons[::lon_step])
                ax.set_xticklabels([f'{x:.0f}°E' for x in lons[::lon_step]], fontsize=7, rotation=45)
            else:
                ax.set_xticklabels([])

            if j == 0:
                ax.set_yticks(lats[::lat_step])
                ax.set_yticklabels([f'{y:.0f}°N' for y in lats[::lat_step]], fontsize=7)
            else:
                ax.set_yticklabels([])

            if i == 0:
                titles = ['(a) Raw Forecast', '(b) Corrected (TP Loss)', '(c) Observation']
                if reference_corrected is not None:
                    titles.append('(d) Corrected (TP+All Met)')
                ax.set_title(titles[j], fontsize=10, pad=2)

            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_edgecolor('black')
                spine.set_linewidth(1.0)

    title_suffix = f' - {region_title}' if region_title else ''
    fig.suptitle(f'Precipitation Comparison{title_suffix} - {date_str}',
                 fontsize=14, fontweight='bold', y=0.96)

    cbar_ax = fig.add_axes([0.92, 0.30, 0.018, 0.40])
    cbar = fig.colorbar(im, cax=cbar_ax, extend='max')
    cbar.set_label('Precipitation (mm)', fontsize=10)
    cbar.set_ticks([5.0, 17.5, 37.5, 75.0, 175.0, 250.0])
    cbar.set_ticklabels(['0–10', '10–25', '25–50', '50–100', '100–250', '≥250'])
    cbar.ax.tick_params(labelsize=8)

    plt.subplots_adjust(left=0.08, right=0.88, top=0.91, bottom=0.05, wspace=0.08, hspace=0.10)
    save_name = f'precip_comparison_all_leads_{date_str}{region_suffix}.png'
    save_path = os.path.join(output_dir, save_name)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_all_leads_error(forecast, corrected, truth, lats, lons, date_str, output_dir,
                         land_mask=None, region_suffix='', region_title='', start_day=1):
    """画所有预报时效的误差对比图：RdBu_r色标，范围固定 -4 到 4"""
    n_leads = forecast.shape[0]

    error_raw = forecast - truth
    error_corrected = corrected - truth
    improvement = np.abs(error_corrected) - np.abs(error_raw)

    # 误差色条范围固定 -4 到 4（根据用户偏好）
    vmax = 4.0

    fig, axes = plt.subplots(n_leads, 3, figsize=(12, 2.8 * n_leads))
    if n_leads == 1:
        axes = axes[np.newaxis, :]

    lon_step = max(1, len(lons) // 6)
    lat_step = max(1, len(lats) // 6)

    for i in range(n_leads):
        data_list = [error_raw[i], error_corrected[i], improvement[i]]
        titles = ['(a) Raw Error', '(b) Corrected Error', '(c) Improvement']

        for j in range(3):
            ax = axes[i, j]
            data = data_list[j]

            # 改善图用同一色标，但改善值可能超出[-4,4]，不过大部分改善都在范围内
            im = ax.pcolormesh(lons, lats, data, cmap='RdBu_r',
                               vmin=-vmax, vmax=vmax, shading='auto')

            if land_mask is not None:
                try:
                    ax.contour(lons, lats, land_mask.astype(np.float32),
                               levels=[0.5], colors='black', linewidths=0.8)
                except Exception:
                    pass

            ax.set_xlim(lons.min(), lons.max())
            ax.set_ylim(lats.min(), lats.max())

            if j == 0:
                ax.set_ylabel(f'Day {i + start_day}', fontsize=10, fontweight='bold', rotation=0,
                              labelpad=40, va='center')

            if i == n_leads - 1:
                ax.set_xticks(lons[::lon_step])
                ax.set_xticklabels([f'{x:.0f}°E' for x in lons[::lon_step]], fontsize=7, rotation=45)
            else:
                ax.set_xticklabels([])

            if j == 0:
                ax.set_yticks(lats[::lat_step])
                ax.set_yticklabels([f'{y:.0f}°N' for y in lats[::lat_step]], fontsize=7)
            else:
                ax.set_yticklabels([])

            if i == 0:
                ax.set_title(titles[j], fontsize=10, pad=2)

            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_edgecolor('black')
                spine.set_linewidth(1.0)

    title_suffix = f' - {region_title}' if region_title else ''
    fig.suptitle(f'Error Comparison{title_suffix} - {date_str}',
                 fontsize=14, fontweight='bold', y=0.96)

    cbar_ax = fig.add_axes([0.92, 0.30, 0.015, 0.40])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Error (mm)', fontsize=10)
    cbar.ax.tick_params(labelsize=8)

    plt.subplots_adjust(left=0.08, right=0.88, top=0.91, bottom=0.05, wspace=0.08, hspace=0.10)
    save_name = f'error_comparison_all_leads_{date_str}{region_suffix}.png'
    save_path = os.path.join(output_dir, save_name)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {save_path}")


def apply_land_mask_to_nan(arrays, land_mask):
    """将海洋/非陆地区域设为NaN，以便pcolormesh不显示"""
    results = []
    ocean_mask = ~land_mask
    for arr in arrays:
        a = np.nan_to_num(arr, nan=0.0).copy()
        if a.ndim == 4:
            a[:, :, ocean_mask] = np.nan
        elif a.ndim == 3:
            a[:, ocean_mask] = np.nan
        elif a.ndim == 2:
            a[ocean_mask] = np.nan
        results.append(a)
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Plot precipitation comparison from NC results")
    parser.add_argument("nc_path", nargs="?", default=None, help="Path to NC file (auto-detected if not provided)")
    parser.add_argument("--reference_nc", type=str, default=None,
                        help="参考模型的NC文件路径（可选，添加第4列显示参考模型订正结果）")
    parser.add_argument("--reference_nc2", type=str, default=None,
                        help="第二参考模型的NC文件路径（可选，添加第5列显示FSS模型订正结果）")
    parser.add_argument("--max_dates", type=int, default=6, help="Max number of dates to plot")
    parser.add_argument("--months", type=int, nargs="+", default=None,
                        help="只绘制指定起报月份，例如 --months 7 8")
    parser.add_argument("--dates_per_month", type=int, default=0,
                        help="指定月份时每个月均匀抽取的起报日期数；0 时仍使用 --max_dates 总数")
    parser.add_argument("--start_day", type=int, default=1, help="起始预报天数（1表示从Day1开始，11表示从Day11开始）")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for plots")
    parser.add_argument("--region_only", type=str, default=None, choices=['china', 'global'],
                        help="Only plot the specified region (china or global). Default: auto-detect and plot both.")
    args = parser.parse_args()

    nc_path = args.nc_path
    if nc_path is None:
        nc_path = find_latest_nc()
        if nc_path is None:
            print("Error: No NC file found. Specify path or run training first.")
            sys.exit(1)
        print(f"Auto-detected NC file: {nc_path}")

    if not os.path.exists(nc_path):
        print(f"Error: NC file not found at {nc_path}")
        sys.exit(1)

    forecast, corrected, truth, lats, lons, sample_dates, lead_times, land_mask = load_nc_file(nc_path)

    # 加载全气象变量模型 NC（作为第4列）；必须与基线逐日期、逐网格对齐。
    ref_forecast = None
    ref_corrected = None
    ref_truth = None
    ref_lats = None
    ref_lons = None
    ref_land_mask = None
    ref_lead_times = None
    if args.reference_nc is not None:
        if not os.path.exists(args.reference_nc):
            print(f"Warning: Reference NC file not found: {args.reference_nc}")
        else:
            ref_forecast, ref_corrected, ref_truth, ref_lats, ref_lons, ref_dates, ref_lead_times, ref_land_mask = load_nc_file(args.reference_nc)
            print(f"Loaded reference NC file: {args.reference_nc}")
            print(f"  Shape: forecast={ref_forecast.shape}, corrected={ref_corrected.shape}, truth={ref_truth.shape}")
            if ref_corrected.shape != corrected.shape or ref_truth.shape != truth.shape:
                raise ValueError("Reference NC and baseline NC have different dimensions; cannot make a fair 4-column map")
            if not (np.allclose(ref_lats, lats) and np.allclose(ref_lons, lons)):
                raise ValueError("Reference NC and baseline NC have different grids; cannot make a fair 4-column map")
            if ref_dates != sample_dates:
                raise ValueError("Reference NC and baseline NC have different initialization dates; cannot make a fair 4-column map")

    # 加载第二参考模型NC数据（FSS模型，橙色线）
    ref2_forecast = None
    ref2_corrected = None
    ref2_truth = None
    ref2_lats = None
    ref2_lons = None
    ref2_land_mask = None
    ref2_lead_times = None
    if args.reference_nc2 is not None:
        if not os.path.exists(args.reference_nc2):
            print(f"Warning: Second reference NC file not found: {args.reference_nc2}")
        else:
            ref2_forecast, ref2_corrected, ref2_truth, ref2_lats, ref2_lons, _, ref2_lead_times, ref2_land_mask = load_nc_file(args.reference_nc2)
            print(f"Loaded second reference NC file: {args.reference_nc2}")
            print(f"  Shape: forecast={ref2_forecast.shape}, corrected={ref2_corrected.shape}, truth={ref2_truth.shape}")

    # 起始天数截取（start_day=1表示从Day1开始，start_day=11表示从Day11开始）
    start_idx = args.start_day - 1
    if start_idx > 0:
        forecast = forecast[:, start_idx:]
        corrected = corrected[:, start_idx:]
        truth = truth[:, start_idx:]
        lead_times = lead_times[start_idx:]
        if ref_forecast is not None:
            ref_forecast = ref_forecast[:, start_idx:]
            ref_corrected = ref_corrected[:, start_idx:]
            ref_truth = ref_truth[:, start_idx:]
        if ref2_forecast is not None:
            ref2_forecast = ref2_forecast[:, start_idx:]
            ref2_corrected = ref2_corrected[:, start_idx:]
            ref2_truth = ref2_truth[:, start_idx:]
        print(f"截取起始天数: Day {args.start_day} → Day {args.start_day + len(lead_times) - 1}")

    print(f"Loaded NC file: {nc_path}")
    print(f"Shape: forecast={forecast.shape}, corrected={corrected.shape}, truth={truth.shape}")
    print(f"Lats: {lats.shape}, range: [{lats.min():.2f}, {lats.max():.2f}]")
    print(f"Lons: {lons.shape}, range: [{lons.min():.2f}, {lons.max():.2f}]")

    # 如果NC文件没有陆地掩码，基于固定地理信息生成（按数据范围）
    if land_mask is None:
        lat_min, lat_max = lats.min(), lats.max()
        lon_min, lon_max = lons.min(), lons.max()
        # 如果是全球范围，使用中国区域固定掩码，不生成全球掩码（CRA只有中国数据）
        if (lat_min < 0 and lat_max > 50) or (lon_min < 50 or lon_max > 150):
            print("Detected global range. Creating China land mask for reference.")
            land_mask = create_china_land_mask(lats, lons)
        else:
            land_mask = create_china_land_mask(lats, lons)
        print(f"Created geographic land_mask: shape={land_mask.shape}, land ratio={np.mean(land_mask):.2%}")

    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(os.path.dirname(nc_path), '..', 'plots')
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # 可按月份选取多个典型个例。指定 dates_per_month 时，每个月独立均匀抽样，
    # 避免 7 月样本把 8 月样本挤掉，或反之。
    candidate_indices = list(range(len(sample_dates)))
    if args.months is not None:
        requested_months = sorted(set(args.months))
        if any(month < 1 or month > 12 for month in requested_months):
            raise ValueError(f"Invalid months: {requested_months}")
        candidate_indices = [
            idx for idx, date in enumerate(sample_dates)
            if len(str(date)) >= 6 and int(str(date)[4:6]) in requested_months
        ]
        if not candidate_indices:
            raise ValueError(f"No initialization dates found for months {requested_months}")

        if args.dates_per_month > 0:
            selected_indices = []
            for month in requested_months:
                month_indices = [idx for idx in candidate_indices if int(str(sample_dates[idx])[4:6]) == month]
                if not month_indices:
                    print(f"[Warning] No initialization dates found for month {month}")
                    continue
                n_month = min(args.dates_per_month, len(month_indices))
                positions = np.linspace(0, len(month_indices) - 1, n_month, dtype=int)
                selected_indices.extend(month_indices[pos] for pos in positions)
        else:
            n_dates = min(args.max_dates, len(candidate_indices))
            positions = np.linspace(0, len(candidate_indices) - 1, n_dates, dtype=int)
            selected_indices = [candidate_indices[pos] for pos in positions]
    else:
        n_dates = min(args.max_dates, len(sample_dates))
        positions = np.linspace(0, len(sample_dates) - 1, n_dates, dtype=int)
        selected_indices = [positions_item for positions_item in positions]

    print(f"\nPlotting {len(selected_indices)} sample dates -> {output_dir}")
    print(f"Selected dates: {[sample_dates[idx] for idx in selected_indices]}")

    for idx in selected_indices:
        date_str = sample_dates[idx]
        print(f"\n=== Processing {date_str} (index {idx}) ===")

        f_all = forecast[idx].copy()   # (T, H, W)
        c_all = corrected[idx].copy()
        t_all = truth[idx].copy()
        lat_all = lats
        lon_all = lons
        lm_all = land_mask

        # 参考模型数据（基线，紫色线）
        rc_all = ref_corrected[idx].copy() if ref_corrected is not None else None
        # 第二参考模型数据（FSS模型，橙色线）
        rc2_all = ref2_corrected[idx].copy() if ref2_corrected is not None else None

        # ===== 全球区域图 =====
        is_global = (lat_all.min() < 0 or lon_all.min() < 50)
        plot_global = (args.region_only != 'china') and is_global
        plot_china = (args.region_only != 'global')

        if plot_global:
            print("  -> Global region plot")
            # 应用海洋掩码
            f_g, c_g, t_g = apply_land_mask_to_nan([f_all, c_all, t_all], lm_all)
            rc_g = apply_land_mask_to_nan([rc_all], lm_all)[0] if rc_all is not None else None
            rc2_g = apply_land_mask_to_nan([rc2_all], lm_all)[0] if rc2_all is not None else None
            plot_all_leads(f_g, c_g, t_g, lat_all, lon_all, date_str, output_dir,
                           land_mask=lm_all, region_suffix='_global', region_title='Global',
                           start_day=args.start_day, reference_corrected=rc_g,
                           reference_corrected2=rc2_g)
            plot_all_leads_error(f_g, c_g, t_g, lat_all, lon_all, date_str, output_dir,
                                 land_mask=lm_all, region_suffix='_global', region_title='Global',
                                 start_day=args.start_day)

        # ===== 中国区域图 =====
        if plot_china:
            print("  -> China region crop (15-55N, 70-140E)")
            # 裁剪数据到中国区域
            try:
                f_cn, lat_cn, lon_cn = crop_to_china(f_all, lat_all, lon_all)
                c_cn, _, _ = crop_to_china(c_all, lat_all, lon_all)
                t_cn, _, _ = crop_to_china(t_all, lat_all, lon_all)
                rc_cn, _, _ = crop_to_china(rc_all, lat_all, lon_all) if rc_all is not None else (None, None, None)
                rc2_cn, _, _ = crop_to_china(rc2_all, lat_all, lon_all) if rc2_all is not None else (None, None, None)
                lm_cn_2d = None
                if lm_all is not None:
                    lm_cn_cropped, _, _ = crop_to_china(lm_all.astype(np.float32), lat_all, lon_all)
                    lm_cn_2d = lm_cn_cropped > 0.5
            except Exception as e:
                print(f"  [Warning] Crop to china failed ({e}), using raw arrays.")
                f_cn, c_cn, t_cn = f_all, c_all, t_all
                lat_cn, lon_cn = lat_all, lon_all
                lm_cn_2d = lm_all
                rc_cn = rc_all
                rc2_cn = rc2_all

            # 如果裁剪后没有陆地掩码，基于中国区域重新生成
            if lm_cn_2d is None:
                lm_cn_2d = create_china_land_mask(lat_cn, lon_cn)

            # 应用海洋掩码（中国区域）
            f_cn, c_cn, t_cn = apply_land_mask_to_nan([f_cn, c_cn, t_cn], lm_cn_2d)
            rc_cn = apply_land_mask_to_nan([rc_cn], lm_cn_2d)[0] if rc_cn is not None else None
            rc2_cn = apply_land_mask_to_nan([rc2_cn], lm_cn_2d)[0] if rc2_cn is not None else None

            plot_all_leads(f_cn, c_cn, t_cn, lat_cn, lon_cn, date_str, output_dir,
                           land_mask=lm_cn_2d, region_suffix='_china', region_title='China',
                           start_day=args.start_day, reference_corrected=rc_cn,
                           reference_corrected2=rc2_cn)
            plot_all_leads_error(f_cn, c_cn, t_cn, lat_cn, lon_cn, date_str, output_dir,
                                 land_mask=lm_cn_2d, region_suffix='_china', region_title='China',
                                 start_day=args.start_day)

    print(f"\nAll plots saved to: {output_dir}")
    print("Done!")


if __name__ == "__main__":
    main()
