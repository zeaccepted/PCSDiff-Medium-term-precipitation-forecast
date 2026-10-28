"""Plot three hand-picked publication cases as one 3-row by 4-column figure."""
from __future__ import annotations

import argparse
import os
from datetime import datetime, timedelta

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np
from scipy.interpolate import RegularGridInterpolator

import config_server as config
from plot_paper_main_figures import _decode_strings, _precipitation_style, _resolve_sr_file


CASES = (("20220815", 4), ("20220715", 5), ("20220822", 8))


def _coordinate(ds, names):
    for name in names:
        if name in ds.variables:
            return np.asarray(ds.variables[name][:], dtype=np.float64)
    raise KeyError(f"None of the coordinates {names} exist")


def _resample_mask_to_high_grid(low_lats, low_lons, low_land_mask, high_lats, high_lons):
    """Smooth fallback with clipped queries, avoiding artificial domain-edge contours."""
    lat_order = np.argsort(low_lats)
    lon_order = np.argsort(low_lons)
    source = low_land_mask[np.ix_(lat_order, lon_order)].astype(np.float32)
    sorted_lats, sorted_lons = low_lats[lat_order], low_lons[lon_order]
    interpolator = RegularGridInterpolator(
        (sorted_lats, sorted_lons), source, method="linear",
        bounds_error=False, fill_value=None,
    )
    high_lon_grid, high_lat_grid = np.meshgrid(high_lons, high_lats)
    # The high and low grid cell centres differ slightly at their outer edge.
    # Clamp only map-mask queries so extrapolation cannot create a frame line.
    query_lats = np.clip(high_lat_grid, sorted_lats[0], sorted_lats[-1])
    query_lons = np.clip(high_lon_grid, sorted_lons[0], sorted_lons[-1])
    points = np.column_stack((query_lats.ravel(), query_lons.ravel()))
    return interpolator(points).reshape(len(high_lats), len(high_lons)).astype(np.float32)


def _native_high_land_mask(truth_dir, high_lats, high_lons, fallback_mask):
    """Read the native 0.25-degree CRA mask used by the high-resolution panels."""
    first_date, first_lead = CASES[0]
    target_date = (datetime.strptime(first_date, "%Y%m%d")
                   + timedelta(days=first_lead - 1)).strftime("%Y%m%d")
    truth_path = os.path.join(truth_dir, f"CRA_truth_{target_date}.nc")
    try:
        with nc.Dataset(truth_path) as dataset:
            variable = dataset.variables["tp"]
            raw = variable[:]
            valid = ~np.ma.getmaskarray(raw)
            values = np.asarray(np.ma.filled(raw, np.nan), dtype=np.float32)
            valid = valid & np.isfinite(values)
            valid = np.squeeze(valid)
            if valid.ndim != 2 or np.all(valid) or not np.any(valid):
                raise ValueError("CRA tp does not contain a usable land/ocean mask")
            source_lats = _coordinate(dataset, ("latitude", "lat", "LAT"))
            source_lons = _coordinate(dataset, ("longitude", "lon", "LON"))
        lat_indices = np.abs(source_lats[:, None] - high_lats[None, :]).argmin(axis=0)
        lon_indices = np.abs(source_lons[:, None] - high_lons[None, :]).argmin(axis=0)
        native_mask = valid[np.ix_(lat_indices, lon_indices)].astype(np.float32)
        print(f"[Selected cases] high-resolution map mask: native CRA {native_mask.shape} ({truth_path})")
        return native_mask
    except (OSError, KeyError, ValueError, IndexError) as error:
        print(f"[Selected cases] WARNING: native CRA map mask unavailable ({error}); "
              "using clipped high-grid mask mapping")
        return fallback_mask


def _configure_microsoft_yahei(project_root):
    explicit_paths = [
        os.environ.get("RVD_PAPER_FONT", ""),
        os.path.join(project_root, "fonts", "msyh.ttc"),
        os.path.join(project_root, "fonts", "msyh.ttf"),
        os.path.expanduser("~/.fonts/msyh.ttc"),
        os.path.expanduser("~/.fonts/msyh.ttf"),
    ]
    for path in explicit_paths:
        if path and os.path.isfile(path):
            font_manager.fontManager.addfont(path)
            family = font_manager.FontProperties(fname=path).get_name()
            plt.rcParams["font.family"] = family
            plt.rcParams["font.sans-serif"] = [family]
            plt.rcParams["axes.unicode_minus"] = False
            print(f"[Selected cases] font: {family} ({path})")
            return family

    for family in ("Microsoft YaHei", "Microsoft YaHei UI"):
        try:
            font_manager.findfont(family, fallback_to_default=False)
            plt.rcParams["font.family"] = family
            plt.rcParams["font.sans-serif"] = [family]
            plt.rcParams["axes.unicode_minus"] = False
            print(f"[Selected cases] font: {family} (system)")
            return family
        except ValueError:
            continue

    # Keep the job runnable on a minimal Linux node, but make the substitution
    # explicit. Upload msyh.ttc to project/fonts/ for exact YaHei rendering.
    fallback = "DejaVu Sans"
    for family in ("Noto Sans CJK SC", "WenQuanYi Micro Hei", "SimHei", fallback):
        try:
            font_manager.findfont(family, fallback_to_default=False)
            fallback = family
            break
        except ValueError:
            continue
    plt.rcParams["font.family"] = fallback
    plt.rcParams["font.sans-serif"] = [fallback]
    plt.rcParams["axes.unicode_minus"] = False
    print(f"[Selected cases] WARNING: Microsoft YaHei not found; using {fallback}. "
          "For exact typography upload fonts/msyh.ttc.")
    return fallback


def _plot_combined(case_data, low_lats, low_lons, high_lats, high_lons,
                   low_land_mask, high_map_mask, output_path):
    cmap, norm, boundaries = _precipitation_style()
    high_land_mask = high_map_mask >= 0.5
    titles = ["(a) Raw ECMWF Forecast",
              "(b) Stage 1 Output\nLow-resolution Correction",
              "(c) Stage 2 Output\nHigh-resolution Correction",
              "(d) CRA Truth"]

    fig, axes = plt.subplots(len(case_data), 4, figsize=(28, 15))
    lon_ticks = np.linspace(float(np.min(high_lons)), float(np.max(high_lons)), 6)
    lat_ticks = np.linspace(float(np.min(high_lats)), float(np.max(high_lats)), 6)
    geographic_aspect = 1.0 / np.cos(np.deg2rad(float(np.mean(high_lats))))
    image = None
    for row, (date, lead, raw, corrected, super_resolved, truth) in enumerate(case_data):
        fields = [np.where(low_land_mask, raw, np.nan),
                  np.where(low_land_mask, corrected, np.nan),
                  np.where(high_land_mask, super_resolved, np.nan),
                  np.where(high_land_mask, truth, np.nan)]
        grids = [(low_lats, low_lons), (low_lats, low_lons),
                 (high_lats, high_lons), (high_lats, high_lons)]
        for column, (field, (lats, lons)) in enumerate(zip(fields, grids)):
            axis = axes[row, column]
            image = axis.pcolormesh(lons, lats, field, cmap=cmap, norm=norm,
                                    shading="auto", rasterized=True)
            if column < 2:
                axis.contour(low_lons, low_lats, low_land_mask.astype(np.float32),
                             levels=[0.5], colors="black", linewidths=1.0, zorder=20)
            else:
                axis.contour(high_lons, high_lats, high_map_mask,
                             levels=[0.5], colors="black", linewidths=1.0, zorder=20)
            axis.set_xlim(float(np.min(high_lons)), float(np.max(high_lons)))
            axis.set_ylim(float(np.min(high_lats)), float(np.max(high_lats)))
            axis.set_aspect(geographic_aspect, adjustable="box")
            # A fixed geographic aspect makes each map shorter than its GridSpec
            # cell.  Anchor the outer rows toward the centre so that this unused
            # vertical space does not become a large gap between case rows.
            if row == 0:
                axis.set_anchor("S")
            elif row == len(case_data) - 1:
                axis.set_anchor("N")
            else:
                axis.set_anchor("C")
            if row == 0:
                axis.set_title(titles[column], fontsize=16, fontweight="bold", pad=10)
            axis.set_xticks(lon_ticks)
            if row == len(case_data) - 1:
                axis.set_xticklabels([f"{value:.0f}°E" for value in lon_ticks],
                                     fontsize=12, rotation=40, ha="right")
            else:
                axis.set_xticklabels([])
            axis.set_yticks(lat_ticks)
            if column == 0:
                axis.set_yticklabels([f"{value:.0f}°N" for value in lat_ticks], fontsize=12)
            else:
                axis.set_yticklabels([])
            axis.tick_params(width=1.1, length=4)
            for spine in axis.spines.values():
                spine.set_visible(True)
                spine.set_edgecolor("black")
                spine.set_linewidth(1.1)
        formatted_date = f"{date[:4]}-{date[4:6]}-{date[6:]}"
        axes[row, 0].set_ylabel(f"Initial Date: {formatted_date}\nLead Time: {lead} Day",
                                fontsize=15, fontweight="bold", rotation=0,
                                labelpad=112, va="center", ha="center")

    colorbar_axis = fig.add_axes([0.925, 0.24, 0.014, 0.52])
    colorbar = fig.colorbar(image, cax=colorbar_axis, boundaries=boundaries, extend="max")
    colorbar.set_label("Daily Total Precipitation (mm)", fontsize=15, fontweight="normal", labelpad=12)
    colorbar.set_ticks([5.0, 17.5, 37.5, 75.0, 175.0, 250.0])
    colorbar.set_ticklabels(["0–10", "10–25", "25–50", "50–100", "100–250", "≥250"])
    colorbar.ax.tick_params(labelsize=12, width=1.0)
    plt.subplots_adjust(left=0.14, right=0.90, top=0.92, bottom=0.08, wspace=0.10, hspace=0.02)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    pdf_path = os.path.splitext(output_path)[0] + ".pdf"
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    project_root = os.path.abspath(os.environ.get("RVD_PROJECT_ROOT", os.path.dirname(__file__)))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1_nc", default=config.sr_config["test_correction_file"])
    parser.add_argument("--sr_nc", default=os.path.join(
        config.sr_config["result_dir"], "evaluation", "super_resolution_test_v3.nc"
    ))
    parser.add_argument("--output_dir", default=os.path.join(project_root, "results", "paper_selected_cases"))
    args = parser.parse_args()
    _configure_microsoft_yahei(project_root)
    if not os.path.isfile(args.stage1_nc):
        raise FileNotFoundError(args.stage1_nc)
    args.sr_nc = _resolve_sr_file(args.sr_nc)
    os.makedirs(args.output_dir, exist_ok=True)

    with nc.Dataset(args.stage1_nc) as stage1:
        stage1_dates = _decode_strings(stage1.variables["time"])
        low_lats = np.asarray(stage1.variables["lat"][:], dtype=np.float64)
        low_lons = np.asarray(stage1.variables["lon"][:], dtype=np.float64)
        low_land_mask = np.asarray(np.ma.filled(stage1.variables["land_mask"][:], 0.0),
                                   dtype=np.float32) > 0.5
    with nc.Dataset(args.sr_nc) as sr:
        sr_dates = _decode_strings(sr.variables["init_date"])
        sr_leads = np.asarray(sr.variables["lead"][:], dtype=np.int32)
        high_lats = np.asarray(sr.variables["lat"][:], dtype=np.float64)
        high_lons = np.asarray(sr.variables["lon"][:], dtype=np.float64)

    stage1_by_date = {date: index for index, date in enumerate(stage1_dates)}
    sr_by_date_lead = {(date, int(lead)): index for index, (date, lead) in enumerate(zip(sr_dates, sr_leads))}
    fallback_high_mask = _resample_mask_to_high_grid(
        low_lats, low_lons, low_land_mask, high_lats, high_lons
    )
    high_map_mask = _native_high_land_mask(
        config.sr_config["truth_dir"], high_lats, high_lons, fallback_high_mask
    )

    case_data = []
    with nc.Dataset(args.stage1_nc) as stage1, nc.Dataset(args.sr_nc) as sr:
        for date, lead in CASES:
            if date not in stage1_by_date or (date, lead) not in sr_by_date_lead:
                raise KeyError(f"Missing selected case: init={date}, lead={lead}")
            stage1_index = stage1_by_date[date]
            sr_index = sr_by_date_lead[(date, lead)]
            raw = np.asarray(np.ma.filled(
                stage1.variables["forecast"][stage1_index, lead - 1], np.nan), dtype=np.float32
            )
            corrected = np.asarray(np.ma.filled(
                stage1.variables["corrected"][stage1_index, lead - 1], np.nan), dtype=np.float32
            )
            super_resolved = np.asarray(np.ma.filled(
                sr.variables["phase2_multiscale_ddim"][sr_index], np.nan), dtype=np.float32
            )
            truth = np.asarray(np.ma.filled(sr.variables["truth"][sr_index], np.nan), dtype=np.float32)
            case_data.append((date, lead, raw, corrected, super_resolved, truth))
    output_path = os.path.join(args.output_dir, "selected_cases_combined_china.png")
    _plot_combined(case_data, low_lats, low_lons, high_lats, high_lons,
                   low_land_mask, high_map_mask, output_path)
    print(f"[Selected cases] combined figure saved: {output_path}")


if __name__ == "__main__":
    main()
