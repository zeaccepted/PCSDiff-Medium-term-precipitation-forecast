"""Create publication-ready 10-day four-column Stage-1/Stage-2 summary maps."""
from __future__ import annotations

import argparse
import glob
import os

# HDF5 locking is unnecessary for this read-only job and is unreliable on
# several Lustre/GPFS configurations used by Slurm clusters.
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import netCDF4 as nc
import numpy as np

import config_server as config


def _validate_sr_file(path):
    required = {"init_date", "lead", "lat", "lon", "phase2_multiscale_ddim", "truth"}
    try:
        with nc.Dataset(path) as dataset:
            missing = required.difference(dataset.variables)
            if missing:
                return False, f"missing variables: {sorted(missing)}"
            samples = int(dataset.variables["phase2_multiscale_ddim"].shape[0])
            if samples <= 0:
                return False, "zero samples"
            # Force HDF5 to read data chunks at both ends, not just the header.
            dataset.variables["phase2_multiscale_ddim"][0, 0, 0]
            dataset.variables["phase2_multiscale_ddim"][samples - 1, -1, -1]
        return True, "ok"
    except (OSError, RuntimeError, IndexError) as error:
        return False, str(error)


def _resolve_sr_file(requested_path):
    directory = os.path.dirname(requested_path)
    stem, extension = os.path.splitext(os.path.basename(requested_path))
    alternatives = sorted(glob.glob(os.path.join(directory, f"{stem}_job*{extension}")),
                          key=os.path.getmtime, reverse=True)
    candidates = [requested_path] + [path for path in alternatives if path != requested_path]
    diagnostics = []
    for path in candidates:
        if not os.path.isfile(path):
            diagnostics.append(f"{path}: missing")
            continue
        valid, reason = _validate_sr_file(path)
        if valid:
            if path != requested_path:
                print(f"[Paper figures] fixed v3 NC is unavailable; using valid fallback: {path}")
            return path
        diagnostics.append(f"{path}: {reason}")
    detail = "\n  ".join(diagnostics)
    raise RuntimeError(
        "No complete readable Stage-2 v3 NetCDF was found. Wait until the evaluation job is COMPLETED "
        "or rerun: sbatch submit_evaluate_super_resolution.sh\n  " + detail
    )


def _decode_strings(variable):
    values = variable[:]
    if values.dtype.kind == "S" and values.ndim == 1:
        return [bytes(value).decode("utf-8").strip("\x00 ") for value in values]
    if values.dtype.kind in {"S", "U"} and values.ndim > 1:
        return [str(value).strip("\x00 ") for value in nc.chartostring(values)]
    return [str(value).strip("\x00 ") for value in values]


def _precipitation_style():
    boundaries = [0.0, 10.0, 25.0, 50.0, 100.0, 250.0]
    cmap = ListedColormap(["#FFFFFF", "#36B83B", "#5AA6DF", "#1236CF", "#D017D9"])
    cmap.set_over("#8B003A")
    cmap.set_bad("#FFFFFF")
    return cmap, BoundaryNorm(boundaries, cmap.N), boundaries


def _evenly_spaced_dates(candidates, count, month):
    candidates = sorted(set(candidates))
    if len(candidates) < count:
        raise RuntimeError(f"2022-{month:02d} only has {len(candidates)} complete init dates; need {count}")
    positions = np.linspace(0, len(candidates) - 1, count)
    indices = [int(round(position)) for position in positions]
    # Rounding can only duplicate indices for very short lists; keep the
    # selection deterministic and fill any missing slots chronologically.
    selected = []
    for index in indices:
        if candidates[index] not in selected:
            selected.append(candidates[index])
    for date in candidates:
        if len(selected) == count:
            break
        if date not in selected:
            selected.append(date)
    return sorted(selected)


def _plot_one_date(date, raw, corrected, super_resolved, truth,
                   low_lats, low_lons, high_lats, high_lons,
                   low_land_mask, high_land_mask, output_path):
    n_leads = raw.shape[0]
    cmap, norm, boundaries = _precipitation_style()
    fig, axes = plt.subplots(n_leads, 4, figsize=(16, 2.8 * n_leads))
    if n_leads == 1:
        axes = axes[np.newaxis, :]

    column_titles = ["(a) Raw ECMWF Forecast", "(b) Low-resolution Correction",
                     "(c) Super-resolved Correction", "(d) CRA Truth"]
    lon_ticks = np.linspace(float(np.min(high_lons)), float(np.max(high_lons)), 6)
    lat_ticks = np.linspace(float(np.min(high_lats)), float(np.max(high_lats)), 6)
    image = None
    for lead_index in range(n_leads):
        fields = [np.where(low_land_mask, raw[lead_index], np.nan),
                  np.where(low_land_mask, corrected[lead_index], np.nan),
                  np.where(high_land_mask, super_resolved[lead_index], np.nan),
                  np.where(high_land_mask, truth[lead_index], np.nan)]
        grids = [(low_lats, low_lons), (low_lats, low_lons),
                 (high_lats, high_lons), (high_lats, high_lons)]
        for column, (field, (lats, lons)) in enumerate(zip(fields, grids)):
            axis = axes[lead_index, column]
            image = axis.pcolormesh(lons, lats, field, cmap=cmap, norm=norm, shading="auto")
            # Exactly the Stage-1 map treatment: native Stage-1 land-mask contour.
            axis.contour(low_lons, low_lats, low_land_mask.astype(np.float32),
                         levels=[0.5], colors="black", linewidths=0.8, zorder=20)
            axis.set_xlim(float(np.min(high_lons)), float(np.max(high_lons)))
            axis.set_ylim(float(np.min(high_lats)), float(np.max(high_lats)))
            if lead_index == 0:
                axis.set_title(column_titles[column], fontsize=10, pad=3)
            if column == 0:
                axis.set_ylabel(f"Day {lead_index + 1}", fontsize=10, fontweight="bold",
                                rotation=0, labelpad=37, va="center")
                axis.set_yticks(lat_ticks)
                axis.set_yticklabels([f"{value:.0f}°N" for value in lat_ticks], fontsize=7)
            else:
                axis.set_yticks(lat_ticks)
                axis.set_yticklabels([])
            if lead_index == n_leads - 1:
                axis.set_xticks(lon_ticks)
                axis.set_xticklabels([f"{value:.0f}°E" for value in lon_ticks],
                                     fontsize=7, rotation=45, ha="right")
            else:
                axis.set_xticks(lon_ticks)
                axis.set_xticklabels([])
            for spine in axis.spines.values():
                spine.set_visible(True)
                spine.set_edgecolor("black")
                spine.set_linewidth(1.0)

    fig.suptitle(f"Precipitation Forecast Correction and Super Resolution - China - {date}",
                 fontsize=14, fontweight="bold", y=0.965)
    colorbar_axis = fig.add_axes([0.92, 0.30, 0.018, 0.40])
    colorbar = fig.colorbar(image, cax=colorbar_axis, boundaries=boundaries, extend="max")
    colorbar.set_label("Precipitation (mm/day)", fontsize=10)
    colorbar.set_ticks([5.0, 17.5, 37.5, 75.0, 175.0, 250.0])
    colorbar.set_ticklabels(["0–10", "10–25", "25–50", "50–100", "100–250", "≥250"])
    colorbar.ax.tick_params(labelsize=8)
    plt.subplots_adjust(left=0.08, right=0.89, top=0.935, bottom=0.045, wspace=0.08, hspace=0.10)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    project_root = os.path.abspath(os.environ.get("RVD_PROJECT_ROOT", os.path.dirname(__file__)))
    default_stage1 = config.sr_config["test_correction_file"]
    default_sr = os.path.join(config.sr_config["result_dir"], "evaluation", "super_resolution_test_v3.nc")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1_nc", default=default_stage1)
    parser.add_argument("--sr_nc", default=default_sr)
    parser.add_argument("--output_dir", default=os.path.join(project_root, "results", "paper_main_figures"))
    parser.add_argument("--dates_per_month", type=int, default=4)
    args = parser.parse_args()
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
    complete_dates = []
    for date in stage1_dates:
        if len(date) == 8 and date.startswith("2022") and all((date, lead) in sr_by_date_lead for lead in range(1, 11)):
            complete_dates.append(date)

    selected_dates = []
    for month in (6, 7, 8, 9):
        candidates = [date for date in complete_dates if int(date[4:6]) == month]
        chosen = _evenly_spaced_dates(candidates, args.dates_per_month, month)
        selected_dates.extend(chosen)
        print(f"[Paper figures] 2022-{month:02d}: {', '.join(chosen)}")

    high_lat_indices = np.abs(low_lats[:, None] - high_lats[None, :]).argmin(axis=0)
    high_lon_indices = np.abs(low_lons[:, None] - high_lons[None, :]).argmin(axis=0)
    high_land_mask = low_land_mask[np.ix_(high_lat_indices, high_lon_indices)]

    manifest_path = os.path.join(args.output_dir, "selected_init_dates.txt")
    # Keep the files open but load only the current date's ten lead days. This
    # avoids materializing the complete native-resolution test set in RAM.
    with nc.Dataset(args.stage1_nc) as stage1, nc.Dataset(args.sr_nc) as sr, \
            open(manifest_path, "w", encoding="utf-8") as manifest:
        for date in selected_dates:
            stage1_index = stage1_by_date[date]
            sr_indices = [sr_by_date_lead[(date, lead)] for lead in range(1, 11)]
            raw = np.asarray(np.ma.filled(stage1.variables["forecast"][stage1_index, :10], np.nan),
                             dtype=np.float32)
            corrected = np.asarray(
                np.ma.filled(stage1.variables["corrected"][stage1_index, :10], np.nan), dtype=np.float32
            )
            super_resolved = np.asarray(
                np.ma.filled(sr.variables["phase2_multiscale_ddim"][sr_indices], np.nan), dtype=np.float32
            )
            truth = np.asarray(np.ma.filled(sr.variables["truth"][sr_indices], np.nan), dtype=np.float32)
            output_path = os.path.join(args.output_dir, f"paper_main_{date}_china.png")
            _plot_one_date(date, raw, corrected, super_resolved, truth,
                           low_lats, low_lons, high_lats, high_lons,
                           low_land_mask, high_land_mask, output_path)
            manifest.write(f"{date}\t{os.path.basename(output_path)}\n")
            print(f"[Paper figures] saved: {output_path}")
    print(f"[Paper figures] complete: {len(selected_dates)} figures -> {args.output_dir}")


if __name__ == "__main__":
    main()
