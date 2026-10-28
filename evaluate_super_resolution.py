"""Run real conditional-diffusion sampling and compare both SR phases."""
from __future__ import annotations

import csv
import os

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import netCDF4 as nc
import numpy as np
import torch
from scipy.ndimage import uniform_filter
from torch.utils.data import DataLoader
from tqdm import tqdm

import config_server as config
from data.datasets.super_resolution import SuperResolutionDataset
from modules.super_resolution_model import ConditionalDenoisingUNet, RegressionConvResNet, TwoStageSuperResolution


def _temporary_output_path(final_path):
    """Write beside the final file so publication is an atomic filesystem rename."""
    job_id = os.environ.get("SLURM_JOB_ID", str(os.getpid()))
    return f"{final_path}.tmp.{job_id}"


def _publish_output(temporary_path, final_path):
    """Replace an old result safely; fall back to a job-specific name if locked."""
    try:
        os.replace(temporary_path, final_path)
        return final_path
    except PermissionError:
        stem, extension = os.path.splitext(final_path)
        job_id = os.environ.get("SLURM_JOB_ID", str(os.getpid()))
        fallback = f"{stem}_job{job_id}{extension}"
        os.replace(temporary_path, fallback)
        print(f"[SR] Existing output is locked/read-only; wrote: {fallback}")
        return fallback


def _metrics(pred, truth, mask):
    p, y = pred[mask], truth[mask]
    result = {"rmse": float(np.sqrt(np.mean((p - y) ** 2))), "mae": float(np.mean(np.abs(p - y)))}
    for threshold in (0.1, 5.0, 10.0, 25.0, 50.0):
        pp, yy = p >= threshold, y >= threshold
        denominator = np.sum(pp | yy)
        result[f"csi_{threshold:g}"] = float(np.sum(pp & yy) / denominator) if denominator else np.nan
    valid = mask.astype(np.float32)
    for threshold in (10.0, 25.0):
        pred_event = ((pred >= threshold) & mask).astype(np.float32)
        truth_event = ((truth >= threshold) & mask).astype(np.float32)
        for window in (3, 9):
            pred_fraction = uniform_filter(pred_event, size=window, mode="constant")
            truth_fraction = uniform_filter(truth_event, size=window, mode="constant")
            use = valid > 0.5
            numerator = np.mean((pred_fraction[use] - truth_fraction[use]) ** 2)
            denominator = np.mean(pred_fraction[use] ** 2 + truth_fraction[use] ** 2)
            result[f"fss_{threshold:g}_w{window}"] = float(1.0 - numerator / denominator) if denominator > 0 else np.nan
    return result


def _plot(fields, titles, grids, map_lats, map_lons, map_land_mask, path):
    fig, axes = plt.subplots(1, 5, figsize=(27.5, 5.5), constrained_layout=True)
    # Same discrete precipitation palette as the main 10-day comparison plot.
    boundaries = [0.0, 10.0, 25.0, 50.0, 100.0, 250.0]
    cmap = ListedColormap(["#FFFFFF", "#36B83B", "#5AA6DF", "#1236CF", "#D017D9"])
    cmap.set_over("#8B003A")
    cmap.set_bad("#FFFFFF")
    norm = BoundaryNorm(boundaries, cmap.N)
    image = None
    high_lats, high_lons = grids[-1]
    lon_ticks = np.linspace(float(np.min(high_lons)), float(np.max(high_lons)), 6)
    lat_ticks = np.linspace(float(np.min(high_lats)), float(np.max(high_lats)), 6)
    for index, (axis, field, title, (lats, lons)) in enumerate(zip(axes, fields, titles, grids)):
        # The first panel is drawn directly on its original coarse grid.  No
        # interpolation is performed; pcolormesh only displays the actual cells.
        image = axis.pcolormesh(lons, lats, field, cmap=cmap, norm=norm, shading="auto")
        # Identical to Stage 1 (plot_results_main.py): draw the 0.5 contour of
        # the land_mask saved by Stage-1 inference on its native grid.
        axis.contour(map_lons, map_lats, map_land_mask.astype(np.float32),
                     levels=[0.5], colors="black", linewidths=0.8, zorder=20)
        axis.set_xlim(float(np.min(high_lons)), float(np.max(high_lons)))
        axis.set_ylim(float(np.min(high_lats)), float(np.max(high_lats)))
        axis.set_title(title)
        axis.set_xticks(lon_ticks)
        axis.set_xticklabels([f"{value:.0f}°E" for value in lon_ticks], rotation=35, ha="right", fontsize=8)
        if index == 0:
            axis.set_yticks(lat_ticks)
            axis.set_yticklabels([f"{value:.0f}°N" for value in lat_ticks], fontsize=8)
        else:
            axis.set_yticks(lat_ticks)
            axis.set_yticklabels([])
        for spine in axis.spines.values():
            spine.set_edgecolor("black")
            spine.set_linewidth(1.0)
    colorbar = fig.colorbar(image, ax=axes, shrink=0.82, label="Precipitation (mm/day)",
                            boundaries=boundaries, ticks=boundaries, extend="max")
    colorbar.ax.set_yticklabels(["0", "10", "25", "50", "100", "≥250"])
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    cfg = config.sr_config
    result_dir = os.environ.get("RVD_SR_RESULT_DIR", cfg["result_dir"])
    train_nc = os.environ.get("RVD_SR_TRAIN_CORRECTION", cfg["train_correction_file"])
    test_nc = os.environ.get("RVD_SR_TEST_CORRECTION", cfg["test_correction_file"])
    checkpoint_path = os.environ.get("RVD_SR_CHECKPOINT", os.path.join(result_dir, "best_model_v3.pt"))
    output_dir = os.path.join(result_dir, "evaluation")
    plot_dir = os.path.join(output_dir, "plots_v3_5col")
    os.makedirs(plot_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = SuperResolutionDataset(train_nc, test_nc, cfg["truth_dir"], "test",
                                     validation_fraction=cfg["validation_fraction"],
                                     lat_range=cfg["lat_range"], lon_range=cfg["lon_range"],
                                     normalization_max_mm=cfg["normalization_max_mm"])
    low_lats = dataset.low_lats[::-1].copy() if dataset.flip_low_lat else dataset.low_lats.copy()
    low_lons = dataset.low_lons[::-1].copy() if dataset.flip_low_lon else dataset.low_lons.copy()
    # Reuse the exact mask written by Stage-1 inference.  This is the source
    # used by plot_results_main.py, so the boundary is identical rather than
    # being inferred again from the high-resolution CRA values.
    with nc.Dataset(test_nc) as stage1_nc:
        if "land_mask" not in stage1_nc.variables:
            raise KeyError(f"{test_nc}: missing Stage-1 land_mask; regenerate SR correction input")
        stage1_land_mask = np.asarray(
            np.ma.filled(stage1_nc.variables["land_mask"][:], 0.0), dtype=np.float32
        ) > 0.5
    if stage1_land_mask.shape != (len(dataset.low_lats), len(dataset.low_lons)):
        raise ValueError(f"Stage-1 land_mask shape {stage1_land_mask.shape} does not match "
                         f"low-resolution grid {(len(dataset.low_lats), len(dataset.low_lons))}")
    if dataset.flip_low_lat:
        stage1_land_mask = stage1_land_mask[::-1, :].copy()
    if dataset.flip_low_lon:
        stage1_land_mask = stage1_land_mask[:, ::-1].copy()
    high_lat_indices = np.abs(low_lats[:, None] - dataset.high_lats[None, :]).argmin(axis=0)
    high_lon_indices = np.abs(low_lons[:, None] - dataset.high_lons[None, :]).argmin(axis=0)
    high_plot_land_mask = stage1_land_mask[np.ix_(high_lat_indices, high_lon_indices)]
    print(f"[SR plot] Stage-1 land_mask reused: {stage1_land_mask.shape}, "
          f"land ratio={stage1_land_mask.mean():.2%}")
    loader = DataLoader(dataset, batch_size=cfg["evaluation_batch_size"], shuffle=False,
                        num_workers=cfg["num_workers"], pin_memory=torch.cuda.is_available(),
                        persistent_workers=cfg["num_workers"] > 0)
    model = TwoStageSuperResolution(
        RegressionConvResNet(cfg["regression_channels"], cfg["regression_blocks"], cfg["regression_residual_scale"]),
        ConditionalDenoisingUNet(cfg["diffusion_channels"], condition_channels=2), cfg["timesteps"],
        cfg["diffusion_residual_clip"], cfg["ddim_sampling_steps"],
        cfg["coarse_residual_fraction"], cfg["coarse_residual_max_fraction"],
        cfg["phase2_gate_min"], cfg["phase2_gate_max"]
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    generator = torch.Generator(device=device).manual_seed(cfg["sampling_seed"])

    output_nc = os.path.join(output_dir, "super_resolution_test_v3.nc")
    csv_path = os.path.join(output_dir, "metrics_by_sample_v3.csv")
    temporary_nc = _temporary_output_path(output_nc)
    temporary_csv = _temporary_output_path(csv_path)
    metric_rows, write_index, visual_count = [], 0, 0
    with nc.Dataset(temporary_nc, "w") as out:
        out.createDimension("sample", None); out.createDimension("lat", len(dataset.high_lats)); out.createDimension("lon", len(dataset.high_lons))
        out.createDimension("low_lat", len(low_lats)); out.createDimension("low_lon", len(low_lons))
        out.createVariable("lat", "f8", ("lat",))[:] = dataset.high_lats
        out.createVariable("lon", "f8", ("lon",))[:] = dataset.high_lons
        out.createVariable("low_lat", "f8", ("low_lat",))[:] = low_lats
        out.createVariable("low_lon", "f8", ("low_lon",))[:] = low_lons
        out.createVariable("stage1_land_mask", "i1", ("low_lat", "low_lon"))[:] = stage1_land_mask.astype(np.int8)
        init_var = out.createVariable("init_date", "S8", ("sample",))
        target_var = out.createVariable("target_date", "S8", ("sample",))
        lead_var = out.createVariable("lead", "i2", ("sample",))
        variables = {name: out.createVariable(name, "f4", ("sample", "lat", "lon"), zlib=True, complevel=2)
                     for name in ("bilinear", "phase1_conv_resnet", "phase2_multiscale_ddim", "truth")}
        low_var = out.createVariable("stage1_low_resolution", "f4", ("sample", "low_lat", "low_lon"),
                                     zlib=True, complevel=2)
        gate_var = out.createVariable("phase2_confidence_gate", "f4", ("sample", "lat", "lon"), zlib=True, complevel=2)
        out.setncattr("pipeline", "low-resolution correction -> 2D Conv-ResNet -> controlled multiscale residual DDIM")
        out.setncattr("test_split", "initialisation date > 2021-12-31; never used for training or validation")

        for batch in tqdm(loader, desc="Conditional diffusion sampling"):
            low = batch["low"].to(device, non_blocking=True)
            truth_norm = batch["truth"].to(device, non_blocking=True)
            baseline_n, phase1_n, phase2_n, gate, _ = model.super_resolve(low, truth_norm.shape[-2:], generator)
            low_mm = dataset.denormalize_tensor(low).cpu().numpy()[:, 0]
            fields_t = [dataset.denormalize_tensor(v).cpu().numpy()[:, 0] for v in (baseline_n, phase1_n, phase2_n, truth_norm)]
            masks = batch["land_mask"].numpy()[:, 0] > 0.5
            batch_size = fields_t[0].shape[0]
            for b in range(batch_size):
                init_date, target_date, lead = batch["init_date"][b], batch["target_date"][b], int(batch["lead"][b])
                init_var[write_index] = init_date.encode(); target_var[write_index] = target_date.encode(); lead_var[write_index] = lead
                for name, values in zip(variables, fields_t):
                    variables[name][write_index] = values[b]
                low_var[write_index] = low_mm[b]
                gate_var[write_index] = gate[b, 0].cpu().numpy()
                row = {"init_date": init_date, "target_date": target_date, "lead": lead}
                for label, values in zip(("bilinear", "phase1", "phase2"), fields_t[:3]):
                    for key, value in _metrics(values[b], fields_t[3][b], masks[b]).items():
                        row[f"{label}_{key}"] = value
                metric_rows.append(row)
                if init_date[4:6] in {"07", "08"} and lead in {1, 3, 5, 7, 10} and visual_count < cfg["max_visualizations"]:
                    shown = [np.where(stage1_land_mask, low_mm[b], np.nan)]
                    shown.extend(np.where(high_plot_land_mask, values[b], np.nan) for values in fields_t)
                    titles = ["(a) Low-resolution input", "(b) Bilinear interpolation",
                              "(c) Phase 1: Conv-ResNet", "(d) Phase 2: Multiscale residual DDIM",
                              "(e) CRA truth"]
                    grids = [(low_lats, low_lons)] + [(dataset.high_lats, dataset.high_lons)] * 4
                    _plot(shown, titles, grids, low_lats, low_lons, stage1_land_mask,
                          os.path.join(plot_dir, f"sr_{init_date}_day{lead:02d}.png"))
                    visual_count += 1
                write_index += 1

    output_nc = _publish_output(temporary_nc, output_nc)
    with open(temporary_csv, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metric_rows[0])); writer.writeheader(); writer.writerows(metric_rows)
    csv_path = _publish_output(temporary_csv, csv_path)
    print(f"[SR] sampled {write_index} test lead-days")
    for label in ("bilinear", "phase1", "phase2"):
        print(f"  {label}: RMSE={np.nanmean([r[f'{label}_rmse'] for r in metric_rows]):.4f}, "
              f"MAE={np.nanmean([r[f'{label}_mae'] for r in metric_rows]):.4f}, "
              f"CSI@5={np.nanmean([r[f'{label}_csi_5'] for r in metric_rows]):.4f}, "
              f"CSI@10={np.nanmean([r[f'{label}_csi_10'] for r in metric_rows]):.4f}, "
              f"CSI@25={np.nanmean([r[f'{label}_csi_25'] for r in metric_rows]):.4f}, "
              f"FSS10(w3/w9)={np.nanmean([r[f'{label}_fss_10_w3'] for r in metric_rows]):.4f}/"
              f"{np.nanmean([r[f'{label}_fss_10_w9'] for r in metric_rows]):.4f}")
    print(f"[SR] NetCDF: {output_nc}\n[SR] metrics: {csv_path}\n[SR] plots: {plot_dir}")


if __name__ == "__main__":
    main()
