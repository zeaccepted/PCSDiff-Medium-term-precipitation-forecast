"""GPU-only table metric extraction for an existing PCSDiff China NetCDF file.

This script intentionally mirrors the deterministic forecast/corrected fields
used by plot_metrics_by_lead_time.py. It neither trains nor runs inference.
"""
import argparse
from pathlib import Path

import netCDF4 as nc
import numpy as np
import torch
import torch.nn.functional as F

CSI_THRESHOLD = 0.1
FSS_THRESHOLD = 1.0
FSS_WINDOW = 3
TP_THRESHOLD = 1.0
TABLE_DAYS = (3, 5, 7, 10)


def _align_mask(mask, height, width):
    if mask.shape == (height, width):
        return mask
    aligned = F.interpolate(torch.as_tensor(mask, dtype=torch.float32)[None, None],
                            size=(height, width), mode="nearest")
    return aligned[0, 0].numpy() > 0.5


def _daily_scores(observation, prediction, land):
    """Return sample-wise RMSE/ACC/CSI/FSS/TS/PC on the selected GPU."""
    obs = observation[:, land]
    pred = prediction[:, land]
    rmse = torch.sqrt(((pred - obs).square()).mean(dim=1))
    obs_anomaly = obs - obs.mean(dim=1, keepdim=True)
    pred_anomaly = pred - pred.mean(dim=1, keepdim=True)
    acc = (pred_anomaly * obs_anomaly).sum(dim=1) / (
        torch.sqrt((pred_anomaly.square().sum(dim=1) * obs_anomaly.square().sum(dim=1)).clamp_min(1e-10)) + 1e-10
    )

    hit = ((pred >= TP_THRESHOLD) & (obs >= TP_THRESHOLD)).sum(dim=1).float()
    miss = ((pred < TP_THRESHOLD) & (obs >= TP_THRESHOLD)).sum(dim=1).float()
    false_alarm = ((pred >= TP_THRESHOLD) & (obs < TP_THRESHOLD)).sum(dim=1).float()
    correct_negative = ((pred < TP_THRESHOLD) & (obs < TP_THRESHOLD)).sum(dim=1).float()
    ts = hit / (hit + miss + false_alarm).clamp_min(1.0)
    pc = (hit + correct_negative) / (hit + miss + false_alarm + correct_negative).clamp_min(1.0)

    csi_hit = ((pred >= CSI_THRESHOLD) & (obs >= CSI_THRESHOLD)).sum(dim=1).float()
    csi_miss = ((pred < CSI_THRESHOLD) & (obs >= CSI_THRESHOLD)).sum(dim=1).float()
    csi_false = ((pred >= CSI_THRESHOLD) & (obs < CSI_THRESHOLD)).sum(dim=1).float()
    csi = csi_hit / (csi_hit + csi_miss + csi_false).clamp_min(1.0)

    # plot_metrics_by_lead_time.py sets ocean to zero before scipy
    # uniform_filter(..., mode='constant'); avg_pool2d has the same zero padding.
    obs_2d = observation.clone(); pred_2d = prediction.clone()
    obs_2d[:, ~land] = 0.0; pred_2d[:, ~land] = 0.0
    frac_obs = F.avg_pool2d((obs_2d >= FSS_THRESHOLD).float()[:, None], FSS_WINDOW,
                            stride=1, padding=FSS_WINDOW // 2)
    frac_pred = F.avg_pool2d((pred_2d >= FSS_THRESHOLD).float()[:, None], FSS_WINDOW,
                             stride=1, padding=FSS_WINDOW // 2)
    fss = 1.0 - (frac_pred - frac_obs).square().sum(dim=(1, 2, 3)) / (
        frac_pred.square().sum(dim=(1, 2, 3)) + frac_obs.square().sum(dim=(1, 2, 3)) + 1e-8
    )
    return rmse, acc, ts, pc, csi, fss


def main():
    parser = argparse.ArgumentParser(description="Extract PCSDiff table metrics on GPU")
    parser.add_argument("nc_file", nargs="?", default=None, help="existing global_corrected.nc")
    parser.add_argument("--pointer-file", default="results/latest_inference_china_nc.txt",
                        help="project-relative text file that points to the input NC")
    parser.add_argument("--method-label", default="PCSDiff",
                        help="name used for the corrected-method rows in the TXT")
    parser.add_argument("--output-name", default="table_metrics_days_3_5_7_10.txt",
                        help="TXT filename written beside the input NC directory")
    parser.add_argument("--latest-output-pointer", default="results/latest_pcsdiff_table_metrics.txt",
                        help="project-relative pointer file written with the TXT path")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    if args.nc_file:
        nc_path = Path(args.nc_file).resolve()
    else:
        pointer = root / args.pointer_file
        if not pointer.is_file():
            raise FileNotFoundError("No latest China NC pointer. Pass the NC path explicitly.")
        nc_path = Path(pointer.read_text(encoding="utf-8").strip())
    if not nc_path.is_file():
        raise FileNotFoundError(f"NC file not found: {nc_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("A GPU is required for this metric task, but CUDA is unavailable.")
    with nc.Dataset(nc_path) as source:
        truth = source.variables["truth"][:].astype(np.float32)
        forecast = source.variables["forecast"][:].astype(np.float32)
        corrected = source.variables["corrected"][:].astype(np.float32)
        lats = source.variables["lat"][:]
        lons = source.variables["lon"][:]
        mask = source.variables["land_mask"][:] > 0.5 if "land_mask" in source.variables else np.ones(truth.shape[-2:], dtype=bool)

    lat_idx = (lats >= 15.0) & (lats <= 55.0)
    lon_idx = (lons >= 70.0) & (lons <= 140.0)
    truth = np.nan_to_num(truth[:, :, lat_idx, :][:, :, :, lon_idx], nan=0.0)
    forecast = np.nan_to_num(forecast[:, :, lat_idx, :][:, :, :, lon_idx], nan=0.0)
    corrected = np.nan_to_num(corrected[:, :, lat_idx, :][:, :, :, lon_idx], nan=0.0)
    mask = _align_mask(mask[np.ix_(lat_idx, lon_idx)], truth.shape[-2], truth.shape[-1])
    land = torch.as_tensor(mask, device=device, dtype=torch.bool)
    truth = torch.as_tensor(truth, device=device)
    forecast = torch.as_tensor(forecast, device=device)
    corrected = torch.as_tensor(corrected, device=device)

    raw, pcsdiff = {}, {}
    # Calculate every available lead because the project's TP score uses the
    # mean TS across the full NC lead-time range, not only the four table rows.
    for day in range(1, truth.shape[1] + 1):
        index = day - 1
        raw_scores = _daily_scores(truth[:, index], forecast[:, index], land)
        corr_scores = _daily_scores(truth[:, index], corrected[:, index], land)
        raw[day] = [value.mean().item() for value in raw_scores]
        pcsdiff[day] = [value.mean().item() for value in corr_scores]
    if max(TABLE_DAYS) > truth.shape[1]:
        raise ValueError(f"NC contains only {truth.shape[1]} lead times; Day 10 is required.")

    # TP aggregation follows the existing script: daily mean TS/PC first, then
    # the cross-lead TS synthesis used by the project TP definition.
    def finalise(values):
        all_days = sorted(values)
        ts = np.array([values[day][2] for day in all_days])
        pc = np.array([values[day][3] for day in all_days])
        ts_s = ts.mean()
        tp = (ts + ts_s) / (1.0 + ts_s)
        for i, day in enumerate(all_days):
            values[day][2] = (tp[i] + pc[i]) / 2.0
    finalise(raw); finalise(pcsdiff)

    lines = [
        f"{args.method_label} — China test set; metrics extracted from existing inference NetCDF on GPU",
        f"Source NC: {nc_path}",
        "Columns: RMSE(mm/day), ACC, TP, CSI@0.1mm, FSS(threshold=1.0mm, window=3)",
        "Method       Lead(day)    RMSE       ACC        TP         CSI@0.1    FSS_w3",
    ]
    for method, values in (("ECMWF raw", raw), (args.method_label, pcsdiff)):
        for day in TABLE_DAYS:
            rmse, acc, tp, csi, fss = values[day][0], values[day][1], values[day][2], values[day][4], values[day][5]
            lines.append(f"{method:<12} {day:<12d} {rmse:.4f}     {acc:.4f}     {tp:.4f}     {csi:.4f}     {fss:.4f}")
    output = nc_path.parent.parent / args.output_name
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    latest_pointer = root / args.latest_output_pointer
    latest_pointer.parent.mkdir(parents=True, exist_ok=True)
    latest_pointer.write_text(str(output) + "\n", encoding="utf-8")
    print(f"Saved table metrics: {output}")


if __name__ == "__main__":
    main()
