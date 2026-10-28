"""Plot national temporal-ACC maps for the raw and corrected forecasts.

The script reads only the already-produced NetCDF result file.  It does not
change the forecast, corrected field, model, or any training/inference path.
For each grid cell, ACC is the Pearson correlation across initialization dates
between lead-time-wise temporal anomalies of the forecast and observation.
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np


CHINA_LAT_RANGE = (15.0, 55.0)
CHINA_LON_RANGE = (70.0, 140.0)
ACC_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "acc_reference",
    ["#b2184c", "#d6604d", "#f4a582", "#fee08b", "#d9ef8b",
     "#66c2a5", "#2c7fb8", "#5e4fa2"],
    N=256,
)


def load_nc(path):
    """Load fields required for ACC without modifying their values."""
    with nc.Dataset(path, "r") as ds:
        required = ("forecast", "corrected", "truth", "lat", "lon")
        missing = [name for name in required if name not in ds.variables]
        if missing:
            raise ValueError(f"NC file is missing required variables: {missing}")
        forecast = np.asarray(ds.variables["forecast"][:], dtype=np.float32)
        corrected = np.asarray(ds.variables["corrected"][:], dtype=np.float32)
        truth = np.asarray(ds.variables["truth"][:], dtype=np.float32)
        lats = np.asarray(ds.variables["lat"][:], dtype=np.float64)
        lons = np.asarray(ds.variables["lon"][:], dtype=np.float64)
        land_mask = None
        if "land_mask" in ds.variables:
            land_mask = np.asarray(ds.variables["land_mask"][:], dtype=np.float32) > 0.5

    if forecast.ndim != 4:
        raise ValueError("Expected forecast/corrected/truth with shape (time, lead_time, lat, lon)")
    return forecast, corrected, truth, lats, lons, land_mask


def crop_national(fields, lats, lons, land_mask):
    """Crop to the full national China plotting extent."""
    lat_sel = (lats >= CHINA_LAT_RANGE[0]) & (lats <= CHINA_LAT_RANGE[1])
    lon_sel = (lons >= CHINA_LON_RANGE[0]) & (lons <= CHINA_LON_RANGE[1])
    if not lat_sel.any() or not lon_sel.any():
        raise ValueError("The NC geographic range does not overlap the China plotting extent")
    cropped = [x[:, :, lat_sel, :][:, :, :, lon_sel] for x in fields]
    if land_mask is not None:
        land_mask = land_mask[np.ix_(lat_sel, lon_sel)]
    return cropped, lats[lat_sel], lons[lon_sel], land_mask


def leadwise_temporal_anomalies(data):
    """Remove each lead's temporal mean at every grid point.

    This makes ACC a temporal statistical skill score over initialization dates,
    rather than the spatial-pattern correlation used in the existing curve.
    """
    valid = np.isfinite(data)
    count = valid.sum(axis=0, keepdims=True)
    total = np.where(valid, data, 0.0).sum(axis=0, keepdims=True)
    mean = np.divide(total, count, out=np.full_like(total, np.nan), where=count > 0)
    return np.where(valid, data - mean, np.nan)


def temporal_acc(forecast_anom, truth_anom):
    """Pearson correlation at each grid point for stacked sample/lead records."""
    valid = np.isfinite(forecast_anom) & np.isfinite(truth_anom)
    n_valid = valid.sum(axis=0)
    numerator = np.where(valid, forecast_anom * truth_anom, 0.0).sum(axis=0)
    f_sq = np.where(valid, forecast_anom ** 2, 0.0).sum(axis=0)
    t_sq = np.where(valid, truth_anom ** 2, 0.0).sum(axis=0)
    denominator = np.sqrt(f_sq * t_sq)
    acc = np.divide(numerator, denominator, out=np.full_like(numerator, np.nan),
                    where=(denominator > 1e-12) & (n_valid >= 3))
    return np.clip(acc, -1.0, 1.0)


def make_windows(n_leads):
    """Use the requested three readable windows for the 10-day experiment."""
    if n_leads >= 10:
        return [(0, 3, "Days 1–3"), (3, 6, "Days 4–6"), (6, 10, "Days 7–10")]
    groups = np.array_split(np.arange(n_leads), min(3, n_leads))
    return [(int(g[0]), int(g[-1]) + 1, f"Days {int(g[0]) + 1}–{int(g[-1]) + 1}") for g in groups]


def plot_acc(raw, corrected, truth, lats, lons, land_mask, output_path):
    raw_anom = leadwise_temporal_anomalies(raw)
    corrected_anom = leadwise_temporal_anomalies(corrected)
    truth_anom = leadwise_temporal_anomalies(truth)
    windows = make_windows(raw.shape[1])

    fig, axes = plt.subplots(2, len(windows), figsize=(5.1 * len(windows), 6.0), squeeze=False)
    norm = mcolors.Normalize(vmin=0.05, vmax=0.80, clip=True)
    last_image = None

    for col, (start, stop, title) in enumerate(windows):
        # Each lead is demeaned separately before stacking, so lead-dependent
        # precipitation climatology cannot inflate correlation in a later window.
        raw_acc = temporal_acc(
            raw_anom[:, start:stop].reshape(-1, raw.shape[2], raw.shape[3]),
            truth_anom[:, start:stop].reshape(-1, truth.shape[2], truth.shape[3]),
        )
        corrected_acc = temporal_acc(
            corrected_anom[:, start:stop].reshape(-1, corrected.shape[2], corrected.shape[3]),
            truth_anom[:, start:stop].reshape(-1, truth.shape[2], truth.shape[3]),
        )

        for row, (label, acc) in enumerate((("RAW", raw_acc), ("Corrected", corrected_acc))):
            ax = axes[row, col]
            shown = acc.copy()
            if land_mask is not None:
                shown[~land_mask] = np.nan
            last_image = ax.pcolormesh(lons, lats, shown, cmap=ACC_CMAP, norm=norm, shading="auto")
            if land_mask is not None:
                ax.contour(lons, lats, land_mask.astype(np.float32), levels=[0.5],
                           colors="#6e6e6e", linewidths=0.45)
            ax.set_xlim(lons.min(), lons.max())
            ax.set_ylim(lats.min(), lats.max())
            ax.set_aspect("equal", adjustable="box")
            if row == 0:
                ax.set_title(title, fontsize=13, fontweight="bold", pad=7)
            if col == 0:
                ax.set_ylabel(label, fontsize=12, fontweight="bold", rotation=90, labelpad=12)
                ax.set_yticks(np.arange(20, 56, 10))
                ax.set_yticklabels([f"{v}°N" for v in np.arange(20, 56, 10)], fontsize=8)
            else:
                ax.set_yticks([])
            if row == len(axes) - 1:
                ax.set_xticks(np.arange(75, 141, 15))
                ax.set_xticklabels([f"{v}°E" for v in np.arange(75, 141, 15)], fontsize=8)
            else:
                ax.set_xticks([])
            for spine in ax.spines.values():
                spine.set_color("#777777")
                spine.set_linewidth(0.55)

    fig.suptitle("Temporal Anomaly Correlation Coefficient (ACC) over China",
                 fontsize=15, fontweight="bold", y=0.97)
    cbar = fig.colorbar(last_image, ax=axes.ravel().tolist(), orientation="horizontal",
                        fraction=0.055, pad=0.10, aspect=40)
    cbar.set_label("ACC", fontsize=11)
    cbar.set_ticks(np.arange(0.05, 0.81, 0.05))
    cbar.ax.tick_params(labelsize=8)
    fig.text(0.5, 0.012,
             "Temporal correlation across initialization dates; anomalies are demeaned separately for each lead time.",
             ha="center", va="bottom", fontsize=8, color="#444444")
    fig.subplots_adjust(left=0.065, right=0.985, top=0.89, bottom=0.18, wspace=0.06, hspace=0.10)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot national raw-versus-corrected temporal ACC maps")
    parser.add_argument("nc_path", help="Path to global_corrected.nc")
    parser.add_argument("--output_dir", required=True, help="Directory for the ACC figure")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    raw, corrected, truth, lats, lons, land_mask = load_nc(args.nc_path)
    (raw, corrected, truth), lats, lons, land_mask = crop_national(
        (raw, corrected, truth), lats, lons, land_mask
    )
    output_path = os.path.join(args.output_dir, "acc_temporal_comparison_china.png")
    plot_acc(raw, corrected, truth, lats, lons, land_mask, output_path)
    print(f"Saved ACC map: {output_path}")


if __name__ == "__main__":
    main()
