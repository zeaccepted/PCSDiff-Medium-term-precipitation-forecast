"""Dataset for Conv-ResNet plus conditional-diffusion precipitation SR."""
from __future__ import annotations

import glob
import os
from datetime import datetime, timedelta

import netCDF4 as nc
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .precipitation import convert_accumulated_precip_to_mm, get_var_units


def _date_plus(date_text: str, days: int) -> str:
    return (datetime.strptime(date_text, "%Y%m%d") + timedelta(days=days)).strftime("%Y%m%d")


def _decode_dates(var) -> list[str]:
    values = var[:]
    if values.dtype.kind == "S" and values.ndim == 1:
        return [bytes(v).decode("utf-8").strip("\x00") for v in values]
    if values.dtype.kind in {"S", "U"} and values.ndim > 1:
        return [str(v).strip() for v in nc.chartostring(values)]
    return [str(v) for v in values]


def _coords(ds):
    lat = next((np.asarray(ds.variables[n][:], dtype=np.float64) for n in ("latitude", "lat", "LAT") if n in ds.variables), None)
    lon = next((np.asarray(ds.variables[n][:], dtype=np.float64) for n in ("longitude", "lon", "LON") if n in ds.variables), None)
    if lat is None or lon is None:
        raise KeyError("NetCDF file has no latitude/longitude coordinate")
    return lat, lon


def _read_2d_precip(path: str, lat_range, lon_range):
    with nc.Dataset(path) as ds:
        if "tp" not in ds.variables:
            raise KeyError(f"{path}: missing variable 'tp'")
        var = ds.variables["tp"]
        data = np.asarray(np.ma.filled(var[:], np.nan), dtype=np.float32).squeeze()
        if data.ndim != 2:
            raise ValueError(f"{path}: expected 2-D precipitation after squeeze, got {data.shape}")
        lat, lon = _coords(ds)
        data = convert_accumulated_precip_to_mm(data, get_var_units(var), truth_scale_factor=1.0)
    lat_idx = np.where((lat >= lat_range[0]) & (lat <= lat_range[1]))[0]
    lon_idx = np.where((lon >= lon_range[0]) & (lon <= lon_range[1]))[0]
    if not len(lat_idx) or not len(lon_idx):
        raise ValueError(f"{path}: China crop is outside the truth grid")
    return data[np.ix_(lat_idx, lon_idx)], lat[lat_idx], lon[lon_idx]


class SuperResolutionDataset(Dataset):
    """One lead day per item, with a strictly isolated post-2021 test file."""

    def __init__(self, train_correction_file: str, test_correction_file: str,
                 truth_dir: str, mode: str, validation_fraction: float = 0.1,
                 lat_range=(15.0, 55.0), lon_range=(70.0, 140.0),
                 normalization_max_mm: float = 250.0):
        if mode not in {"train", "val", "test"}:
            raise ValueError("mode must be train, val, or test")
        self.mode, self.truth_dir = mode, truth_dir
        self.lat_range, self.lon_range = lat_range, lon_range
        self.normalization_max_mm = float(normalization_max_mm)
        self.log_scale = float(np.log1p(self.normalization_max_mm))
        self.correction_file = test_correction_file if mode == "test" else train_correction_file
        if not os.path.isfile(self.correction_file):
            raise FileNotFoundError(f"Missing corrected forecast file: {self.correction_file}")

        truth_files = sorted(glob.glob(os.path.join(truth_dir, "CRA_truth_*.nc")))
        self.truth_by_date = {os.path.basename(p).split("_")[-1].split(".")[0]: p for p in truth_files}
        if not self.truth_by_date:
            raise FileNotFoundError(f"No CRA_truth_*.nc files under {truth_dir}")

        with nc.Dataset(self.correction_file) as ds:
            if "corrected" not in ds.variables:
                raise KeyError(f"{self.correction_file}: missing 'corrected'")
            self.init_dates = _decode_dates(ds.variables["time"])
            self.low_lats = np.asarray(ds.variables["lat"][:], dtype=np.float64)
            self.low_lons = np.asarray(ds.variables["lon"][:], dtype=np.float64)
            n_leads = int(ds.variables["corrected"].shape[1])

        init_indices = list(range(len(self.init_dates)))
        if mode != "test":
            split = max(1, int(round(len(init_indices) * (1.0 - validation_fraction))))
            split = min(split, max(1, len(init_indices) - 1))
            init_indices = init_indices[:split] if mode == "train" else init_indices[split:]

        self.records = []
        for init_idx in init_indices:
            init_date = self.init_dates[init_idx]
            for lead_idx in range(n_leads):
                target_date = _date_plus(init_date, lead_idx)
                truth_path = self.truth_by_date.get(target_date)
                if truth_path:
                    self.records.append((init_idx, lead_idx, init_date, target_date, truth_path))
        if not self.records:
            raise RuntimeError(f"No complete correction/truth pairs for mode={mode}")

        first_truth, self.high_lats, self.high_lons = _read_2d_precip(self.records[0][-1], lat_range, lon_range)
        self.high_shape = first_truth.shape
        self.flip_low_lat = len(self.low_lats) > 1 and (self.low_lats[1] > self.low_lats[0]) != (self.high_lats[1] > self.high_lats[0])
        self.flip_low_lon = len(self.low_lons) > 1 and (self.low_lons[1] > self.low_lons[0]) != (self.high_lons[1] > self.high_lons[0])
        print(f"[SR dataset] {mode}: {len(init_indices)} init dates, {len(self.records)} lead-days; "
              f"low={len(self.low_lats)}x{len(self.low_lons)}, truth={self.high_shape}")

    def normalize(self, mm):
        return np.log1p(np.clip(mm, 0.0, self.normalization_max_mm)) / self.log_scale

    def denormalize_tensor(self, value):
        return torch.expm1(value.clamp(0.0, 1.0) * self.log_scale)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        init_idx, lead_idx, init_date, target_date, truth_path = self.records[index]
        with nc.Dataset(self.correction_file) as ds:
            low = np.asarray(np.ma.filled(ds.variables["corrected"][init_idx, lead_idx], np.nan), dtype=np.float32)
        low = np.nan_to_num(low, nan=0.0, posinf=self.normalization_max_mm, neginf=0.0)
        if self.flip_low_lat:
            low = low[::-1, :].copy()
        if self.flip_low_lon:
            low = low[:, ::-1].copy()

        truth, lats, lons = _read_2d_precip(truth_path, self.lat_range, self.lon_range)
        if truth.shape != self.high_shape or not np.allclose(lats, self.high_lats) or not np.allclose(lons, self.high_lons):
            raise ValueError(f"Truth grid changed at {truth_path}")
        land_mask = np.isfinite(truth)
        truth = np.nan_to_num(truth, nan=0.0, posinf=self.normalization_max_mm, neginf=0.0)

        low_norm = torch.from_numpy(self.normalize(low).astype(np.float32))[None]
        truth_norm = torch.from_numpy(self.normalize(truth).astype(np.float32))[None]
        baseline = F.interpolate(low_norm[None], size=self.high_shape, mode="bilinear", align_corners=True)[0]
        return {"low": low_norm, "baseline": baseline, "truth": truth_norm,
                "land_mask": torch.from_numpy(land_mask.astype(np.float32))[None],
                "init_date": init_date, "target_date": target_date, "lead": lead_idx + 1}
