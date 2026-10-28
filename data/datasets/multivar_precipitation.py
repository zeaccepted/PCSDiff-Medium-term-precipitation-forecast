"""TP + 多气象变量辅助数据集。

在严格时间外推切分的基础上，将确定性 2t、MSL、GH、q、sp、u、v、w
拼接到已经归一化的 10 个 TP 集合成员通道之后。辅助变量归一化统计量只由
训练集计算，测试和推理必须复用；ttr 明确不参与本实验。
"""
import glob
import os

import netCDF4 as nc
import numpy as np
import torch

from .precipitation import PrecipitationDataset, compute_region_indices, get_lat_lon_vars
import config_server as config


DEFAULT_AUX_DIRS = {
    variable: os.path.join(config.data_config["data_path"], f"ecmwf_{variable}_nc")
    for variable in ("2t", "msl", "gh", "q", "sp", "u", "v", "w")
}  # RVD_DATA_ROOT 下各气象变量的预处理 NC 目录，见 README.md

# (变量名, 气压层)。None 表示单层/地表变量。ttr 有意不在此列表中。
FEATURE_SPECS = (
    ("2t", None), ("msl", None),
    ("gh", 850), ("gh", 700), ("gh", 500),
    ("q", 1000), ("q", 925), ("q", 850), ("q", 700), ("q", 500),
    ("sp", None),
    ("u", 1000), ("u", 925), ("u", 850), ("u", 700), ("u", 500), ("u", 300),
    ("v", 1000), ("v", 925), ("v", 850), ("v", 700), ("v", 500), ("v", 300),
    ("w", 1000), ("w", 925), ("w", 850), ("w", 700), ("w", 500), ("w", 300),
)

# 文件名沿用 FEATURE_SPECS 的变量名；这里仅兼容 NC 内 2t 的常见别名 t2m。
VARIABLE_ALIASES = {"2t": ("2t", "t2m")}


class MultiVariablePrecipitationDataset(PrecipitationDataset):
    """在 TP 10 通道之后追加 29 个确定性气象通道，总输入 39 通道。"""

    aux_names = tuple(variable if level is None else f"{variable}{level}" for variable, level in FEATURE_SPECS)

    def __init__(self, *args, aux_dirs=None, aux_normalization=None, **kwargs):
        self._is_train = kwargs.get("mode", "train") == "train"
        super().__init__(*args, **kwargs)
        self.aux_dirs = dict(DEFAULT_AUX_DIRS)
        if aux_dirs:
            self.aux_dirs.update(aux_dirs)
        self.aux_normalization = None
        if aux_normalization is not None:
            self.set_aux_normalization(aux_normalization)
        elif self._is_train:
            self.aux_normalization = self._compute_aux_normalization()

    def _find_aux_file(self, init_date, variable):
        directory = self.aux_dirs[variable]
        exact = os.path.join(directory, f"s2s_ecmf_{init_date}_{variable}.nc")
        if os.path.isfile(exact):
            return exact
        # 只允许确定性文件；gh_1 等集合成员不能作为辅助输入。
        candidates = []
        for path in glob.glob(os.path.join(directory, f"*{init_date}*{variable}*.nc")):
            if os.path.splitext(os.path.basename(path))[0].endswith(f"_{variable}"):
                candidates.append(path)
        if len(candidates) != 1:
            raise FileNotFoundError(
                f"Expected one deterministic {variable} file for {init_date} in {directory}; "
                f"found {len(candidates)}: {candidates}"
            )
        return candidates[0]

    @staticmethod
    def _axis(dimensions, candidates):
        for i, name in enumerate(dimensions):
            if name.lower() in candidates:
                return i
        return None

    def _crop_and_orient(self, data, lats, lons):
        """裁剪到 TP 网格，并严格验证经纬度完全对齐。"""
        target_lats = np.asarray(self.forecast_lats_region)
        target_lons = np.asarray(self.forecast_lons_region)
        lats, lons = np.asarray(lats), np.asarray(lons)
        if self.region == "china":
            indices = compute_region_indices(
                lats, lons, (float(target_lats.min()), float(target_lats.max())),
                (float(target_lons.min()), float(target_lons.max())),
            )
            if any(v is None for v in indices):
                raise ValueError("Unable to crop auxiliary field to the TP China domain")
            a, b, c, d = indices
            data, lats, lons = data[:, a:b, c:d], lats[a:b], lons[c:d]
        if data.shape[-2:] != (len(target_lats), len(target_lons)):
            raise ValueError(
                f"Auxiliary grid {data.shape[-2:]} does not match TP grid "
                f"{(len(target_lats), len(target_lons))}; regridding is not implicit."
            )
        if np.allclose(lats[::-1], target_lats):
            data, lats = data[:, ::-1, :], lats[::-1]
        if np.allclose(lons[::-1], target_lons):
            data, lons = data[:, :, ::-1], lons[::-1]
        if not (np.allclose(lats, target_lats) and np.allclose(lons, target_lons)):
            raise ValueError("Auxiliary latitude/longitude coordinates are not aligned with TP coordinates")
        return data

    def _read_field(self, init_date, variable, level_hpa=None):
        path = self._find_aux_file(init_date, variable)
        with nc.Dataset(path, "r") as f:
            data_variable = next((name for name in VARIABLE_ALIASES.get(variable, (variable,)) if name in f.variables), None)
            if data_variable is None:
                raise KeyError(f"{path} has no deterministic variable '{variable}' (available: {list(f.variables)})")
            var = f.variables[data_variable]
            data = np.ma.filled(var[:], np.nan).astype(np.float32)
            dims = tuple(var.dimensions)
            lats, lons, lat_name, lon_name = get_lat_lon_vars(f)
            if lats is None or lons is None:
                raise KeyError(f"{path} has no latitude/longitude coordinate variables")
            time_axis = self._axis(dims, {"time", "forecast_day", "step", "lead_time"})
            lat_axis = dims.index(lat_name) if lat_name in dims else self._axis(dims, {"lat", "latitude"})
            lon_axis = dims.index(lon_name) if lon_name in dims else self._axis(dims, {"lon", "longitude"})
            if time_axis is None or lat_axis is None or lon_axis is None:
                raise ValueError(f"Cannot identify time/lat/lon axes for {variable} in {path}: {dims}")

            if level_hpa is not None:
                level_axis = self._axis(dims, {"plev", "level", "isobaricinpa", "isobaricinhpa", "pressure", "pressure_level"})
                if level_axis is None:
                    raise ValueError(f"GH file {path} has no pressure-level axis: {dims}")
                levels = np.asarray(f.variables[dims[level_axis]][:], dtype=np.float64)
                levels_hpa = levels / 100.0 if np.nanmax(np.abs(levels)) > 2000 else levels
                selected = int(np.argmin(np.abs(levels_hpa - level_hpa)))
                if abs(levels_hpa[selected] - level_hpa) > 1.0:
                    raise ValueError(f"GH pressure level {level_hpa} hPa not found in {levels_hpa.tolist()}")
                data = np.take(data, selected, axis=level_axis)
                dims = tuple(d for i, d in enumerate(dims) if i != level_axis)
                time_axis = self._axis(dims, {"time", "forecast_day", "step", "lead_time"})
                lat_axis = dims.index(lat_name) if lat_name in dims else self._axis(dims, {"lat", "latitude"})
                lon_axis = dims.index(lon_name) if lon_name in dims else self._axis(dims, {"lon", "longitude"})

            data = np.moveaxis(data, (time_axis, lat_axis, lon_axis), (0, 1, 2))
            # 2t / MSL / sp 等单层变量常保留一个 height 或 realization=1 维。
            # 该维没有物理上的额外输入信息，可以安全压缩；若是长度大于 1 的
            # 未识别维度（例如集合成员），必须拒绝，不能静默混入通道。
            if data.ndim > 3:
                extra_shape = data.shape[3:]
                if all(size == 1 for size in extra_shape):
                    data = np.squeeze(data, axis=tuple(range(3, data.ndim)))
                else:
                    raise ValueError(
                        f"Unexpected non-singleton dimensions {extra_shape} for {variable} in {path}; "
                        "deterministic auxiliary input must be (time, lat, lon) after level selection"
                    )
            if data.ndim != 3:
                raise ValueError(f"Expected (time, lat, lon) for {variable}, got {data.shape} from {path}")
            data = self._crop_and_orient(data, lats, lons)
            if data.shape[0] < self.forecast_lead_time:
                raise ValueError(f"{path} has only {data.shape[0]} lead times, need {self.forecast_lead_time}")
            return data[:self.forecast_lead_time]

    @property
    def forecast_lead_time(self):
        from .precipitation import FORECAST_LEAD_TIME
        return FORECAST_LEAD_TIME

    def _read_auxiliary(self, init_date):
        return np.stack([self._read_field(init_date, variable, level) for variable, level in FEATURE_SPECS], axis=1)

    def _compute_aux_normalization(self):
        n_aux = len(self.aux_names)
        mins = np.full(n_aux, np.inf, dtype=np.float64)
        maxs = np.full(n_aux, -np.inf, dtype=np.float64)
        print(f"[多变量归一化] 只使用严格训练集计算 {n_aux} 个辅助通道 min/max...")
        for i, sample in enumerate(self.valid_samples):
            fields = self._read_auxiliary(sample["init_date"])
            mins = np.minimum(mins, np.nanmin(fields, axis=(0, 2, 3)))
            maxs = np.maximum(maxs, np.nanmax(fields, axis=(0, 2, 3)))
            if (i + 1) % 100 == 0 or i + 1 == len(self.valid_samples):
                print(f"  auxiliary normalization: {i + 1}/{len(self.valid_samples)}")
        if not (np.all(np.isfinite(mins)) and np.all(np.isfinite(maxs)) and np.all(maxs > mins)):
            raise ValueError(f"Invalid auxiliary min/max: min={mins}, max={maxs}")
        return {"names": np.asarray(self.aux_names), "min": mins.astype(np.float32), "max": maxs.astype(np.float32)}

    def set_aux_normalization(self, stats):
        names = tuple(str(v) for v in np.asarray(stats["names"]).tolist())
        mins, maxs = np.asarray(stats["min"], dtype=np.float32), np.asarray(stats["max"], dtype=np.float32)
        expected_shape = (len(self.aux_names),)
        if names != self.aux_names or mins.shape != expected_shape or maxs.shape != expected_shape or np.any(maxs <= mins):
            raise ValueError("Invalid auxiliary normalization statistics")
        self.aux_normalization = {"names": np.asarray(self.aux_names), "min": mins, "max": maxs}

    def get_aux_normalization(self):
        if self.aux_normalization is None:
            raise RuntimeError("Auxiliary normalization has not been assigned")
        return self.aux_normalization

    def __getitem__(self, idx):
        result = super().__getitem__(idx)
        fields = self._read_auxiliary(result["init_date"])
        stats = self.get_aux_normalization()
        mins, maxs = stats["min"][None, :, None, None], stats["max"][None, :, None, None]
        fields = np.clip(fields, mins, maxs)
        fields = 2.0 * (fields - mins) / (maxs - mins + 1e-8) - 1.0
        result["forecast"] = torch.cat((result["forecast"], torch.from_numpy(fields.astype(np.float32))), dim=1)
        return result
