"""严格时间外推的 TP+MSL+GH 多变量数据加载器。"""
import os

import config_server as config
from .datasets.precipitation import build_valid_samples
from .datasets.multivar_precipitation import MultiVariablePrecipitationDataset


def load_multivar_dataset(data_config, aux_normalization=None):
    data_path = data_config["data_path"]
    if not data_path or not os.path.isdir(data_path):
        raise FileNotFoundError("Set RVD_DATA_ROOT to the preprocessed data directory (see README.md).")
    valid_samples = build_valid_samples()
    common = dict(
        region=data_config.get("region", "global"),
        lat_range=data_config.get("lat_range"), lon_range=data_config.get("lon_range"),
        use_land_mask=data_config.get("use_land_mask", True),
        super_resolution=data_config.get("super_resolution", False),
        target_resolution=data_config.get("target_resolution", (0.25, 0.25)),
        train_end_date="20211231", use_july_only=False,
        use_anomaly=getattr(config, "use_anomaly", False),
        anomaly_scale=getattr(config, "anomaly_scale", "month"),
        anomaly_max_samples_per_group=getattr(config, "anomaly_max_samples_per_group", 15),
    )
    train = MultiVariablePrecipitationDataset(valid_samples, mode="train", aux_normalization=aux_normalization, **common)
    val = MultiVariablePrecipitationDataset(valid_samples, mode="test", aux_normalization=train.get_aux_normalization(), **common)
    val.forecast_min, val.forecast_max = train.forecast_min, train.forecast_max
    val.truth_min, val.truth_max = train.truth_min, train.truth_max
    if getattr(config, "use_anomaly", False):
        val.clim_forecast, val.clim_truth = train.clim_forecast, train.clim_truth
    return train, val
