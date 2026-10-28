import os
import tarfile
import numpy as np
from zipfile import ZipFile
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config_server as config


def load_dataset(data_config):
    """
    Downloads and loads a variety of standard benchmark sequence datasets.
    Arguments:
        data_config (dict): dictionary containing data configuration arguments
    Returns:
        tuple of (train, val), each of which is a PyTorch dataset.
    """
    data_path = data_config["data_path"]  # path to data directory
    if not data_path or not os.path.isdir(data_path):
        raise FileNotFoundError("Set RVD_DATA_ROOT to the preprocessed data directory (see README.md).")

    # the name of the dataset to load
    dataset_name = data_config["dataset_name"]
    dataset_name = dataset_name.lower()  # cast dataset_name to lower case
    train = val = None
    
    # 只保留 precipitation 数据集的处理
    if dataset_name == "precipitation":
        from .datasets import PrecipitationDataset, build_valid_samples
        
        # 构建有效样本
        valid_samples = build_valid_samples()
        
        # 获取区域参数
        region = data_config.get("region", "global")
        lat_range = data_config.get("lat_range", None)
        lon_range = data_config.get("lon_range", None)
        use_land_mask = data_config.get("use_land_mask", True)
        
        # 获取超分参数
        super_resolution = data_config.get("super_resolution", False)
        target_resolution = data_config.get("target_resolution", (0.25, 0.25))

        # 距平(Anomaly)参数
        use_anomaly = getattr(config, "use_anomaly", False)
        anomaly_scale = getattr(config, "anomaly_scale", "month")
        anomaly_max_samples = getattr(config, "anomaly_max_samples_per_group", 15)

        # 严格时间外推划分：
        # - 训练：所有目标日期均不晚于 2021-12-31 的样本；
        # - 测试：起报日期从 2022-01-01 起的全部月份样本。
        # 训练样本按最后一个目标日期过滤，避免 2021 年末起报的长时效
        # 样本把 2022 年观测带入训练标签。
        split_date = "20211231"
        train = PrecipitationDataset(valid_samples, mode="train", region=region,
                                     lat_range=lat_range, lon_range=lon_range,
                                     use_land_mask=use_land_mask,
                                     super_resolution=super_resolution,
                                     target_resolution=target_resolution,
                                     train_end_date=split_date,
                                     use_july_only=False,
                                     use_anomaly=use_anomaly,
                                     anomaly_scale=anomaly_scale,
                                     anomaly_max_samples_per_group=anomaly_max_samples)
        val = PrecipitationDataset(valid_samples, mode="test", region=region,
                                   lat_range=lat_range, lon_range=lon_range,
                                   use_land_mask=use_land_mask,
                                   super_resolution=super_resolution,
                                   target_resolution=target_resolution,
                                   train_end_date=split_date,
                                   use_july_only=False,
                                   use_anomaly=use_anomaly,
                                   anomaly_scale=anomaly_scale,
                                   anomaly_max_samples_per_group=anomaly_max_samples)

        # 测试集始终复用训练集的归一化范围，绝不使用测试期数据估计分位数。
        val.forecast_min = train.forecast_min
        val.forecast_max = train.forecast_max
        val.truth_min = train.truth_min
        val.truth_max = train.truth_max

        # 距平模式：测试集共享仅由训练期估计的气候态。
        if use_anomaly:
            val.clim_forecast = train.clim_forecast
            val.clim_truth = train.clim_truth
            print(f"[距平] 测试集已同步训练集气候态 ({len(train.clim_forecast)} 个分组)")
        
        print(f"[归一化] forecast范围: [{train.forecast_min:.2f}, {train.forecast_max:.2f}]")
        print(f"[归一化] truth范围: [{train.truth_min:.2f}, {train.truth_max:.2f}]")
    else:
        raise Exception("Dataset name not found. Only 'precipitation' dataset is supported.")

    return train, val
