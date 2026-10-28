from torch.utils.data.dataloader import default_collate
import torch


def train_transposed_collate(batch):
    """
    Wrapper around the default collate function to return sequences of PyTorch
    tensors with sequence step as the first dimension and batch index as the
    second dimension.

    Args:
        batch (list): data examples
    """
    batch = filter(lambda img: img is not None, batch)
    collated_batch = default_collate(list(batch))
    transposed_batch = collated_batch.transpose_(0, 1)
    # assert transposed_batch.shape[0] >= 4
    # idx = torch.randint(4, transposed_batch.shape[0] + 1, size=(1,)).item()
    # return transposed_batch[:idx]
    return transposed_batch


def test_transposed_collate(batch):
    """
    Wrapper around the default collate function to return sequences of PyTorch
    tensors with sequence step as the first dimension and batch index as the
    second dimension.

    Args:
        batch (list): data examples
    """
    batch = filter(lambda img: img is not None, batch)
    collated_batch = default_collate(list(batch))
    transposed_batch = collated_batch.transpose_(0, 1)
    return transposed_batch


def precipitation_collate_fn(batch):
    """
    自定义collate函数，用于处理气象数据
    数据格式: (batch_size, 46, lat, lon)
    """
    # 提取数值数据
    forecasts = torch.stack([item["forecast"] for item in batch])
    truths = torch.stack([item["truth"] for item in batch])

    # 提取原始forecast和truth（未经归一化）
    raw_forecasts = torch.stack([item["raw_forecast"] for item in batch])
    raw_truths = torch.stack([item["raw_truth"] for item in batch])

    # 提取原始10个集合成员（未经归一化，用于CRPS计算）
    raw_ensembles = None
    if "raw_ensemble" in batch[0]:
        raw_ensembles = torch.stack([item["raw_ensemble"] for item in batch])
        raw_ensembles = raw_ensembles.transpose(0, 1)  # (T, B, 10, H, W)

    # 距平模式：提取clim_truth（用于推理时加回真值气候态）
    clim_truths = None
    if "clim_truth" in batch[0]:
        clim_truths = torch.stack([item["clim_truth"] for item in batch])
        clim_truths = clim_truths.transpose(0, 1)  # (T, B, 1, H, W)

    # 转置为 (46, batch_size, lat, lon) 格式，与原始代码格式一致
    forecasts = forecasts.transpose(0, 1)
    raw_forecasts = raw_forecasts.transpose(0, 1)
    truths = truths.transpose(0, 1)
    raw_truths = raw_truths.transpose(0, 1)

    # 字符串元数据保留列表形式
    init_dates = [item["init_date"] for item in batch]
    target_dates_list = [item["target_dates"] for item in batch]

    result = {
        "forecast": forecasts,
        "raw_forecast": raw_forecasts,
        "truth": truths,
        "raw_truth": raw_truths,
        "init_date": init_dates,
        "target_dates": target_dates_list
    }

    if raw_ensembles is not None:
        result["raw_ensemble"] = raw_ensembles

    if clim_truths is not None:
        result["clim_truth"] = clim_truths
    
    # 传递陆地掩码（如果存在）
    if "land_mask" in batch[0]:
        land_masks = torch.stack([item["land_mask"] for item in batch])  # (B, H, W) or (B, 1, H, W)
        # land_mask 不需要transpose（空间维度不随时间变化），但要确保维度一致
        result["land_mask"] = land_masks
    
    return result
