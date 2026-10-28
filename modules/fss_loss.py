"""
FSS (Fractions Skill Score) Loss - 降水专用损失函数
FSS衡量在不同空间尺度下降水"分数"的匹配程度

关键优化：使用 sigmoid 软二值化替代硬阈值，保证梯度可反传
"""
import torch
import torch.nn.functional as F


def compute_fss_loss(pred, target, threshold=0.5, scales=[3, 5, 7], temperature=0.02):
    """
    计算 FSS (Fractions Skill Score) Loss

    参数:
        pred: 预测场 [B, C, H, W]，物理空间（mm或归一化后）
        target: 真值场 [B, C, H, W]，与pred同空间
        threshold: 降水阈值，超过该值被认为有降水
        scales: 计算FSS的邻域尺度列表
        temperature: sigmoid 温度参数，越小越接近硬二值化，但梯度更稀疏

    返回:
        fss_loss: 1 - mean(FSS)，范围 [0, 1]
    """
    # 将负值裁剪为0（降水为非负）
    pred_pos = torch.clamp(pred, min=0.0)
    target_pos = torch.clamp(target, min=0.0)

    # 软二值化（带梯度）：使用 sigmoid 近似阶跃函数
    # x > threshold 时 -> 1；x < threshold 时 -> 0
    pred_binary = torch.sigmoid((pred_pos - threshold) / temperature)
    # target 也可以直接硬阈值，因为它不需要梯度；但保持同样的软形式更平滑
    target_binary = torch.sigmoid((target_pos - threshold) / (temperature * 0.5))

    total_fss = 0.0
    valid_scales = 0

    for scale in scales:
        if scale <= 1:
            pred_frac = pred_binary
            target_frac = target_binary
        else:
            kernel_size = scale
            padding = scale // 2
            # count_include_pad=False 避免边缘padding引入的0影响降水比例
            pred_frac = F.avg_pool2d(
                pred_binary, kernel_size=kernel_size, stride=1, padding=padding,
                count_include_pad=False,
            )
            target_frac = F.avg_pool2d(
                target_binary, kernel_size=kernel_size, stride=1, padding=padding,
                count_include_pad=False,
            )

        # FSS = 1 - sum((p - t)^2) / sum(p^2 + t^2)
        numerator = torch.sum((pred_frac - target_frac) ** 2)
        denominator = torch.sum(pred_frac ** 2 + target_frac ** 2)

        if denominator.detach() > 1e-8:
            fss = 1.0 - numerator / (denominator + 1e-10)
            total_fss = total_fss + fss
            valid_scales += 1

    if valid_scales > 0:
        mean_fss = total_fss / valid_scales
    else:
        mean_fss = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

    # FSS 范围异常保护（数值稳定性）
    mean_fss = torch.clamp(mean_fss, 0.0, 1.0)

    fss_loss = 1.0 - mean_fss
    return fss_loss


def compute_fss_loss_continuous(pred, target, thresholds=[0.2, 0.4, 0.6], scales=[3, 7]):
    """
    连续版本的 FSS Loss - 在多个降水阈值上计算FSS并平均

    温度参数根据阈值自适应：阈值越小，温度略放宽以获得更稳定梯度
    """
    total_loss = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
    for threshold in thresholds:
        # 阈值较小时温度稍大，避免几乎全是0时梯度消失
        temp = max(0.01, min(0.05, threshold * 0.15))
        loss = compute_fss_loss(pred, target, threshold=threshold, scales=scales, temperature=temp)
        total_loss = total_loss + loss
    return total_loss / len(thresholds)
