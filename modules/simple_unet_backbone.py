"""
SimpleUNet Backbone for Ensemble Precipitation Post-processing
移植自test目录/simple_unet.py：标准4层UNet，带对称skip connection
Step 1: SimpleUNet基线（10通道）
Step 2: PIMD - 降水强度感知的多分支解码模块
  - 3个专精分支：轻雨/中雨/强雨
  - 温度缩放的空间软门控融合
  - 熵正则化防止门控塌缩
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_channels * 2, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


# ==================================================================
# PIMD: Precipitation Intensity-aware Multi-branch Decoder
#   - 核心思想：不同降水强度区域需要不同的订正策略
#   - 3个专精分支：light(轻雨0-2mm), moderate(中雨2-5mm), heavy(强雨5mm+)
#   - 每个分支2层3x3 Conv，参数量可控
#   - 温度缩放软门控融合，熵正则防止塌缩
# ==================================================================

class IntensityAwareBranch(nn.Module):
    """单个雨强专精分支：2层3x3 Conv，输出该分支的残差预测"""
    def __init__(self, channels, hidden_ratio=0.5):
        super().__init__()
        hidden = int(channels * hidden_ratio)
        self.branch = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, kernel_size=3, padding=1),  # 输出1通道残差
        )

    def forward(self, x):
        return self.branch(x)


class PIMDDecoder(nn.Module):
    """
    降水强度感知的多分支解码器
    Args:
        base_channels: 输入特征通道数（=base_dim）
        num_branches: 分支数（默认3：轻雨/中雨/强雨）
        temperature: softmax温度系数（越小越尖锐，越大越平滑）
    Returns:
        residual_fused: (B, 1, H, W) 融合后的残差预测
        gate_weights: (B, num_branches, H, W) 空间门控权重（供熵正则用）
    """
    def __init__(self, base_channels, num_branches=3, temperature=1.0):
        super().__init__()
        self.num_branches = num_branches
        self.temperature = temperature

        # 多个专精分支
        self.branches = nn.ModuleList([
            IntensityAwareBranch(base_channels)
            for _ in range(num_branches)
        ])

        # 空间门控网络：从解码特征生成每个像素的分支选择权重
        # 使用1x1 Conv生成logits，再做softmax得到权重
        self.gate_head = nn.Sequential(
            nn.Conv2d(base_channels, base_channels // 4, kernel_size=1),
            nn.BatchNorm2d(base_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels // 4, num_branches, kernel_size=1),  # 每个分支一个logit
        )

    def forward(self, x):
        """
        Args:
            x: (B, base_channels, H, W) UNet解码特征
        Returns:
            residual_fused: (B, 1, H, W) 融合后的残差
            gate_weights: (B, num_branches, H, W) softmax后的门控权重
        """
        # 1. 各分支独立预测残差
        branch_residuals = []
        for branch in self.branches:
            branch_residuals.append(branch(x))  # 每个: (B, 1, H, W)
        # stack成 (B, num_branches, H, W)
        branch_residuals = torch.cat(branch_residuals, dim=1)

        # 2. 计算空间门控权重
        gate_logits = self.gate_head(x)  # (B, num_branches, H, W)
        # 温度缩放的softmax
        gate_logits = gate_logits / self.temperature
        gate_weights = F.softmax(gate_logits, dim=1)  # (B, num_branches, H, W)

        # 3. 加权融合
        # gate_weights: (B, num_branches, H, W)
        # branch_residuals: (B, num_branches, H, W)
        # 逐元素加权求和
        residual_fused = (gate_weights * branch_residuals).sum(dim=1, keepdim=True)

        return residual_fused, gate_weights


class SimpleUNetBackbone(nn.Module):
    """
    SimpleUNet + PIMD: 降水强度感知的多分支解码
    - 编码器：标准4层UNet下采样
    - 解码器：标准4层UNet上采样 + PIMD多分支融合
    - 输入：10通道 ensemble members
    - 输出：融合后的残差预测
    """
    def __init__(self, n_channels=10, n_classes=1, base_dim=64,
                 use_pimd=True, pimd_num_branches=3, pimd_temperature=1.0):
        super(SimpleUNetBackbone, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.base_dim = base_dim
        self.use_pimd = use_pimd
        self.dim_mults = (1, 2, 4, 8)

        d = base_dim
        # 编码器
        self.inc = DoubleConv(n_channels, d)
        self.down1 = Down(d, d * 2)
        self.down2 = Down(d * 2, d * 4)
        self.down3 = Down(d * 4, d * 8)
        self.down4 = Down(d * 8, d * 8)
        # 解码器
        self.up1 = Up(d * 8, d * 8)
        self.up2 = Up(d * 8, d * 4)
        self.up3 = Up(d * 4, d * 2)
        self.up4 = Up(d * 2, d)

        # PIMD模块：在outc之前插入
        if use_pimd:
            self.pimd = PIMDDecoder(
                base_channels=d,
                num_branches=pimd_num_branches,
                temperature=pimd_temperature,
            )
            self.outc = None
            print(f"[PIMD] 已启用降水强度感知多分支解码: "
                  f"分支数={pimd_num_branches}, 温度={pimd_temperature}")
        else:
            self.pimd = None
            self.outc = OutConv(d, n_classes)

    def forward(self, x):
        """
        Args:
            x: (B, n_channels, H, W) 单lead time day的10个ensemble成员特征
        Returns:
            residual_pred: (B, n_classes, H, W) 融合后的残差预测
            gate_weights: (B, num_branches, H, W) 仅PIMD启用时返回
        """
        # 编码器
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        # 解码器
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)  # (B, base_dim, H, W)

        # PIMD多分支融合
        if self.pimd is not None:
            residual_fused, gate_weights = self.pimd(x)
            return residual_fused, gate_weights
        else:
            logits = self.outc(x)
            return logits, None
