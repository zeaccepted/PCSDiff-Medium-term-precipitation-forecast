"""
DeterministicCorrection: 确定性降水订正模型
Step 2 PIMD版：SimpleUNet + 降水强度感知多分支解码
  - backbone: SimpleUNetBackbone（4层UNet + PIMD模块）
  - PIMD创新：3个专精分支（轻雨/中雨/强雨）+ 温度缩放软门控融合
  - 熵正则损失：防止门控塌缩到单分支
  - 输入: 10通道 ensemble forecast
  - 监督: residual_target = truth_norm - baseline_tp_norm
"""
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

import config_server as config
from modules.simple_unet_backbone import SimpleUNetBackbone


def _compute_pad_size(dim_mults, H, W):
    """padding到2^(len(dim_mults))倍数，skip connection尺寸对齐"""
    factor = 2 ** len(dim_mults)
    H_pad = ((H + factor - 1) // factor) * factor
    W_pad = ((W + factor - 1) // factor) * factor
    return H_pad, W_pad


def _pad_tensor(x, H_pad, W_pad):
    """对称padding（reflect模式）"""
    H, W = x.shape[-2], x.shape[-1]
    pad_h = H_pad - H
    pad_w = W_pad - W
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    x_padded = F.pad(x, [pad_left, pad_right, pad_top, pad_bottom], mode='reflect')
    return x_padded, (pad_top, pad_left, H, W)


def _crop_tensor(x, pad_top, pad_left, H, W):
    """与_pad_tensor对称，裁剪回原始尺寸"""
    return x[..., pad_top:pad_top + H, pad_left:pad_left + W]


class DeterministicCorrection(nn.Module):
    """
    Step 2 SimpleUNet + PIMD：降水强度感知的多分支解码
    - 编码器：标准4层UNet
    - 解码器：PIMD多分支融合（3个专精分支 + 软门控）
    - 熵正则损失防止门控塌缩
    """
    def __init__(
        self,
        denoise_fn=None,
        history_fn=None,
        transform_fn=None,
        channels=3,
        timesteps=1000,
        loss_type="l2",
        betas=None,
        pred_mode="pred_true",
        clip_noise=True,
        aux_loss=False,
        spatial_loss_weight=0.0,
        heavy_precip_weight=1.0,
        heavy_precip_threshold=0.35,
        negative_residual_loss_weight=1.0,
        correction_loss_weight=0.0,
        residual_blend_weight=1.0,
        max_abs_residual_norm=1.0,
        positive_residual_clip_value=1.0,
        negative_residual_clip_value=1.0,
        positive_correction_gain=1.0,
        negative_correction_gain=1.0,
        correction_target_weight=0.0,
        fss_loss_weight=0.0,
        use_fss_loss=False,
        fss_thresholds=None,
        fss_scales=None,
        huber_delta=1.0,
        use_ensemble_spread=False,
        physical_space_loss_weight=0.0,
        forecast_min=None,
        forecast_max=None,
        truth_min=None,
        truth_max=None,
        simple_unet_dim=64,
        n_ensemble_channels=10,
        n_residual_channels=1,
        # ===== PIMD新增参数 =====
        use_pimd=True,
        pimd_num_branches=3,
        pimd_temperature=1.0,
        pimd_entropy_loss_weight=0.01,
        **kwargs,
    ):
        super().__init__()
        # 兼容参数
        self.denoise_fn = None
        self.history_fn = None
        self.transform_fn = None
        self.pred_mode = pred_mode
        self.loss_type = loss_type
        self.clip_noise = clip_noise
        self.aux_loss = aux_loss
        self.residual_blend_weight = residual_blend_weight
        self.max_abs_residual_norm = max_abs_residual_norm
        self.positive_residual_clip_value = positive_residual_clip_value
        self.negative_residual_clip_value = negative_residual_clip_value
        self.use_ensemble_spread = use_ensemble_spread
        self.huber_delta = huber_delta

        # PIMD参数
        self.use_pimd = use_pimd
        self.pimd_num_branches = pimd_num_branches
        self.pimd_entropy_loss_weight = pimd_entropy_loss_weight

        # 归一化参数
        self.register_buffer("forecast_min", torch.tensor(forecast_min if forecast_min is not None else -1.0, dtype=torch.float32))
        self.register_buffer("forecast_max", torch.tensor(forecast_max if forecast_max is not None else 1.0, dtype=torch.float32))
        self.register_buffer("truth_min", torch.tensor(truth_min if truth_min is not None else -1.0, dtype=torch.float32))
        self.register_buffer("truth_max", torch.tensor(truth_max if truth_max is not None else 1.0, dtype=torch.float32))

        # SimpleUNet + PIMD backbone
        self.simple_unet = SimpleUNetBackbone(
            n_channels=n_ensemble_channels,
            n_classes=n_residual_channels,
            base_dim=simple_unet_dim,
            use_pimd=use_pimd,
            pimd_num_branches=pimd_num_branches,
            pimd_temperature=pimd_temperature,
        )
        self.dim_mults = self.simple_unet.dim_mults
        self.n_ensemble_channels = n_ensemble_channels
        self.n_residual_channels = n_residual_channels
        self._last_loss_dict = {}

    def decode_residual_prediction(self, residual):
        """Step 1 基线版：纯identity（不做任何正负clip）"""
        return residual

    # ==================================================================
    # 可微TP损失：soft阈值 + 多阈值TP加权
    # ==================================================================
    def _denormalize_forecast(self, x_norm):
        """归一化空间→物理空间(mm)"""
        f_min = self.forecast_min
        f_max = self.forecast_max
        return (x_norm + 1.0) * 0.5 * (f_max - f_min) + f_min

    def _compute_soft_tp_loss(
        self,
        corrected_mm,     # (B, 1, H, W) 订正后预报 mm
        truth_mm,          # (B, 1, H, W) 真值 mm
        land_mask=None,    # (H, W) or (B, H, W)
        lats=None,         # (H,) 纬度数组
    ):
        """可微TP损失：soft阈值近似 + 多阈值加权TP

        Args:
            corrected_mm: (B, 1, H, W) 订正后预报 mm
            truth_mm: (B, 1, H, W) 真值 mm
            land_mask: (H, W) 或 (B, H, W) 陆地掩码
            lats: (H,) 纬度数组，用于加权

        Returns:
            tp_loss: 标量张量（取负号，使TP最大化 = loss最小化）
        """
        B, _, H, W = corrected_mm.shape
        temperature = getattr(config, 'tp_loss_temperature', 10.0)
        thresholds = getattr(config, 'tp_loss_thresholds', [0.1, 1.0, 5.0])

        # 陆地掩码
        if land_mask is not None:
            lm = land_mask.float()
            if lm.dim() == 2:
                lm = lm.unsqueeze(0).unsqueeze(0)
            elif lm.dim() == 3:
                lm = lm.unsqueeze(1)
            if lm.shape[-2:] != (H, W):
                lm = F.interpolate(lm, size=(H, W), mode='nearest')
        else:
            lm = torch.ones(B, 1, H, W, device=corrected_mm.device)

        # 纬度加权：15°-60° 权重=1，其他区域权重=0
        lat_weight = torch.ones(H, device=corrected_mm.device)
        if lats is not None:
            lat_min = getattr(config, 'tp_loss_lat_min', 15.0)
            lat_max = getattr(config, 'tp_loss_lat_max', 60.0)
            lats_t = torch.as_tensor(lats, device=corrected_mm.device, dtype=torch.float32)
            # 如果lats尺寸与H不匹配（数据被裁剪/插值），则将lats插值到H
            if lats_t.shape[0] != H:
                # 用线性插值调整纬度数组长度到H
                lats_t_interp = F.interpolate(
                    lats_t.view(1, 1, -1, 1), size=(H, 1), mode='bilinear', align_corners=False
                ).view(H)
                lats_t = lats_t_interp
            lat_weight = ((lats_t >= lat_min) & (lats_t <= lat_max)).float()
            if lat_weight.sum() < 1.0:
                lat_weight = torch.ones(H, device=corrected_mm.device)

        # 扩展纬度权重到 (1, 1, H, 1) 便于广播
        lat_w = lat_weight.view(1, 1, H, 1)

        # 逐阈值计算TP分量
        tp_j_list = []
        ts_j_list = []
        pc_j_list = []

        for th in thresholds:
            th_tensor = torch.tensor(th, device=corrected_mm.device, dtype=torch.float32)

            # soft阈值: P(x>=t) ≈ sigmoid(τ*(x-t))
            p_corr = torch.sigmoid(temperature * (corrected_mm - th_tensor))
            p_truth = torch.sigmoid(temperature * (truth_mm - th_tensor))

            # soft列联表
            NA_soft = (p_corr * p_truth * lm * lat_w).sum()
            NB_soft = ((1.0 - p_corr) * p_truth * lm * lat_w).sum()
            NC_soft = (p_corr * (1.0 - p_truth) * lm * lat_w).sum()
            ND_soft = ((1.0 - p_corr) * (1.0 - p_truth) * lm * lat_w).sum()

            # TS_j = NA / (NA + NB + NC)
            denom_ts = NA_soft + NB_soft + NC_soft + 1e-8
            TS_j = NA_soft / denom_ts

            # PC_j = Precision = NA / (NA + NC)  — 预报阳性中实际阳性的比例
            # （注：不是Accuracy=(NA+ND)/Total，后者被大量正确无雨ND像素主导没有参考价值）
            denom_pc = NA_soft + NC_soft + 1e-8
            PC_j = NA_soft / denom_pc

            ts_j_list.append(TS_j)
            pc_j_list.append(PC_j)

        # TS_S = mean(TS_j)
        ts_S = torch.stack(ts_j_list).mean()

        # TS_syn_j = (TS_j + TS_S) / (1 + TS_S)
        ts_syn_list = [(ts_j + ts_S) / (1.0 + ts_S) for ts_j in ts_j_list]

        # TP_j = (TS_syn_j + PC_j) / 2
        tp_j_list = [(ts_syn + pc_j) / 2.0 for ts_syn, pc_j in zip(ts_syn_list, pc_j_list)]

        # TP_all = 多阈值平均（等权）
        tp_all = torch.stack(tp_j_list).mean()

        # 损失 = 1 - TP：最大化TP = 最小化(1-TP)
        # （注意：不能用-TP，否则整体损失为负，MSE量级远小于TP loss项，训练失衡）
        tp_loss = 1.0 - tp_all

        return tp_loss, tp_all.detach()

    # ==================================================================
    # 可微FSS Loss（Fractional Skill Score）
    # FSS = 1 - MSE(邻域预报降水比例, 邻域观测降水比例)
    # 窗口大小=3，阈值=1.0mm
    # ==================================================================
    def _compute_soft_fss_loss(
        self,
        corrected_mm,    # (B, 1, H, W) 订正后降水量(mm)
        truth_mm,        # (B, 1, H, W) 真值降水量(mm)
        land_mask=None,  # (H, W) 陆地掩码
        lats=None,       # (H,) 纬度数组
    ):
        """计算可微FSS Loss
        
        使用soft阈值(sigmoid)定义降水事件，使用avg_pool2d计算邻域降水比例，
        然后计算FSS = 1 - MSE(预报邻域比例, 观测邻域比例)
        
        Args:
            corrected_mm: 订正后降水量 (B, 1, H, W) mm
            truth_mm: 真值降水量 (B, 1, H, W) mm
            land_mask: 陆地掩码，可选
            lats: 纬度数组，可选
            
        Returns:
            fss_loss: 1 - FSS (需最小化)
            fss_val: FSS值 (0~1，越大越好)
        """
        window_size = getattr(config, 'fss_loss_window_size', 3)
        threshold = getattr(config, 'fss_loss_threshold', 1.0)
        temperature = getattr(config, 'fss_loss_temperature', 10.0)
        
        B, C, H, W = corrected_mm.shape
        
        # soft阈值: P(x>=t) ≈ sigmoid(τ*(x-t))
        th_tensor = torch.tensor(threshold, device=corrected_mm.device, dtype=torch.float32)
        
        p_corr = torch.sigmoid(temperature * (corrected_mm - th_tensor))
        p_truth = torch.sigmoid(temperature * (truth_mm - th_tensor))
        
        # 邻域降水比例：使用avg_pool2d计算窗口内降水概率均值
        # 这相当于每个窗口内的降水比例（0~1）
        padding = window_size // 2
        frac_corr = F.avg_pool2d(p_corr, kernel_size=window_size, stride=1, padding=padding)
        frac_truth = F.avg_pool2d(p_truth, kernel_size=window_size, stride=1, padding=padding)
        
        # 纬度加权（与TP loss一致）
        lat_weight = torch.ones(H, device=corrected_mm.device)
        if lats is not None:
            lat_min = getattr(config, 'tp_loss_lat_min', 15.0)
            lat_max = getattr(config, 'tp_loss_lat_max', 60.0)
            lats_t = torch.as_tensor(lats, device=corrected_mm.device, dtype=torch.float32)
            if lats_t.shape[0] != H:
                lats_t_interp = F.interpolate(
                    lats_t.view(1, 1, -1, 1), size=(H, 1), mode='bilinear', align_corners=False
                ).view(H)
                lats_t = lats_t_interp
            lat_weight = ((lats_t >= lat_min) & (lats_t <= lat_max)).float()
            if lat_weight.sum() < 1.0:
                lat_weight = torch.ones(H, device=corrected_mm.device)
        
        lat_w = lat_weight.view(1, 1, H, 1)
        
        # 组合掩码：陆地掩码 × 纬度权重
        lm = torch.ones_like(frac_corr)
        if land_mask is not None:
            if land_mask.ndim == 2:
                lm = torch.as_tensor(land_mask, device=corrected_mm.device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            elif land_mask.ndim == 3:
                lm = torch.as_tensor(land_mask, device=corrected_mm.device, dtype=torch.float32).unsqueeze(1)
            else:
                lm = torch.as_tensor(land_mask, device=corrected_mm.device, dtype=torch.float32)
            if lm.shape[-2:] != (H, W):
                lm = F.interpolate(lm, size=(H, W), mode='nearest')
        
        combined_mask = lm * lat_w
        
        # 加权MSE：MSE(预报邻域比例, 观测邻域比例)
        diff = (frac_corr - frac_truth) ** 2
        weighted_diff = diff * combined_mask
        
        # 计算加权平均MSE
        mask_sum = combined_mask.sum() + 1e-8
        mse_val = weighted_diff.sum() / mask_sum
        
        # FSS = 1 - MSE
        fss_val = 1.0 - mse_val
        
        # FSS Loss = 1 - FSS = MSE（最小化MSE即最大化FSS）
        fss_loss = mse_val
        
        return fss_loss, fss_val.detach()

    # ==================================================================
    # 训练损失计算：MSE + PIMD熵正则 + TP Loss(可选) + FSS Loss(可选)
    # ==================================================================
    def _compute_loss(
        self,
        x_start,              # residual_target (B, 1, H, W)
        context=None,
        land_mask=None,
        weight_target=None,
        baseline_target=None,
        correction_target=None,
        ensemble_spread=None,
        ensemble_input=None,  # (B,10,H,W)
        raw_forecast_mm=None,  # (B,1,H,W) 物理空间预报
        raw_truth_mm=None,     # (B,1,H,W) 物理空间真值
        baseline_norm=None,    # (B,1,H,W) tp归一化基准
        lats=None,             # (H,) 纬度数组
    ):
        B, C, H, W = x_start.shape
        if ensemble_input is None:
            raise ValueError("SimpleUNet requires ensemble_input (B,10,H,W)")

        if ensemble_input.shape[-2:] != (H, W):
            ensemble_input = F.interpolate(ensemble_input, size=(H, W), mode='bilinear', align_corners=False)

        # SimpleUNet + PIMD前向
        output = self.simple_unet(ensemble_input)
        # PIMD返回 (residual, gate_weights)，非PIMD返回 (residual, None)
        if isinstance(output, tuple):
            x_recon, gate_weights = output
        else:
            x_recon, gate_weights = output, None

        if x_recon.shape != x_start.shape:
            x_recon = F.interpolate(x_recon, size=x_start.shape[-2:], mode='bilinear', align_corners=False)

        # 主损失：MSE
        if self.loss_type == "l2" or self.loss_type == "mse":
            main_loss = F.mse_loss(x_recon, x_start, reduction='none')
        elif self.loss_type == "l1":
            main_loss = F.l1_loss(x_recon, x_start, reduction='none')
        elif self.loss_type == "huber":
            main_loss = F.huber_loss(x_recon, x_start, reduction='none', delta=self.huber_delta)
        else:
            main_loss = F.mse_loss(x_recon, x_start, reduction='none')

        # land_mask加权
        if land_mask is not None:
            lm = land_mask
            if lm.dim() == 2:
                lm = lm.unsqueeze(0).unsqueeze(0)
            elif lm.dim() == 3:
                lm = lm.unsqueeze(1)
            if lm.shape[-2:] != main_loss.shape[-2:]:
                lm = F.interpolate(lm.float(), size=main_loss.shape[-2:], mode='nearest')
            lm = lm.to(main_loss.device)
            main_loss = (main_loss * lm).sum() / (lm.sum() + 1e-8)
        else:
            main_loss = main_loss.mean()

        # PIMD熵正则损失：防止门控塌缩到单分支
        entropy_loss = torch.tensor(0.0, device=x_start.device)
        if gate_weights is not None and self.pimd_entropy_loss_weight > 0:
            # gate_weights: (B, num_branches, H, W)
            # 计算每个像素的熵，然后取负号（最大化熵 = 鼓励多分支使用）
            eps = 1e-8
            entropy = -(gate_weights * torch.log(gate_weights + eps)).sum(dim=1)
            # 均匀分布的熵 = log(num_branches)
            max_entropy = torch.log(torch.tensor(self.pimd_num_branches, device=x_start.device, dtype=torch.float32))
            # 归一化熵：1.0 = 完全均匀，0.0 = 完全塌缩
            normalized_entropy = entropy.mean() / max_entropy
            # 损失 = (1 - normalized_entropy)，塌缩时损失大
            entropy_loss = (1.0 - normalized_entropy) * self.pimd_entropy_loss_weight

        # ===== TP Loss (可配置) =====
        tp_loss_val = torch.tensor(0.0, device=x_start.device)
        tp_all_val = torch.tensor(0.0, device=x_start.device)
        use_tp = getattr(config, 'use_tp_loss', False)

        # ===== FSS Loss (可配置) =====
        fss_loss_val = torch.tensor(0.0, device=x_start.device)
        fss_all_val = torch.tensor(0.0, device=x_start.device)
        use_fss = getattr(config, 'use_fss_loss', False)

        if use_tp and raw_forecast_mm is not None and raw_truth_mm is not None and baseline_norm is not None:
            # corrected_norm = baseline_norm + x_recon (模型残差预测)
            corrected_norm = baseline_norm + x_recon
            corrected_mm = self._denormalize_forecast(corrected_norm)
            tp_loss_val, tp_all_val = self._compute_soft_tp_loss(
                corrected_mm, raw_truth_mm,
                land_mask=land_mask, lats=lats,
            )

        if use_fss and raw_forecast_mm is not None and raw_truth_mm is not None and baseline_norm is not None:
            # corrected_norm = baseline_norm + x_recon (模型残差预测)
            corrected_norm = baseline_norm + x_recon
            corrected_mm = self._denormalize_forecast(corrected_norm)
            fss_loss_val, fss_all_val = self._compute_soft_fss_loss(
                corrected_mm, raw_truth_mm,
                land_mask=land_mask, lats=lats,
            )

        # 组合loss
        if use_tp and getattr(config, 'tp_loss_blend_with_mse', True):
            tp_weight = getattr(config, 'tp_loss_weight', 0.5)
            total_loss = main_loss + entropy_loss + tp_weight * tp_loss_val
        elif use_tp:
            tp_weight = getattr(config, 'tp_loss_weight', 0.5)
            total_loss = tp_weight * tp_loss_val + entropy_loss
        elif use_fss and getattr(config, 'fss_loss_blend_with_mse', True):
            fss_weight = getattr(config, 'fss_loss_weight', 0.01)
            total_loss = main_loss + entropy_loss + fss_weight * fss_loss_val
        elif use_fss:
            fss_weight = getattr(config, 'fss_loss_weight', 0.01)
            total_loss = fss_weight * fss_loss_val + entropy_loss
        else:
            total_loss = main_loss + entropy_loss

        # 诊断信息
        diag_info = {
            "total_loss": float(total_loss.detach().item()),
            "main_mse_loss": float(main_loss.detach().item()),
            "spatial_consistency_loss": 0.0,
            "negative_residual_weighted_factor": 1.0,
            "SimpleUNet_output_min": float(x_recon.detach().min().item()),
            "SimpleUNet_output_max": float(x_recon.detach().max().item()),
            "target_residual_min": float(x_start.detach().min().item()),
            "target_residual_max": float(x_start.detach().max().item()),
            "pimd_entropy_loss": float(entropy_loss.detach().item()),
            "use_tp_loss": float(use_tp),
            "tp_loss_value": float(tp_loss_val.detach().item()),
            "tp_all_metric": float(tp_all_val.item()),
            "use_fss_loss": float(use_fss),
            "fss_loss_value": float(fss_loss_val.detach().item()),
            "fss_all_metric": float(fss_all_val.item()),
        }

        # PIMD门控诊断
        if gate_weights is not None:
            # 每个分支的平均权重
            for i in range(self.pimd_num_branches):
                diag_info[f"pimd_branch_{i}_avg_weight"] = float(
                    gate_weights[:, i, :, :].mean().detach().item()
                )

        self._last_loss_dict = diag_info
        return total_loss

    # ==================================================================
    # 训练forward入口（逐lead time day独立前向，参数共享）
    # ==================================================================
    def forward(
        self,
        video,                 # residual_target (T, B, 1, H, W)
        cond_video=None,       # 归一化forecast (T, B, 10, H, W)
        land_mask=None,
        weight_video=None,
        baseline_video=None,   # (T, B, 1, H, W) tp归一化基准
        target_video=None,     # (T, B, 1, H, W) truth归一化
        raw_forecast_mm=None,  # (T, B, 1, H, W) 物理空间预报mm
        raw_truth_mm=None,     # (T, B, 1, H, W) 物理空间真值mm
        lats=None,             # (H,) 纬度数组
    ):
        device = video.device

        # 维度统一：保持与trainer.py传参接口兼容
        if video.ndim == 4:
            video = video.unsqueeze(2)            # (T, B, 1, H, W)
        if cond_video is None:
            cond_video = video
        elif cond_video.ndim == 4:
            cond_video = cond_video.unsqueeze(2)  # (T, B, 1, H, W)

        if baseline_video is not None and baseline_video.ndim == 4:
            baseline_video = baseline_video.unsqueeze(2)
        if target_video is not None and target_video.ndim == 4:
            target_video = target_video.unsqueeze(2)
        if weight_video is not None and weight_video.ndim == 4:
            weight_video = weight_video.unsqueeze(2)
        if raw_forecast_mm is not None and raw_forecast_mm.ndim == 4:
            raw_forecast_mm = raw_forecast_mm.unsqueeze(2)
        if raw_truth_mm is not None and raw_truth_mm.ndim == 4:
            raw_truth_mm = raw_truth_mm.unsqueeze(2)

        T, B, C, H, W = video.shape

        # 纬度数组预处理：确保长度与当前数据高度一致（padding前），避免 dataset 原始纬度 != 数据纬度
        if lats is not None:
            lats_t = torch.as_tensor(lats, device=device, dtype=torch.float32)
            if lats_t.shape[0] != H:
                lats_interp = F.interpolate(
                    lats_t.view(1, 1, -1, 1), size=(H, 1), mode='bilinear', align_corners=False
                ).view(H)
                lats = lats_interp.detach().cpu().numpy()

        # Padding：对齐UNet 4层下采样（2^4=16的倍数）
        H_pad, W_pad = _compute_pad_size(self.dim_mults, H, W)
        if H != H_pad or W != W_pad:
            video = _pad_tensor(video.reshape(-1, C, H, W), H_pad, W_pad)[0].reshape(T, B, C, H_pad, W_pad)
            C_cond = cond_video.shape[2]
            cond_video = _pad_tensor(cond_video.reshape(-1, C_cond, H, W), H_pad, W_pad)[0].reshape(T, B, C_cond, H_pad, W_pad)
            if baseline_video is not None:
                C_bl = baseline_video.shape[2]
                baseline_video = _pad_tensor(baseline_video.reshape(-1, C_bl, H, W), H_pad, W_pad)[0].reshape(T, B, C_bl, H_pad, W_pad)
            if target_video is not None:
                C_tg = target_video.shape[2]
                target_video = _pad_tensor(target_video.reshape(-1, C_tg, H, W), H_pad, W_pad)[0].reshape(T, B, C_tg, H_pad, W_pad)
            if raw_forecast_mm is not None:
                C_rf = raw_forecast_mm.shape[2]
                raw_forecast_mm = _pad_tensor(raw_forecast_mm.reshape(-1, C_rf, H, W), H_pad, W_pad)[0].reshape(T, B, C_rf, H_pad, W_pad)
            if raw_truth_mm is not None:
                C_rt = raw_truth_mm.shape[2]
                raw_truth_mm = _pad_tensor(raw_truth_mm.reshape(-1, C_rt, H, W), H_pad, W_pad)[0].reshape(T, B, C_rt, H_pad, W_pad)
            if land_mask is not None:
                if land_mask.ndim == 2:
                    lm4 = land_mask.unsqueeze(0).unsqueeze(0)
                elif land_mask.ndim == 3:
                    lm4 = land_mask.unsqueeze(1)
                else:
                    lm4 = land_mask
                if lm4.shape[-2:] != (H, W):
                    lm4 = F.interpolate(lm4.float(), size=(H, W), mode='nearest')
                lm4_pad = _pad_tensor(lm4, H_pad, W_pad)[0]
                if land_mask.ndim == 2:
                    land_mask = lm4_pad.squeeze()
                elif land_mask.ndim == 3:
                    land_mask = lm4_pad.reshape(land_mask.shape[0], H_pad, W_pad)
                else:
                    land_mask = lm4_pad

            # 纬度数组也需要插值/扩展到padding后的高度H_pad（实际用于TP loss）
            if lats is not None and (H_pad != H):
                lats_t = torch.as_tensor(lats, device=device, dtype=torch.float32)
                if lats_t.shape[0] != H_pad:
                    lats_interp = F.interpolate(
                        lats_t.view(1, 1, -1, 1), size=(H_pad, 1), mode='bilinear', align_corners=False
                    ).view(H_pad)
                    lats = lats_interp.detach().cpu().numpy()

        # 逐day独立前向（Temporal-Independent范式，对齐test目录）
        total_loss = 0.0
        for i in range(T):
            res_target_t = video[i]              # (B, 1, H_pad, W_pad)
            ensemble_input_t = cond_video[i]     # (B, 10, H_pad, W_pad)
            land_mask_i = land_mask

            # TP Loss需要的物理空间数据
            raw_fc_mm_t = raw_forecast_mm[i] if raw_forecast_mm is not None else None
            raw_tr_mm_t = raw_truth_mm[i] if raw_truth_mm is not None else None
            baseline_norm_t = baseline_video[i] if baseline_video is not None else None

            # 输入通道数对齐（期望10通道）
            exp_ch = self.n_ensemble_channels
            C_in = ensemble_input_t.shape[1]
            if C_in < exp_ch:
                pad = torch.zeros(ensemble_input_t.shape[0], exp_ch - C_in,
                                  ensemble_input_t.shape[-2], ensemble_input_t.shape[-1],
                                  device=device, dtype=ensemble_input_t.dtype)
                ensemble_input_t = torch.cat([ensemble_input_t, pad], dim=1)
            elif C_in > exp_ch:
                ensemble_input_t = ensemble_input_t[:, :exp_ch, :, :]

            L = self._compute_loss(
                res_target_t,
                land_mask=land_mask_i,
                ensemble_input=ensemble_input_t,
                raw_forecast_mm=raw_fc_mm_t,
                raw_truth_mm=raw_tr_mm_t,
                baseline_norm=baseline_norm_t,
                lats=lats,
            )
            total_loss = total_loss + L

        return total_loss / T

    # ==================================================================
    # 推理入口
    # ==================================================================
    @torch.no_grad()
    def sample(self, forecast_frames, num_of_frames=3):
        """
        Args:
            forecast_frames: (T, B, 10, H_orig, W_orig)  归一化forecast
            num_of_frames: 需要推理的lead time天数
        Returns:
            residuals: (num_of_frames, B, 1, H_orig, W_orig)  归一化空间的残差预测
        """
        device = next(self.simple_unet.parameters()).device

        if forecast_frames.ndim == 4:
            forecast_frames = forecast_frames.unsqueeze(2)

        T, B, C, H_orig, W_orig = forecast_frames.shape

        H_pad, W_pad = _compute_pad_size(self.dim_mults, H_orig, W_orig)
        need_pad = (H_orig != H_pad) or (W_orig != W_pad)
        if need_pad:
            forecast_frames = _pad_tensor(
                forecast_frames.reshape(-1, C, H_orig, W_orig), H_pad, W_pad
            )[0].reshape(T, B, C, H_pad, W_pad)

        residuals = []
        for i in range(num_of_frames):
            current = forecast_frames[i] if i < T else forecast_frames[-1]
            exp_ch = self.n_ensemble_channels
            C_in = current.shape[1]
            if C_in < exp_ch:
                pad = torch.zeros(current.shape[0], exp_ch - C_in, current.shape[-2], current.shape[-1],
                                  device=device, dtype=current.dtype)
                current = torch.cat([current, pad], dim=1)
            elif C_in > exp_ch:
                current = current[:, :exp_ch, :, :]

            # PIMD返回 (residual, gate_weights)
            output = self.simple_unet(current)
            if isinstance(output, tuple):
                res = output[0]  # 只取残差预测
            else:
                res = output

            if need_pad:
                res = _crop_tensor(res,
                                   (H_pad - H_orig) // 2,
                                   (W_pad - W_orig) // 2,
                                   H_orig, W_orig)
            if res.shape[-2:] != (H_orig, W_orig):
                res = F.interpolate(res, size=(H_orig, W_orig), mode='bilinear', align_corners=False)
            residuals.append(res)

        return torch.stack(residuals, dim=0)
