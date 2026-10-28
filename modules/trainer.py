import copy
import torch
import os
import shutil
import torch.distributed as dist
from pathlib import Path
from torch.optim import Adam, AdamW
from torch.utils.tensorboard import SummaryWriter
import netCDF4 as nc
import numpy as np
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .utils import cycle, exists
from torch.optim.lr_scheduler import LambdaLR
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config_server as config


class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


# trainer class
class Trainer(object):
    def __init__(
        self,
        rank,
        diffusion_model,
        train_dl,
        val_dl,
        sample_num_of_frame,
        init_num_of_frame,
        scheduler_function,
        ema_decay=0.995,
        train_lr=1e-4,
        train_num_steps=1000000,
        scheduler_checkpoint_step=100000,
        step_start_ema=2000,
        update_ema_every=10,
        save_and_sample_every=1000,
        results_folder="./results",
        tensorboard_dir="./tensorboard_logs/diffusion-video/",
        model_name="model",
        val_num_of_batch=2,
        optimizer="adam",
        dataset=None,
        val_dataset=None,  # 新增：保存验证数据集引用
    ):
        super().__init__()
        self.model = diffusion_model
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = update_ema_every
        self.sample_num_of_frame = sample_num_of_frame
        self.val_num_of_batch = val_num_of_batch

        self.step_start_ema = step_start_ema
        self.save_and_sample_every = save_and_sample_every

        self.train_num_steps = train_num_steps

        self.train_dl_class = train_dl
        self.val_dl_class = val_dl
        self.train_dl = cycle(train_dl)
        self.val_dl = cycle(val_dl)
        
        # 保存数据集对象，用于反归一化
        self.dataset = dataset
        self.val_dataset = val_dataset  # 保存验证数据集引用
        
        # 保存固定的验证数据，确保每次验证使用相同的batch
        self.fixed_val_batches = []
        for i, batch in enumerate(val_dl):
            if i >= val_num_of_batch:
                break
            self.fixed_val_batches.append(batch)
        if optimizer == "adam":
            self.opt = Adam(diffusion_model.parameters(), lr=train_lr)
        elif optimizer == "adamw":
            self.opt = AdamW(diffusion_model.parameters(), lr=train_lr)
        self.scheduler = LambdaLR(self.opt, lr_lambda=scheduler_function)

        self.step = 0
        self.device = rank
        self.init_num_of_frame = init_num_of_frame
        self.scheduler_checkpoint_step = scheduler_checkpoint_step

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name

        if os.path.isdir(tensorboard_dir):
            shutil.rmtree(tensorboard_dir, ignore_errors=True)
        self.writer = SummaryWriter(tensorboard_dir)

        self.train_losses = []
        self.train_steps = []

        self.reset_parameters()

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def _plot_loss_curve(self):
        if len(self.train_losses) < 2:
            return
        
        # 顶刊风格：Times New Roman 字体 + 白底 + 去掉上右边框
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
        plt.rcParams['mathtext.fontset'] = 'stix'
        plt.rcParams['axes.spines.top'] = False
        plt.rcParams['axes.spines.right'] = False
        
        fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
        
        steps = np.array(self.train_steps)
        losses = np.array(self.train_losses, dtype=np.float64)
        
        # ---- 顶刊专业配色方案 ----
        # Raw Loss：极淡蓝灰，只保留细线无marker，不抢眼
        ax.plot(steps, losses, linestyle='-', linewidth=0.4,
                color='#9ecae1', alpha=0.25, label='Raw Loss')
        
        # 主平滑曲线：EMA β=0.995 (~200步)，皇家蓝 + 深线宽，突出趋势
        ema_slow = self._compute_ema(losses, beta=0.995)
        ax.plot(steps, ema_slow, linestyle='-', linewidth=2.2,
                color='#2166ac', alpha=1.0, zorder=5,
                label='EMA (β=0.995, ≈200 steps)')
        
        # 辅助平滑：100步滑动平均，暗橙虚线，与主蓝线互补验证
        if len(losses) >= 100:
            ma_100 = self._compute_moving_avg(losses, window=100)
            ax.plot(steps, ma_100, linestyle=(0, (5, 3)), linewidth=1.6,
                    color='#d94801', alpha=0.95,
                    label='Moving Avg (w=100)')
        
        # 收敛参考线：收敛水平虚线 + 最终值标注
        if len(self.train_losses) >= 100:
            avg_100 = np.mean(self.train_losses[-100:])
            ax.axhline(y=avg_100, color='#555555', linestyle=(0, (3, 3)),
                       linewidth=1.1, alpha=0.85)
            # 在参考线右侧标注数值
            ax.text(steps[-1] * 1.01, avg_100, f'  Final: {avg_100:.4f}',
                    va='center', fontsize=9, color='#333333', fontweight='bold')
        
        # ---- 轴样式 ----
        ax.set_xlabel('Training Step', fontsize=12, fontweight='bold')
        ax.set_ylabel('Loss Value', fontsize=12, fontweight='bold')
        ax.tick_params(axis='both', labelsize=10, direction='out', length=4)
        ax.spines['left'].set_linewidth(1.1)
        ax.spines['bottom'].set_linewidth(1.1)
        
        # 网格：极淡，只画横线
        ax.yaxis.grid(True, linestyle='--', linewidth=0.4, alpha=0.55, color='#cccccc')
        ax.xaxis.grid(False)
        ax.set_axisbelow(True)
        
        # Y轴自动收紧：从 0 到 loss 最大值的 1.05 倍
        y_min = 0.0
        y_max = min(np.max(losses) * 1.05, np.percentile(losses, 99) * 1.15)
        ax.set_ylim([y_min, y_max])
        ax.set_xlim([steps[0], steps[-1]])
        
        # 图例：右上角无边框，白底
        legend = ax.legend(fontsize=9, loc='upper right', frameon=True,
                           fancybox=False, edgecolor='#dddddd',
                           borderpad=0.5, labelspacing=0.4)
        legend.get_frame().set_alpha(0.95)
        
        plt.tight_layout(pad=0.8)
        loss_plot_path = str(self.results_folder / "training_loss.png")
        plt.savefig(loss_plot_path, dpi=300, bbox_inches='tight',
                    facecolor='white')
        plt.close()
        
        # 恢复默认
        plt.rcParams.update(plt.rcParamsDefault)
        
        print(f"[Step {self.step}] Loss curve saved to: {loss_plot_path}")
    
    @staticmethod
    def _compute_ema(data, beta=0.99):
        """计算带偏差修正的指数移动平均 (Bias-Corrected EMA)"""
        n = len(data)
        ema = np.zeros(n, dtype=np.float64)
        ema_uncorrected = 0.0
        for i in range(n):
            ema_uncorrected = beta * ema_uncorrected + (1 - beta) * data[i]
            # 偏差修正：除以 (1 - beta^(i+1))，消除前期偏移
            bias_correction = 1.0 - beta ** (i + 1)
            ema[i] = ema_uncorrected / bias_correction if bias_correction > 1e-8 else ema_uncorrected
        return ema
    
    @staticmethod
    def _compute_moving_avg(data, window=100):
        """计算简单滑动平均（顶刊常用100窗口，更稳定）"""
        n = len(data)
        result = np.zeros(n, dtype=np.float64)
        for i in range(n):
            start = max(0, i - window + 1)
            result[i] = np.mean(data[start:i+1])
        return result

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
        else:
            self.ema.update_model_average(self.ema_model, self.model)

    def save(self):
        if self.device == 0:
            data = {
                "step": self.step,
                "model": self.model.module.state_dict(),
                "ema": self.ema_model.module.state_dict(),
            }
            idx = (self.step // self.save_and_sample_every) % 3
            torch.save(data, str(self.results_folder / f"{self.model_name}_{idx}.pt"))

    def load(self, idx=0, load_step=True):
        data = torch.load(
            str(self.results_folder / f"{self.model_name}_{idx}.pt"),
            map_location=lambda storage, loc: storage,
        )

        if load_step:
            self.step = data["step"]
        
        model_state_dict = data["model"]
        ema_state_dict = data["ema"]
        
        # 处理time_pos_emb的尺寸不匹配（从46天改为10天时）
        for state_dict in [model_state_dict, ema_state_dict]:
            for key in list(state_dict.keys()):
                if 'time_pos_emb' in key:
                    old_emb = state_dict[key]
                    if old_emb.shape[0] > 10:
                        state_dict[key] = old_emb[:10]
                        print(f"[Checkpoint加载] 截断 {key}: {old_emb.shape} -> {state_dict[key].shape}")
        
        self.model.module.load_state_dict(model_state_dict)
        self.ema_model.module.load_state_dict(ema_state_dict)

    def _get_output_lat_lon(self, height, width):
        if (
            hasattr(self, "dataset")
            and self.dataset is not None
        ):
            if (
                getattr(self.dataset, "super_resolution", False)
                and getattr(self.dataset, "sr_target_lats", None) is not None
                and getattr(self.dataset, "sr_target_lons", None) is not None
            ):
                lats = np.asarray(self.dataset.sr_target_lats, dtype=np.float32)
                lons = np.asarray(self.dataset.sr_target_lons, dtype=np.float32)
                if len(lats) == height and len(lons) == width:
                    return lats, lons
            elif (
                getattr(self.dataset, "forecast_lats_region", None) is not None
                and getattr(self.dataset, "forecast_lons_region", None) is not None
            ):
                lats = np.asarray(self.dataset.forecast_lats_region, dtype=np.float32)
                lons = np.asarray(self.dataset.forecast_lons_region, dtype=np.float32)
                if len(lats) == height and len(lons) == width:
                    return lats, lons

        lats = np.linspace(-90, 90, height, dtype=np.float32)
        lons = np.linspace(-180, 180, width, dtype=np.float32)
        return lats, lons

    def _set_lat_lon_attrs(self, lat_var, lon_var):
        lat_var.units = "degrees_north"
        lat_var.standard_name = "latitude"
        lat_var.long_name = "latitude"
        lat_var.axis = "Y"
        lon_var.units = "degrees_east"
        lon_var.standard_name = "longitude"
        lon_var.long_name = "longitude"
        lon_var.axis = "X"

    def _forecast_to_truth_space(self, forecast):
        """线性归一化下forecast和truth已在同一空间，直接返回"""
        return forecast

    def _residual_clip_value(self):
        return float(getattr(config, "residual_clip_value", 0.8))

    def _correction_gain(self):
        return float(getattr(config, "correction_gain", 1.0))

    def _positive_residual_clip_value(self):
        return float(getattr(config, "positive_residual_clip_value", self._residual_clip_value()))

    def _negative_residual_clip_value(self):
        return float(getattr(config, "negative_residual_clip_value", self._residual_clip_value()))

    def _positive_correction_gain(self):
        return float(getattr(config, "positive_correction_gain", 1.0))

    def _negative_correction_gain(self):
        return float(getattr(config, "negative_correction_gain", 1.0))

    def _residual_vis_scale(self):
        return max(
            self._positive_residual_clip_value(),
            self._negative_residual_clip_value(),
            1e-8,
        )

    def _encode_residual_target(self, residual):
        pos_clip = self._positive_residual_clip_value()
        neg_clip = self._negative_residual_clip_value()
        positive = residual.clamp(min=0.0, max=pos_clip) / pos_clip
        negative = residual.clamp(min=-neg_clip, max=0.0) / neg_clip
        return (positive + negative).clamp(-1.0, 1.0)

    def _decode_residual_prediction(self, residual):
        positive = residual.clamp(min=0.0) * self._positive_residual_clip_value()
        negative = residual.clamp(max=0.0) * self._negative_residual_clip_value()
        decoded = positive + negative
        max_abs = float(getattr(config, "max_abs_residual_norm", 1.0))
        return decoded.clamp(-max_abs, max_abs)

    def _stabilize_residual_correction(self, forecast_truth_space, residual_decoded):
        residual_decoded = np.asarray(residual_decoded, dtype=np.float32)

        if bool(getattr(config, "remove_negative_residual_bias", True)):
            axes = tuple(range(1, residual_decoded.ndim))
            residual_mean = np.mean(residual_decoded, axis=axes, keepdims=True)
            negative_mean = np.minimum(residual_mean, 0.0)
            strength = float(getattr(config, "negative_bias_remove_strength", 0.75))
            residual_decoded = residual_decoded - strength * negative_mean

        if bool(getattr(config, "preserve_strong_forecast", False)):
            protect_threshold = float(getattr(config, "protect_forecast_threshold", 0.35))
            negative_limit = float(getattr(config, "protected_negative_residual_limit", 0.03))
            strong_forecast_mask = forecast_truth_space >= protect_threshold
            residual_decoded = np.where(
                strong_forecast_mask,
                np.maximum(residual_decoded, -negative_limit),
                residual_decoded,
            )
        max_abs_residual = float(getattr(config, "max_abs_residual_norm", 1.0))
        residual_decoded = np.clip(residual_decoded, -max_abs_residual, max_abs_residual)
        return residual_decoded

    def _clip_corrected_norm(self, corrected_norm):
        clip_min = float(getattr(config, "prediction_norm_clip_min", -1.0))
        clip_max = float(getattr(config, "prediction_norm_clip_max", 1.0))
        return np.clip(corrected_norm, clip_min, clip_max)

    def _protect_forecast_intensity(self, forecast_truth_space, corrected_norm):
        if not bool(getattr(config, "preserve_strong_forecast", False)):
            return corrected_norm

        protect_threshold = float(getattr(config, "protect_forecast_threshold", 0.35))
        allowed_drop = float(getattr(config, "strong_forecast_allowed_norm_drop", 0.05))
        retain_ratio = float(getattr(config, "strong_forecast_retain_ratio", 0.9))
        strong_forecast_mask = forecast_truth_space >= protect_threshold
        floor_by_drop = forecast_truth_space - allowed_drop
        floor_by_ratio = forecast_truth_space * retain_ratio
        protected_floor = np.maximum(floor_by_drop, floor_by_ratio)
        return np.where(strong_forecast_mask, np.maximum(corrected_norm, protected_floor), corrected_norm)

    def _get_residual_blend_weight(self, step):
        return float(getattr(config, "residual_blend_weight", 1.0))

    def _apply_residual_correction(self, forecast_truth_space, residual_prediction, step=None):
        """Simplified residual correction: corrected = forecast + blend_weight * decoded_residual
        与test目录完全一致，去掉所有stabilize/protect/clip中间步骤。
        """
        # Step 1: Decode model output to residual (this is needed - it converts the model's
        # sigmoid-like output to actual residual values)
        raw_residual_decoded = self._decode_residual_prediction(residual_prediction).detach().cpu().numpy()
        residual_decoded = raw_residual_decoded.copy()
        
        # Step 1.5: Apply residual blend weight (与infer_global保持一致，防止配置生效不一致)
        blend_w = self._get_residual_blend_weight(step)
        if blend_w != 1.0:
            residual_decoded = residual_decoded * blend_w
        
        # Step 2: Simple addition - no stabilization, no protection, no clipping
        # 与test目录一致: corrected = forecast + residual
        if forecast_truth_space.ndim == 4:
            if forecast_truth_space.shape[1] > 1:
                corrected_norm = forecast_truth_space + residual_decoded
            else:
                corrected_norm = forecast_truth_space + residual_decoded[:, np.newaxis, :, :]
        else:
            if residual_decoded.ndim == 4 and residual_decoded.shape[1] == 1:
                residual_decoded = residual_decoded[:, 0, :, :]
            corrected_norm = forecast_truth_space + residual_decoded
        
        return corrected_norm, residual_decoded, raw_residual_decoded

    def _zero_protect_correction(self, forecast_denorm, corrected_denorm, raw_ensemble=None):
        """零值保护函数（修复MAE/CRPS/FSS恶化核心）

        原理：RMSE好但MAE差的典型原因是大量无雨格点(raw<0.1mm)被模型叠加正残差，
             生成0.1~1mm虚假小降水。这些小虚假值对平方误差(RMSE)贡献小但被线性累加到MAE；
             同步影响集合分布（CRPS）和降水区域空间结构（FSS）。

             CRPS/MAE根因：模型学习的残差基于tp确定性预报（训练基线），
             但对集合成员直接加同一残差。若tp与ensemble_mean有系统性偏差，
             则 corrected_members_mean = ensemble_mean + residual ≠ truth，
             导致CRPS和MAE系统性恶化。

             修复：对集合成员应用残差时加上(tp - ensemble_mean)补偿，保证
             mean(corrected_members) = ensemble_mean + residual + tp - ensemble_mean
             = tp + residual ≈ truth

        Args:
            forecast_denorm: (T, H, W) 原始预报 mm（raw_forecast即tp单成员）
            corrected_denorm: (T, H, W) 订正后预报 mm
            raw_ensemble: (T, 10, H, W) 原始集合成员 mm（可选）

        Returns:
            (corrected_denorm, corrected_members_denorm)
        """
        residual_mm = corrected_denorm - forecast_denorm  # (T, H, W)
        T_days = residual_mm.shape[0]

        # ========== 残差门控 v5（与infer_global完全一致）==========
        # ===== 预计算"弱降水通行区"mask（v4机制保留）=====
        if raw_ensemble is not None:
            ensemble_mean_2d = np.mean(raw_ensemble, axis=1)  # (T, H, W)
            weak_rain_candidate = (ensemble_mean_2d > 0.1) | (forecast_denorm > 0.1)
        else:
            ensemble_mean_2d = None
            weak_rain_candidate = forecast_denorm > 0.1

        # ===== 门控1：三档阈值零值保护 + 通行豁免（v4不变）=====
        zero_T1 = forecast_denorm < 0.2
        block_strict = zero_T1 & (~weak_rain_candidate) & (residual_mm > 0)
        residual_mm = np.where(block_strict, 0.0, residual_mm)
        limit_weak = zero_T1 & weak_rain_candidate & (residual_mm > 0.3)
        residual_mm = np.where(limit_weak, 0.3, residual_mm)

        zero_T2 = (forecast_denorm >= 0.2) & (forecast_denorm < 0.35)
        limit_T2_normal = zero_T2 & (~weak_rain_candidate) & (residual_mm > 0.5)
        residual_mm = np.where(limit_T2_normal, 0.5, residual_mm)
        limit_T2_weak = zero_T2 & weak_rain_candidate & (residual_mm > 0.8)
        residual_mm = np.where(limit_T2_weak, 0.8, residual_mm)

        strict_mask = forecast_denorm < 0.05
        tmp_corrected = forecast_denorm + residual_mm
        residual_mm = np.where(strict_mask & (tmp_corrected < 0.1), -forecast_denorm, residual_mm)

        # ===== 门控2 v5：预报偏高/偏低区域残差抑制（阈值0.2mm，系数×0.42/×0.55）=====
        if ensemble_mean_2d is not None:
            over_mask = forecast_denorm > (ensemble_mean_2d + 0.2)  # 0.3→0.2
            damp_over_coef = np.where(weak_rain_candidate & over_mask, 0.55,  # 0.65→0.55
                             np.where(over_mask, 0.42, 1.0))  # 0.5→0.42
            residual_mm = np.where(over_mask & (residual_mm > 0),
                                   residual_mm * damp_over_coef, residual_mm)

            under_mask = forecast_denorm < (ensemble_mean_2d - 0.3)
            residual_mm = np.where(under_mask & (residual_mm < 0),
                                   residual_mm * 0.7, residual_mm)

        # ===== 门控3 v5：中低雨强正残差阻尼（微收紧）=====
        damp_05_2 = (forecast_denorm >= 0.5) & (forecast_denorm < 2.0) & (residual_mm > 0)
        damp_coef_05_2 = np.where(weak_rain_candidate & damp_05_2, 0.72, 0.60)  # 0.8/0.65→0.72/0.60
        residual_mm = np.where(damp_05_2, residual_mm * damp_coef_05_2, residual_mm)

        damp_2_5 = (forecast_denorm >= 2.0) & (forecast_denorm < 5.0) & (residual_mm > 0)
        damp_coef_2_5 = np.where(weak_rain_candidate & damp_2_5, 0.90, 0.82)  # 0.95/0.85→0.90/0.82
        residual_mm = np.where(damp_2_5, residual_mm * damp_coef_2_5, residual_mm)

        # ===== 门控4（新增）：长时效正残差衰减 =====
        lt_decay = np.ones((T_days, 1, 1), dtype=np.float64)
        for t in range(T_days):
            if t >= 7:
                lt_decay[t, 0, 0] = 0.92
            elif t >= 5:
                lt_decay[t, 0, 0] = 0.96
        pos_res_mask = residual_mm > 0
        residual_mm = np.where(pos_res_mask, residual_mm * lt_decay, residual_mm)

        # 应用残差并非负
        corrected_denorm = forecast_denorm + residual_mm
        corrected_denorm = np.maximum(corrected_denorm, 0.0)

        # ===== 门控5（新增）：空间孤立弱降水碎点过滤 =====
        cand_isolated = (corrected_denorm >= 0.05) & (corrected_denorm <= 1.5)
        if np.any(cand_isolated):
            try:
                from scipy.ndimage import uniform_filter
                has_rain_arr = (corrected_denorm > 0.1).astype(np.float32)
                nc = uniform_filter(has_rain_arr, size=(1, 3, 3), mode='constant') * 9.0
                real_neighbors = nc - has_rain_arr
                isolated_mask = cand_isolated & (real_neighbors <= 1.0)
                corrected_denorm = np.where(isolated_mask, 0.0, corrected_denorm)
            except Exception:
                pass

        corrected_members_denorm = None
        if raw_ensemble is not None:
            # ---- 集合成员同步五重门控 v5 ----
            residual_mm_final = corrected_denorm - forecast_denorm
            residual_4d = residual_mm_final[:, np.newaxis, :, :]
            T_members = raw_ensemble.shape[0]
            N_members = raw_ensemble.shape[1]

            # ===== CRPS/MAE修复：tp vs ensemble_mean补偿 =====
            ensemble_mean = np.mean(raw_ensemble, axis=1, keepdims=True)
            tp_ensemble_bias = forecast_denorm[:, np.newaxis, :, :] - ensemble_mean
            adjusted_residual_4d = residual_4d + tp_ensemble_bias

            corrected_members_denorm = raw_ensemble + adjusted_residual_4d
            corrected_members_denorm = np.maximum(corrected_members_denorm, 0.0)

            member_ens_mean_5d = np.broadcast_to(ensemble_mean, raw_ensemble.shape)
            mem_weak_cand = (member_ens_mean_5d > 0.1) | (raw_ensemble > 0.1)

            # 门控1：成员三档零值保护 + 通行豁免（不变）
            member_T1 = raw_ensemble < 0.2
            mem_T1_res = corrected_members_denorm - raw_ensemble
            mem_T1_strict = member_T1 & (~mem_weak_cand) & (mem_T1_res > 0)
            corrected_members_denorm = np.where(mem_T1_strict, raw_ensemble, corrected_members_denorm)
            mem_T1_weak = member_T1 & mem_weak_cand & (mem_T1_res > 0.3)
            corrected_members_denorm = np.where(mem_T1_weak, raw_ensemble + 0.3, corrected_members_denorm)

            member_T2 = (raw_ensemble >= 0.2) & (raw_ensemble < 0.35)
            mem_T2_res = corrected_members_denorm - raw_ensemble
            mem_T2_norm = member_T2 & (~mem_weak_cand) & (mem_T2_res > 0.5)
            corrected_members_denorm = np.where(mem_T2_norm, raw_ensemble + 0.5, corrected_members_denorm)
            mem_T2_weak = member_T2 & mem_weak_cand & (mem_T2_res > 0.8)
            corrected_members_denorm = np.where(mem_T2_weak, raw_ensemble + 0.8, corrected_members_denorm)

            strict_member = raw_ensemble < 0.05
            corrected_members_denorm = np.where(
                strict_member & (corrected_members_denorm < 0.1), 0.0, corrected_members_denorm
            )

            # 门控2 v5：成员级偏高抑制（阈值0.2mm，×0.42/×0.55）
            mem_res_post = corrected_members_denorm - raw_ensemble
            mem_over = raw_ensemble > (member_ens_mean_5d + 0.2)
            damp_over_mem = np.where(mem_weak_cand & mem_over, 0.55,
                            np.where(mem_over, 0.42, 1.0))
            mem_damp_over = mem_over & (mem_res_post > 0)
            corrected_members_denorm = np.where(mem_damp_over,
                                                raw_ensemble + mem_res_post * damp_over_mem,
                                                corrected_members_denorm)
            mem_under = raw_ensemble < (member_ens_mean_5d - 0.3)
            mem_res_post2 = corrected_members_denorm - raw_ensemble
            mem_damp_under = mem_under & (mem_res_post2 < 0)
            corrected_members_denorm = np.where(mem_damp_under,
                                                raw_ensemble + mem_res_post2 * 0.7,
                                                corrected_members_denorm)
            # 门控3 v5：成员级中低雨强阻尼（微收紧）
            mem_res_post3 = corrected_members_denorm - raw_ensemble
            damp_mem_05_2 = (raw_ensemble >= 0.5) & (raw_ensemble < 2.0) & (mem_res_post3 > 0)
            damp_mem_c1 = np.where(mem_weak_cand & damp_mem_05_2, 0.72, 0.60)
            corrected_members_denorm = np.where(damp_mem_05_2,
                                                raw_ensemble + mem_res_post3 * damp_mem_c1,
                                                corrected_members_denorm)
            mem_res_post4 = corrected_members_denorm - raw_ensemble
            damp_mem_2_5 = (raw_ensemble >= 2.0) & (raw_ensemble < 5.0) & (mem_res_post4 > 0)
            damp_mem_c2 = np.where(mem_weak_cand & damp_mem_2_5, 0.90, 0.82)
            corrected_members_denorm = np.where(damp_mem_2_5,
                                                raw_ensemble + mem_res_post4 * damp_mem_c2,
                                                corrected_members_denorm)
            # 门控4：成员级长时效正残差衰减
            lt_decay_4d = np.ones((T_members, 1, 1, 1), dtype=np.float64)
            for t in range(T_members):
                if t >= 7:
                    lt_decay_4d[t, 0, 0, 0] = 0.92
                elif t >= 5:
                    lt_decay_4d[t, 0, 0, 0] = 0.96
            mem_res_lt = corrected_members_denorm - raw_ensemble
            pos_mem_mask = mem_res_lt > 0
            corrected_members_denorm = np.where(pos_mem_mask,
                                                raw_ensemble + mem_res_lt * lt_decay_4d,
                                                corrected_members_denorm)
            corrected_members_denorm = np.maximum(corrected_members_denorm, 0.0)

            # 门控5：成员级空间孤立碎点过滤（逐成员）
            try:
                from scipy.ndimage import uniform_filter
                for m in range(N_members):
                    mem_slice = corrected_members_denorm[:, m, :, :]
                    cand_mem = (mem_slice >= 0.05) & (mem_slice <= 1.5)
                    if np.any(cand_mem):
                        has_rain_m = (mem_slice > 0.1).astype(np.float32)
                        nc_m = uniform_filter(has_rain_m, size=(1, 3, 3), mode='constant') * 9.0
                        real_n_m = nc_m - has_rain_m
                        isolated_m = cand_mem & (real_n_m <= 1.0)
                        mem_slice = np.where(isolated_m, 0.0, mem_slice)
                        corrected_members_denorm[:, m, :, :] = mem_slice
            except Exception:
                pass
            corrected_members_denorm = np.maximum(corrected_members_denorm, 0.0)

        return corrected_denorm, corrected_members_denorm

    def train(self):

        while self.step < self.train_num_steps:
            if (self.step >= self.scheduler_checkpoint_step) and (self.step != 0):
                self.scheduler.step()
            
            # 获取数据
            data = next(self.train_dl)
            # Meteorological correction: learn the residual in normalized truth space,
            # conditioned on the original forecast sequence.
            land_mask = None
            cond_data = None
            if isinstance(data, dict):
                cond_data = data["forecast"].to(self.device)
                truth_data = data["truth"].to(self.device)
                if "land_mask" in data:
                    land_mask = data["land_mask"].to(self.device)
            else:
                truth_data = data.to(self.device)

            if len(truth_data.shape) == 4:
                truth_data = truth_data.unsqueeze(2)
            # cond_data: (T, B, n_members, H, W) - 保持5D，不额外处理
            # DeterministicCorrection.forward 直接接收5D cond_video
            train_data = truth_data
            baseline_data = None
            if cond_data is not None:
                forecast_in_truth_space = self._forecast_to_truth_space(cond_data)

                # ===== BUG修复 #1：残差基准改用tp单成员（与红线raw_forecast完全一致）=====
                # 原逻辑：residual_target = truth - mean(10成员)
                #   → 蓝线基础是10成员均值，但NC红线=tp单成员mm，天然不一致导致RMSE/MAE差！
                # 新逻辑（对齐test目录）：
                #   residual_target = truth_norm - tp_norm
                #   其中tp_norm来自raw_forecast(即tp单成员)做同样的normalize
                #   这样训练/推理/红线三方基准完全相同
                raw_forecast_batch = data.get("raw_forecast")  # (T, B, 1, H, W) mm单位
                if raw_forecast_batch is not None and hasattr(self.dataset, 'normalize'):
                    raw_fc_np = raw_forecast_batch.cpu().numpy()
                    # dataset.normalize内部自动按forecast_min/max归一化（距平模式下forecast已是距平）
                    tp_norm_np = self.dataset.normalize(raw_fc_np, "forecast")
                    tp_baseline = torch.as_tensor(tp_norm_np, device=self.device, dtype=truth_data.dtype)
                    # 对齐维度：(T, B, 1, H, W) -> (T, B, 1, H, W) 与truth_data(5D)一致
                    if len(tp_baseline.shape) == 4:
                        tp_baseline = tp_baseline.unsqueeze(2)
                    baseline_data = tp_baseline  # (T, B, 1, H, W)
                    residual_target = truth_data - baseline_data
                else:
                    # Fallback：取不到tp单成员时保持10成员均值逻辑（兼容无raw_forecast的情况）
                    baseline_data = forecast_in_truth_space.mean(dim=2, keepdim=True)
                    residual_target = truth_data - baseline_data
                    if self.step < 3:
                        print(f"[WARN] 无法获取tp归一化基准，fallback到10成员均值（可能导致基准不一致）")

                train_data = self._encode_residual_target(residual_target)

            # 保存最近一次的baseline，用于每100步诊断弱降水残差分布
            self._last_baseline = baseline_data.detach() if baseline_data is not None else None

            with torch.cuda.amp.autocast():
                # 准备TP Loss / FSS Loss所需的物理空间数据
                raw_fc_mm_tensor = None
                raw_tr_mm_tensor = None
                lats_tensor = None
                use_tp = getattr(config, 'use_tp_loss', False)
                use_fss = getattr(config, 'use_fss_loss', False)

                if use_tp or use_fss:
                    # raw_forecast: (T, B, 1, H, W) mm单位
                    raw_forecast_batch = data.get("raw_forecast")
                    if raw_forecast_batch is not None:
                        raw_fc_mm_tensor = raw_forecast_batch.to(self.device)
                        if raw_fc_mm_tensor.ndim == 4:
                            raw_fc_mm_tensor = raw_fc_mm_tensor.unsqueeze(2)

                    # raw_truth: (T, B, 1, H, W) mm单位
                    raw_truth_batch = data.get("raw_truth")
                    if raw_truth_batch is not None:
                        raw_tr_mm_tensor = raw_truth_batch.to(self.device)
                        if raw_tr_mm_tensor.ndim == 4:
                            raw_tr_mm_tensor = raw_tr_mm_tensor.unsqueeze(2)

                    # 获取纬度数组
                    if hasattr(self, 'dataset') and self.dataset is not None:
                        ds = self.dataset
                        if hasattr(ds, 'forecast_lats_region') and ds.forecast_lats_region is not None:
                            lats_tensor = torch.as_tensor(
                                ds.forecast_lats_region, dtype=torch.float32
                            )

                loss = self.model(
                    train_data,
                    cond_video=cond_data,
                    land_mask=land_mask,
                    weight_video=truth_data,
                    baseline_video=baseline_data,
                    target_video=truth_data,
                    raw_forecast_mm=raw_fc_mm_tensor,
                    raw_truth_mm=raw_tr_mm_tensor,
                    lats=lats_tensor,
                )

            loss = loss / config.gradient_accumulate_every
            loss.backward()
            
            if (self.step + 1) % config.gradient_accumulate_every == 0:
                # 添加梯度裁剪，防止梯度爆炸，max_norm=0.5让训练更稳定
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                self.opt.step()
                self.opt.zero_grad()
            
            if self.device == 0:
                current_loss = loss.item() * config.gradient_accumulate_every
                self.writer.add_scalar("sequence_length", train_data.shape[0], self.step)
                self.writer.add_scalar("loss", current_loss, self.step)
                
                self.train_losses.append(current_loss)
                self.train_steps.append(self.step)
                
                if self.step % 100 == 0:
                    print(f"[Step {self.step}] Loss: {current_loss:.6f}")
                    
                    # ===== 辅助诊断：读取并打印_loss分项（帮助判断correction_loss是否生效、残差范围是否合理）=====
                    try:
                        core_model = self.model.module if hasattr(self.model, "module") else self.model
                        ld = getattr(core_model, "_last_loss_dict", None)
                        if ld is not None and isinstance(ld, dict):
                            print(f"[Step {self.step}] ===== Loss分项诊断 =====")
                            order = [
                                ("编码空间主损失(未加权)", True),
                                ("物理空间correction_loss(×w前)", True),
                                ("解码后残差损失(×w前)", True),
                                ("FSS损失(×w前)", True),
                                ("物理空间mm损失(×w前)", True),
                                ("spatial_consistency_loss", True),
                                ("negative_residual_weighted_factor", False),
                                ("基线baseline_norm_mean", False),
                                ("目标correction_norm_mean", False),
                                ("tp_loss_value", True),
                                ("tp_all_metric", False),
                            ]
                            for name, is_loss in order:
                                if name in ld:
                                    v = ld[name]
                                    tag = "损失" if is_loss else "指标"
                                    print(f"    {tag}-{name}: {v:.6f}")
                            # 范围检查
                            if "模型输出x_recon范围_min" in ld and "目标x_start范围_min" in ld:
                                print(f"    范围-模型输出x_recon: [{ld['模型输出x_recon范围_min']:.3f}, {ld['模型输出x_recon范围_max']:.3f}]")
                                print(f"    范围-目标x_start:      [{ld['目标x_start范围_min']:.3f}, {ld['目标x_start范围_max']:.3f}]")
                                # 如果超出±1说明Tanh饱和或clip异常，给警告
                                if abs(ld["模型输出x_recon范围_min"]) > 1.05 or ld["模型输出x_recon范围_max"] > 1.05:
                                    print(f"    ⚠️  WARNING: 模型输出超出Tanh输出范围±1，可能存在梯度饱和！")
                    except Exception:
                        # 任何诊断打印失败都不影响训练
                        pass
                    
                    total_grad_norm = 0.0
                    for param in self.model.parameters():
                        if param.grad is not None:
                            total_grad_norm += (param.grad.norm().item() ** 2)
                    total_grad_norm = total_grad_norm ** 0.5
                    print(f"[Step {self.step}] Gradient norm: {total_grad_norm:.6f}")
                    
                    print(f"[Step {self.step}] train_data stats: min={train_data.min().item():.6f}, max={train_data.max().item():.6f}, mean={train_data.mean().item():.6f}, std={train_data.std().item():.6f}")
                    print(f"[Step {self.step}] truth_data stats: min={truth_data.min().item():.6f}, max={truth_data.max().item():.6f}, mean={truth_data.mean().item():.6f}, std={truth_data.std().item():.6f}")
                    if cond_data is not None:
                        print(f"[Step {self.step}] cond_data stats: min={cond_data.min().item():.6f}, max={cond_data.max().item():.6f}, mean={cond_data.mean().item():.6f}, std={cond_data.std().item():.6f}")
                    
                    # ===== 辅助诊断：训练数据弱降水区残差分布（与推理时一致的分bin口径）=====
                    # 从truth_data和tp_baseline计算残差目标，观察训练目标中正/负弱降水残差比例
                    try:
                        if cond_data is not None and truth_data is not None:
                            # baseline_data(tp_norm)与truth_data的差异就是残差目标（近似，未过encode）
                            # cond_data shape: (T, B, n_members, H, W), baseline 取 n_members 第0位即tp
                            if hasattr(self, '_last_baseline') and self._last_baseline is not None:
                                base = self._last_baseline
                                if base.shape == truth_data.shape:
                                    res_target = (truth_data - base).detach().cpu().numpy().astype(np.float32)
                                    # 取第一个batch做统计，避免太重
                                    first = res_target[:, 0] if res_target.ndim == 5 else res_target[0]
                                    pos_wk = np.mean((first > 0) & (first < 0.85 / 20))  # 归一化空间下"弱正残差"估计
                                    neg_wk = np.mean((first < 0) & (first > -1.2 / 20))
                                    print(f"[Step {self.step}] 训练目标残差分布: 弱正残差比例≈{pos_wk:.2%}, 弱负残差比例≈{neg_wk:.2%}")
                                    print(f"[Step {self.step}] 训练目标残差范围: [{first.min():.4f}, {first.max():.4f}], mean={first.mean():.4f}")
                    except Exception:
                        pass
                    
                    self._plot_loss_curve()
            dist.barrier()

            if (self.update_ema_every > 0) and (self.step % self.update_ema_every == 0):
                self.step_ema()

            # 测试阶段每隔 save_and_sample_every 步生成输出
            # if (self.step in [999]) and (self.step != 0):
            if (self.step % self.save_and_sample_every == 0) and (self.step != 0):
                # 确定性模型无需检查transform_fn和otherlogs
                # 使用固定的验证数据，确保每次验证使用相同的batch
                for i, batch in enumerate(self.fixed_val_batches):
                    # 处理气象数据格式
                    if isinstance(batch, dict):
                        batch_data = batch["forecast"][: self.init_num_of_frame].to(self.device)  # (T, B, n_members, H, W)
                        batch_true = batch["truth"].to(self.device)
                    else:
                        batch_data = batch[: self.init_num_of_frame].to(self.device)
                        batch_true = batch

                    # 确保形状为 5D: (T, B, C, H, W)
                    if len(batch_data.shape) == 4:
                        batch_data = batch_data.unsqueeze(2)  # 变为 (T, B, 1, H, W)
                    elif len(batch_data.shape) == 3:
                        batch_data = batch_data.unsqueeze(1).unsqueeze(2)  # 变为 (T, 1, 1, H, W)
                    
                    # 模型预测 - 使用完整的预报序列作为输入
                    # 传入完整的forecast数据，而不仅仅是init_num_of_frame帧
                    residuals = self.ema_model.module.sample(
                        batch_data,
                        10,  # 生成与forecast和truth相同长度的预测
                    )
                    
                    # 保存原始预测结果用于nc文件（未经过激活函数处理）
                    residuals_raw = residuals
                    
                    # 反归一化到 [0, 1] 范围并添加非负约束，仅用于可视化
                    residual_vis_scale = self._residual_vis_scale()
                    residuals_vis = (self._decode_residual_prediction(residuals) + residual_vis_scale) / (
                        2.0 * residual_vis_scale
                    )
                    self.writer.add_video(
                        f"samples_device/device{self.device}/num{i}",
                        residuals_vis.clamp(0.0, 1.0).transpose(0, 1),
                        self.step // self.save_and_sample_every,
                    )
                    
                    # 处理真实帧数据格式
                    if isinstance(batch, dict):
                        self.writer.add_video(
                            f"true_frames/device{self.device}/num{i}",
                            batch_true.transpose(0, 1),
                            self.step // self.save_and_sample_every,
                        )
                    else:
                        self.writer.add_video(
                            f"true_frames/device{self.device}/num{i}",
                            batch.transpose(0, 1),
                            self.step // self.save_and_sample_every,
                        )
                    
                    # ===== 训练中期只做轻量级诊断，不再保存NC（避免产生多个冗余.nc文件） =====
                    # NC文件仅在训练结束时由 final_save() 统一保存，保证结果目录干净，只有一个最终NC
                    if isinstance(batch, dict) and self.device == 0:
                        forecast_norm = batch["forecast"][:, 0, :, :, :]  # (T, n_members, H, W)
                        truth_norm = batch["truth"][:, 0, 0, :, :]       # (T, H, W)
                        residuals_np = residuals_raw.detach().cpu().numpy()
                        if residuals_np.ndim == 5:
                            residuals_np = residuals_np[:, 0, 0, :, :]
                        elif residuals_np.ndim == 4:
                            residuals_np = residuals_np[:, 0, :, :]
                        print(f"[Step {self.step}] 中期诊断 batch{i}:")
                        print(f"  forecast_norm range: [{forecast_norm.min().item():.4f}, {forecast_norm.max().item():.4f}]")
                        print(f"  truth_norm range:    [{truth_norm.min().item():.4f}, {truth_norm.max().item():.4f}]")
                        print(f"  model_residual range:[{residuals_np.min():.4f}, {residuals_np.max():.4f}], mean={residuals_np.mean():.4f}")
                if self.device == 0:
                    self.save()
                dist.barrier()
                
                # 训练将正常进行到配置的训练步数，不再在step999停止

            self.step += 1
        
        if self.device == 0:
            self.writer.flush()
            print(f"[训练结束] TensorBoard日志已刷新")
        
        # 训练结束后保存最终结果
        self.final_save()
    
    def final_save(self):
        """训练结束后保存最终结果（与test目录一致：遍历所有val样本保存多样本NC）"""
        print("\n" + "="*80)
        print("[DEBUG] 训练结束，开始保存最终结果 - 详细调试输出")
        print("="*80)
        
        if self.device == 0:
            # ========== 保存训练归一化参数到文件（供推理使用）==========
            norm_params_path = self.results_folder / "normalization_params.npz"
            if hasattr(self, 'dataset') and self.dataset is not None:
                ds = self.dataset
                f_min = float(getattr(ds, 'forecast_min', 0.0))
                f_max = float(getattr(ds, 'forecast_max', 100.0))
                t_min = float(getattr(ds, 'truth_min', 0.0))
                t_max = float(getattr(ds, 'truth_max', 100.0))
                print(f"\n[归一化保存] forecast_min={f_min:.6f}, forecast_max={f_max:.6f}")
                print(f"[归一化保存] truth_min={t_min:.6f}, truth_max={t_max:.6f}")
                print(f"[归一化保存] 保存到: {norm_params_path}")
                np.savez(str(norm_params_path),
                         forecast_min=f_min, forecast_max=f_max,
                         truth_min=t_min, truth_max=t_max)
            
            nc_results_dir = self.results_folder / "nc_results"
            nc_results_dir.mkdir(parents=True, exist_ok=True)
            
            # 存储所有样本的结果
            all_forecast = []
            all_corrected = []
            all_truth = []
            all_forecast_members = []  # 10个集合成员(未归一化)，用于CRPS计算
            all_corrected_members = []  # 10个订正后集合成员，用于corrected CRPS
            sample_dates = []
            
            n_days = 10  # 预报时效天数
            
            # ========== DEBUG 1: 数据集信息 ==========
            print("\n" + "-"*60)
            print("[DEBUG Step 1] 数据集基本信息")
            print("-"*60)
            if hasattr(self, 'val_dataset') and self.val_dataset is not None:
                ds = self.val_dataset
                print(f"  val_dataset类型: {type(ds).__name__}")
                print(f"  验证集样本数: {len(ds)}")
                if hasattr(ds, 'region'):
                    print(f"  region: {ds.region}")
                if hasattr(ds, 'forecast_lat_start'):
                    print(f"  forecast_lat_range: [{ds.forecast_lat_start}, {ds.forecast_lat_end})")
                    print(f"  forecast_lon_range: [{ds.forecast_lon_start}, {ds.forecast_lon_end})")
                if hasattr(ds, 'forecast_min'):
                    print(f"  forecast_min: {ds.forecast_min:.6f}")
                    print(f"  forecast_max: {ds.forecast_max:.6f}")
                if hasattr(ds, 'truth_min'):
                    print(f"  truth_min: {ds.truth_min:.6f}")
                    print(f"  truth_max: {ds.truth_max:.6f}")
                if hasattr(ds, 'samples'):
                    print(f"  前3个样本日期:")
                    for i, sample in enumerate(ds.samples[:3]):
                        print(f"    [{i}] init_date={sample.get('init_date', 'N/A')}")
                        print(f"        forecast_file={sample.get('forecast_file', 'N/A')}")
            print("-"*60)
            
            print(f"\n[final_save] 开始遍历验证数据集...")
            
            # 优先使用val_dataset创建不带分布式采样的DataLoader
            if hasattr(self, 'val_dataset') and self.val_dataset is not None:
                from torch.utils.data import DataLoader
                from data.transposed_collate import precipitation_collate_fn
                
                # 创建不带分布式采样的DataLoader来获取所有样本
                val_loader = DataLoader(
                    self.val_dataset,
                    batch_size=1,
                    shuffle=False,
                    collate_fn=precipitation_collate_fn,
                    num_workers=0,
                )
                total_samples = len(self.val_dataset)
                print(f"[final_save] 使用val_dataset: 验证集总样本数: {total_samples}")
                
                # 遍历所有样本并收集结果
                sample_count = 0
                for batch in val_loader:
                    if isinstance(batch, dict):
                        batch_data = batch["forecast"][:, :self.init_num_of_frame].to(self.device)
                        batch_true = batch["truth"].to(self.device)
                    else:
                        batch_data = batch[:, :self.init_num_of_frame].to(self.device)
                        batch_true = batch
                    
                    if len(batch_data.shape) == 4:
                        batch_data = batch_data.unsqueeze(2)
                    elif len(batch_data.shape) == 3:
                        batch_data = batch_data.unsqueeze(1).unsqueeze(2)
                    
                    with torch.no_grad():
                        residuals = self.ema_model.module.sample(batch_data, n_days)
                        residuals_raw = residuals
                    
                    if isinstance(batch, dict):
                        # 获取raw_forecast(tp确定性预报，未归一化，单位mm)
                        forecast = batch["raw_forecast"][:, 0, 0, :, :].cpu().numpy()  # (T, H, W) tp确定性预报
                        truth = batch["truth"][:, 0, 0, :, :].cpu().numpy()  # (T, H, W)
                        # 获取原始truth（未经归一化，与test目录一致）
                        if "raw_truth" in batch:
                            raw_truth = batch["raw_truth"][:, 0, 0, :, :].cpu().numpy()  # (T, H, W)
                        else:
                            raw_truth = None
                        # 获取原始10个集合成员（未归一化，用于CRPS）
                        raw_ensemble = None
                        if "raw_ensemble" in batch:
                            raw_ensemble = batch["raw_ensemble"][:, 0, :, :, :].cpu().numpy()  # (T, 10, H, W)
                        prediction = residuals_raw.cpu().numpy()
                        
                        if prediction.ndim == 5:
                            prediction_squeezed = prediction[:, 0, 0, :, :]
                        elif prediction.ndim == 4:
                            prediction_squeezed = prediction[:, 0, :, :]
                        else:
                            prediction_squeezed = prediction
                        
                        # ========== DEBUG 2: 第一个样本的原始数据 ==========
                        if sample_count == 0:
                            print("\n" + "-"*60)
                            print("[DEBUG Step 2] 第一个样本的原始数据（归一化空间）")
                            print("-"*60)
                            print(f"  forecast (归一化后): shape={forecast.shape}")
                            print(f"    min={forecast.min():.6f}, max={forecast.max():.6f}, mean={forecast.mean():.6f}")
                            print(f"    day0: min={forecast[0].min():.6f}, max={forecast[0].max():.6f}, mean={forecast[0].mean():.6f}")
                            print(f"  truth (归一化后): shape={truth.shape}")
                            print(f"    min={truth.min():.6f}, max={truth.max():.6f}, mean={truth.mean():.6f}")
                            print(f"    day0: min={truth[0].min():.6f}, max={truth[0].max():.6f}, mean={truth[0].mean():.6f}")
                            if raw_truth is not None:
                                print(f"  raw_truth (原始值): shape={raw_truth.shape}")
                                print(f"    min={raw_truth.min():.6f}, max={raw_truth.max():.6f}, mean={raw_truth.mean():.6f}")
                                print(f"    day0: min={raw_truth[0].min():.6f}, max={raw_truth[0].max():.6f}, mean={raw_truth[0].mean():.6f}")
                            print(f"  prediction (残差): shape={prediction_squeezed.shape}")
                            print(f"    min={prediction_squeezed.min():.6f}, max={prediction_squeezed.max():.6f}")
                            print("-"*60)
                        
                        # 反归一化
                        if hasattr(self, 'dataset') and hasattr(self.dataset, 'denormalize'):
                            # forecast已是原始tp(mm)，无需反归一化（红线不变）
                            forecast_denorm = forecast.astype(np.float32)
                            # truth使用原始值（与test目录一致），不使用反归一化
                            if raw_truth is not None:
                                truth_denorm = raw_truth.astype(np.float32)
                            else:
                                truth_denorm = self.dataset.denormalize(truth, "truth")

                            # ===== BUG修复 #1：final_save残差基准统一为tp_norm（与训练/红线一致）=====
                            # 原逻辑：距平模式用10成员均值做基准，非距平模式用tp(forecast)做基准
                            #   → 距平模式下训练基准=tp_norm，推理基准=mean(10成员)_norm，不一致导致蓝线偏差！
                            # 新逻辑（对齐test目录）：统一用 forecast（raw_forecast即tp mm）normalize → tp_norm 做基准
                            #   训练目标: truth_norm - tp_norm → 推理: corrected_norm = tp_norm + residual_decoded
                            use_anomaly = getattr(self.dataset, 'use_anomaly', False)
                            tp_norm_np = self.dataset.normalize(forecast, "forecast")
                            forecast_truth_space_np = tp_norm_np  # (T, H, W) 与训练时baseline_data完全一致

                            prediction_tensor = torch.as_tensor(
                                prediction_squeezed, device=self.device, dtype=residuals_raw.dtype
                            )
                            corrected_norm, _, _ = self._apply_residual_correction(
                                forecast_truth_space_np,
                                prediction_tensor,
                                step=self.step,
                            )
                            corrected_denorm = self.dataset.denormalize(corrected_norm, "truth")

                            # 距平模式：加回真值气候态，恢复到物理空间
                            if use_anomaly and "clim_truth" in batch:
                                clim_truth_np = batch["clim_truth"][:, 0, 0, :, :].cpu().numpy()
                                corrected_denorm = corrected_denorm + clim_truth_np
                                # 恢复物理空间后，降水量不能为负
                                corrected_denorm = np.maximum(corrected_denorm, 0.0)
                                if sample_count == 0:
                                    print(f"[距平] 已加回真值气候态: 均值={np.mean(clim_truth_np):.4f}, "
                                          f"corrected范围=[{corrected_denorm.min():.2f}, {corrected_denorm.max():.2f}]")
                        else:
                            forecast_denorm = forecast
                            truth_denorm = raw_truth if raw_truth is not None else truth
                            corrected_denorm = forecast + prediction_squeezed
                        
                        # ========== DEBUG 3: 第一个样本的反归一化数据 ==========
                        if sample_count == 0:
                            print("\n" + "-"*60)
                            print("[DEBUG Step 3] 第一个样本的反归一化数据（物理空间 mm）")
                            print("-"*60)
                            print(f"  forecast_denorm: shape={forecast_denorm.shape}")
                            print(f"    min={forecast_denorm.min():.6f}, max={forecast_denorm.max():.6f}, mean={forecast_denorm.mean():.6f}")
                            print(f"    day0: min={forecast_denorm[0].min():.6f}, max={forecast_denorm[0].max():.6f}, mean={forecast_denorm[0].mean():.6f}")
                            print(f"    day1: min={forecast_denorm[1].min():.6f}, max={forecast_denorm[1].max():.6f}, mean={forecast_denorm[1].mean():.6f}")
                            print(f"    day5: min={forecast_denorm[5].min():.6f}, max={forecast_denorm[5].max():.6f}, mean={forecast_denorm[5].mean():.6f}")
                            print(f"  truth_denorm: shape={truth_denorm.shape}")
                            print(f"    min={truth_denorm.min():.6f}, max={truth_denorm.max():.6f}, mean={truth_denorm.mean():.6f}")
                            print(f"    day0: min={truth_denorm[0].min():.6f}, max={truth_denorm[0].max():.6f}, mean={truth_denorm[0].mean():.6f}")
                            print(f"    day1: min={truth_denorm[1].min():.6f}, max={truth_denorm[1].max():.6f}, mean={truth_denorm[1].mean():.6f}")
                            print(f"    day5: min={truth_denorm[5].min():.6f}, max={truth_denorm[5].max():.6f}, mean={truth_denorm[5].mean():.6f}")
                            print(f"  corrected_denorm: shape={corrected_denorm.shape}")
                            print(f"    min={corrected_denorm.min():.6f}, max={corrected_denorm.max():.6f}, mean={corrected_denorm.mean():.6f}")
                            print("-"*60)
                        
                        # 确保时间步一致
                        target_steps = min(n_days, forecast_denorm.shape[0])
                        forecast_denorm = forecast_denorm[:target_steps]
                        truth_denorm = truth_denorm[:target_steps]
                        corrected_denorm = corrected_denorm[:target_steps]
                        if raw_ensemble is not None:
                            raw_ensemble = raw_ensemble[:target_steps]

                        # 填充时间步至n_days
                        def pad_to_n_days(arr, n_days):
                            if arr.shape[0] < n_days:
                                pad = np.zeros((n_days - arr.shape[0],) + arr.shape[1:], dtype=arr.dtype)
                                pad[:] = arr[-1:]
                                arr = np.concatenate([arr, pad], axis=0)
                            return arr

                        forecast_denorm = pad_to_n_days(forecast_denorm, n_days)
                        truth_denorm = pad_to_n_days(truth_denorm, n_days)
                        corrected_denorm = pad_to_n_days(corrected_denorm, n_days)
                        if raw_ensemble is not None:
                            raw_ensemble = pad_to_n_days(raw_ensemble, n_days)

                        # 获取日期信息
                        if hasattr(val_loader, 'dataset') and hasattr(val_loader.dataset, 'samples'):
                            sample_idx = sample_count % len(val_loader.dataset.samples)
                            sample_info = val_loader.dataset.samples[sample_idx]
                            sample_dates.append(sample_info.get('init_date', f'sample_{sample_count}'))
                        else:
                            sample_dates.append(f'sample_{sample_count}')

                        # ===== 零值保护：拦截raw<0.1mm格点正残差导致的虚假小降水（修复MAE/CRPS/FSS）=====
                        corrected_denorm, corrected_members_denorm = self._zero_protect_correction(
                            forecast_denorm, corrected_denorm, raw_ensemble
                        )

                        all_forecast.append(forecast_denorm.copy())
                        all_corrected.append(corrected_denorm.copy())
                        all_truth.append(truth_denorm.copy())
                        if raw_ensemble is not None:
                            all_forecast_members.append(raw_ensemble.copy())
                            all_corrected_members.append(corrected_members_denorm.copy())
                        
                        sample_count += 1
                        if sample_count % 5 == 0:
                            print(f"[final_save] 已处理 {sample_count}/{total_samples} 个样本")
            elif hasattr(self, 'val_dl_class') and self.val_dl_class is not None:
                val_loader = self.val_dl_class
                total_samples = len(val_loader.dataset) if hasattr(val_loader, 'dataset') else len(val_loader)
                print(f"[final_save] 使用val_dl_class: 验证集总样本数: {total_samples}")
                
                sample_count = 0
                for batch in val_loader:
                    if isinstance(batch, dict):
                        batch_data = batch["forecast"][:, :self.init_num_of_frame].to(self.device)
                        batch_true = batch["truth"].to(self.device)
                    else:
                        batch_data = batch[:, :self.init_num_of_frame].to(self.device)
                        batch_true = batch
                    
                    if len(batch_data.shape) == 4:
                        batch_data = batch_data.unsqueeze(2)
                    elif len(batch_data.shape) == 3:
                        batch_data = batch_data.unsqueeze(1).unsqueeze(2)
                    
                    with torch.no_grad():
                        residuals = self.ema_model.module.sample(batch_data, n_days)
                        residuals_raw = residuals
                    
                    if isinstance(batch, dict):
                        # 获取raw_forecast(tp确定性预报，未归一化，单位mm)
                        forecast = batch["raw_forecast"][:, 0, 0, :, :].cpu().numpy()  # (T, H, W) tp确定性预报
                        truth = batch["truth"][:, 0, 0, :, :].cpu().numpy()  # (T, H, W)
                        # 获取原始truth（未经归一化，与test目录一致）
                        if "raw_truth" in batch:
                            raw_truth = batch["raw_truth"][:, 0, 0, :, :].cpu().numpy()
                        else:
                            raw_truth = None
                        # 获取原始10个集合成员（未归一化，用于CRPS）
                        raw_ensemble = None
                        if "raw_ensemble" in batch:
                            raw_ensemble = batch["raw_ensemble"][:, 0, :, :, :].cpu().numpy()  # (T, 10, H, W)
                        prediction = residuals_raw.cpu().numpy()

                        if prediction.ndim == 5:
                            prediction_squeezed = prediction[:, 0, 0, :, :]
                        elif prediction.ndim == 4:
                            prediction_squeezed = prediction[:, 0, :, :]
                        else:
                            prediction_squeezed = prediction

                        # 反归一化
                        if hasattr(self, 'dataset') and hasattr(self.dataset, 'denormalize'):
                            # forecast已是原始tp(mm)，无需反归一化
                            forecast_denorm = forecast.astype(np.float32)
                            # 优先使用raw_truth（与test目录一致，不经过归一化/反归一化）
                            if raw_truth is not None:
                                truth_denorm = raw_truth.astype(np.float32)
                            else:
                                truth_denorm = self.dataset.denormalize(truth, "truth")

                            # ===== BUG修复 #2：val_dl_class路径残差基准+clim_truth回加对齐训练 =====
                            forecast_norm = self.dataset.normalize(forecast, "forecast")
                            forecast_tensor = torch.as_tensor(
                                forecast_norm, device=self.device, dtype=residuals_raw.dtype
                            )
                            forecast_truth_space = self._forecast_to_truth_space(forecast_tensor)
                            forecast_truth_space_np = forecast_truth_space.detach().cpu().numpy()

                            prediction_tensor = torch.as_tensor(
                                prediction_squeezed, device=self.device, dtype=residuals_raw.dtype
                            )
                            corrected_norm, _, _ = self._apply_residual_correction(
                                forecast_truth_space_np,
                                prediction_tensor,
                                step=self.step,
                            )
                            corrected_denorm = self.dataset.denormalize(corrected_norm, "truth")

                            # 距平模式：加回真值气候态（与Path1/Path3一致，修复MAE/CRPS）
                            use_anomaly = getattr(self.dataset, 'use_anomaly', False)
                            if use_anomaly and "clim_truth" in batch:
                                clim_truth_np = batch["clim_truth"][:, 0, 0, :, :].cpu().numpy()
                                corrected_denorm = corrected_denorm + clim_truth_np
                                corrected_denorm = np.maximum(corrected_denorm, 0.0)
                        else:
                            forecast_denorm = forecast
                            truth_denorm = truth
                            corrected_denorm = forecast + prediction_squeezed

                        # 确保时间步一致
                        target_steps = min(n_days, forecast_denorm.shape[0])
                        forecast_denorm = forecast_denorm[:target_steps]
                        truth_denorm = truth_denorm[:target_steps]
                        corrected_denorm = corrected_denorm[:target_steps]
                        if raw_ensemble is not None:
                            raw_ensemble = raw_ensemble[:target_steps]

                        # 填充时间步至n_days
                        def pad_to_n_days(arr, n_days):
                            if arr.shape[0] < n_days:
                                pad = np.zeros((n_days - arr.shape[0],) + arr.shape[1:], dtype=arr.dtype)
                                pad[:] = arr[-1:]
                                arr = np.concatenate([arr, pad], axis=0)
                            return arr

                        forecast_denorm = pad_to_n_days(forecast_denorm, n_days)
                        truth_denorm = pad_to_n_days(truth_denorm, n_days)
                        corrected_denorm = pad_to_n_days(corrected_denorm, n_days)
                        if raw_ensemble is not None:
                            raw_ensemble = pad_to_n_days(raw_ensemble, n_days)

                        # 获取日期信息
                        if hasattr(val_loader, 'dataset') and hasattr(val_loader.dataset, 'samples'):
                            sample_idx = sample_count % len(val_loader.dataset.samples)
                            sample_info = val_loader.dataset.samples[sample_idx]
                            sample_dates.append(sample_info.get('init_date', f'sample_{sample_count}'))
                        else:
                            sample_dates.append(f'sample_{sample_count}')

                        # ===== 零值保护：拦截raw<0.1mm格点正残差导致的虚假小降水（修复MAE/CRPS/FSS）=====
                        corrected_denorm, corrected_members_denorm = self._zero_protect_correction(
                            forecast_denorm, corrected_denorm, raw_ensemble
                        )

                        all_forecast.append(forecast_denorm.copy())
                        all_corrected.append(corrected_denorm.copy())
                        all_truth.append(truth_denorm.copy())
                        if raw_ensemble is not None:
                            all_forecast_members.append(raw_ensemble.copy())
                            all_corrected_members.append(corrected_members_denorm.copy())
                        
                        sample_count += 1
                        if sample_count % 5 == 0:
                            print(f"[final_save] 已处理 {sample_count}/{total_samples} 个样本")
            else:
                # 回退：使用fixed_val_batches
                print(f"[final_save] 使用fixed_val_batches（共{len(self.fixed_val_batches)}个batch）")
                for i, batch in enumerate(self.fixed_val_batches):
                    if isinstance(batch, dict):
                        batch_data = batch["forecast"][: self.init_num_of_frame].to(self.device)
                        batch_true = batch["truth"].to(self.device)
                    else:
                        batch_data = batch[: self.init_num_of_frame].to(self.device)
                        batch_true = batch
                    
                    if len(batch_data.shape) == 4:
                        batch_data = batch_data.unsqueeze(2)
                    elif len(batch_data.shape) == 3:
                        batch_data = batch_data.unsqueeze(1).unsqueeze(2)
                    
                    with torch.no_grad():
                        residuals = self.ema_model.module.sample(batch_data, n_days)
                        residuals_raw = residuals
                    
                    if isinstance(batch, dict):
                        # 获取raw_forecast(tp确定性预报，未归一化，单位mm)
                        forecast = batch["raw_forecast"][:, 0, 0, :, :].cpu().numpy()
                        truth = batch["truth"][:, 0, 0, :, :].cpu().numpy()
                        # 获取原始truth（未经归一化，与test目录一致）
                        if "raw_truth" in batch:
                            raw_truth = batch["raw_truth"][:, 0, 0, :, :].cpu().numpy()
                        else:
                            raw_truth = None
                        # 获取原始10个集合成员（未归一化，用于CRPS）
                        raw_ensemble = None
                        if "raw_ensemble" in batch:
                            raw_ensemble = batch["raw_ensemble"][:, 0, :, :, :].cpu().numpy()  # (T, 10, H, W)
                        prediction = residuals_raw.cpu().numpy()

                        if prediction.ndim == 5:
                            prediction_squeezed = prediction[:, 0, 0, :, :]
                        elif prediction.ndim == 4:
                            prediction_squeezed = prediction[:, 0, :, :]
                        else:
                            prediction_squeezed = prediction

                        if hasattr(self, 'dataset') and hasattr(self.dataset, 'denormalize'):
                            # forecast已是原始tp(mm)，无需反归一化（红线不变）
                            forecast_denorm = forecast.astype(np.float32)
                            # 优先使用raw_truth（与test目录一致，不经过归一化/反归一化）
                            if raw_truth is not None:
                                truth_denorm = raw_truth.astype(np.float32)
                            else:
                                truth_denorm = self.dataset.denormalize(truth, "truth")

                            # ===== BUG修复 #1：final_save残差基准统一为tp_norm（与红线/训练一致）=====
                            # 原逻辑：距平模式用10成员均值做基准，非距平模式用tp(forecast)做基准
                            #   → 距平模式下训练基准=tp_norm，推理基准=mean(10成员)_norm，不一致导致蓝线偏差！
                            # 新逻辑（对齐test目录）：统一用 forecast（raw_forecast即tp mm）normalize → tp_norm 做基准
                            #   训练目标: truth_norm - tp_norm → 推理: corrected_norm = tp_norm + residual_decoded
                            use_anomaly = getattr(self.dataset, 'use_anomaly', False)
                            # forecast = raw_forecast mm (T, H, W) 直接normalize
                            tp_norm_np = self.dataset.normalize(forecast, "forecast")
                            forecast_truth_space_np = tp_norm_np  # (T, H, W) 与训练时baseline_data完全一致

                            prediction_tensor = torch.as_tensor(prediction_squeezed, device=self.device, dtype=residuals_raw.dtype)
                            corrected_norm, _, _ = self._apply_residual_correction(
                                forecast_truth_space_np, prediction_tensor, step=self.step,
                            )
                            corrected_denorm = self.dataset.denormalize(corrected_norm, "truth")

                            # 距平模式：加回真值气候态（保持不变）
                            if use_anomaly and "clim_truth" in batch:
                                clim_truth_np = batch["clim_truth"][:, 0, 0, :, :].cpu().numpy()
                                corrected_denorm = corrected_denorm + clim_truth_np
                                # 恢复物理空间后，降水量不能为负
                                corrected_denorm = np.maximum(corrected_denorm, 0.0)
                        else:
                            forecast_denorm = forecast
                            truth_denorm = truth
                            corrected_denorm = forecast + prediction_squeezed

                        target_steps = min(n_days, forecast_denorm.shape[0])
                        forecast_denorm = forecast_denorm[:target_steps]
                        truth_denorm = truth_denorm[:target_steps]
                        corrected_denorm = corrected_denorm[:target_steps]
                        if raw_ensemble is not None:
                            raw_ensemble = raw_ensemble[:target_steps]
                            if raw_ensemble.shape[0] < n_days:
                                pad = np.zeros((n_days - raw_ensemble.shape[0],) + raw_ensemble.shape[1:], dtype=raw_ensemble.dtype)
                                pad[:] = raw_ensemble[-1:]
                                raw_ensemble = np.concatenate([raw_ensemble, pad], axis=0)

                        sample_dates.append(f'batch_{i}')
                        # ===== 零值保护：拦截raw<0.1mm格点正残差导致的虚假小降水（修复MAE/CRPS/FSS）=====
                        corrected_denorm, corrected_members_denorm = self._zero_protect_correction(
                            forecast_denorm, corrected_denorm, raw_ensemble
                        )

                        all_forecast.append(forecast_denorm.copy())
                        all_corrected.append(corrected_denorm.copy())
                        all_truth.append(truth_denorm.copy())
                        if raw_ensemble is not None:
                            all_forecast_members.append(raw_ensemble.copy())
                            all_corrected_members.append(corrected_members_denorm.copy())
            
            # 保存为NC文件（与test目录一致格式）
            if len(all_forecast) > 0:
                print(f"\n[final_save] 共收集 {len(all_forecast)} 个样本，准备保存NC文件...")
                
                forecast_stack = np.stack(all_forecast, axis=0)  # (n_samples, T, H, W)
                corrected_stack = np.stack(all_corrected, axis=0)
                truth_stack = np.stack(all_truth, axis=0)
                # 堆叠10个集合成员: (n_samples, T, 10, H, W)
                forecast_members_stack = None
                if len(all_forecast_members) > 0:
                    forecast_members_stack = np.stack(all_forecast_members, axis=0)
                # 堆叠订正后集合成员: (n_samples, T, 10, H, W)
                corrected_members_stack = None
                if len(all_corrected_members) > 0:
                    corrected_members_stack = np.stack(all_corrected_members, axis=0)

                n_samples, T, H, W = forecast_stack.shape
                print(f"[final_save] 数据形状: forecast={forecast_stack.shape}, corrected={corrected_stack.shape}, truth={truth_stack.shape}")
                if forecast_members_stack is not None:
                    print(f"[final_save] 集合成员形状: forecast_members={forecast_members_stack.shape}")
                
                # ========== DEBUG 4: 保存前的数据统计 ==========
                print("\n" + "-"*60)
                print("[DEBUG Step 4] 保存NC文件前的数据统计")
                print("-"*60)
                print(f"  样本数: {n_samples}, 时效数: {T}, 纬度数: {H}, 经度数: {W}")
                print(f"  日期列表: {sample_dates[:5]}...")
                print(f"\n  forecast (所有样本平均):")
                print(f"    day0: min={forecast_stack[:,0,:,:].min():.6f}, max={forecast_stack[:,0,:,:].max():.6f}, mean={forecast_stack[:,0,:,:].mean():.6f}")
                print(f"    day1: min={forecast_stack[:,1,:,:].min():.6f}, max={forecast_stack[:,1,:,:].max():.6f}, mean={forecast_stack[:,1,:,:].mean():.6f}")
                print(f"    day5: min={forecast_stack[:,5,:,:].min():.6f}, max={forecast_stack[:,5,:,:].max():.6f}, mean={forecast_stack[:,5,:,:].mean():.6f}")
                print(f"    day9: min={forecast_stack[:,9,:,:].min():.6f}, max={forecast_stack[:,9,:,:].max():.6f}, mean={forecast_stack[:,9,:,:].mean():.6f}")
                print(f"\n  truth (所有样本平均):")
                print(f"    day0: min={truth_stack[:,0,:,:].min():.6f}, max={truth_stack[:,0,:,:].max():.6f}, mean={truth_stack[:,0,:,:].mean():.6f}")
                print(f"    day1: min={truth_stack[:,1,:,:].min():.6f}, max={truth_stack[:,1,:,:].max():.6f}, mean={truth_stack[:,1,:,:].mean():.6f}")
                print(f"    day5: min={truth_stack[:,5,:,:].min():.6f}, max={truth_stack[:,5,:,:].max():.6f}, mean={truth_stack[:,5,:,:].mean():.6f}")
                print(f"    day9: min={truth_stack[:,9,:,:].min():.6f}, max={truth_stack[:,9,:,:].max():.6f}, mean={truth_stack[:,9,:,:].mean():.6f}")
                print("-"*60)

                # ========== final_save 辅助诊断汇总：Day1-Day10 弱降水命中率 + CSI@0.1 + MAE ==========
                # 从超算下载.out后，我可以对照这段与 infer_global 的诊断结果交叉验证
                print("\n" + "=" * 110)
                print("[final_save-DIAG] Day1-Day10 关键指标汇总（下载.out后请提供这段给我分析）")
                print("=" * 110)
                print(f"{'Day':<4} {'RAW弱命中率<0.1-1mm':<22} {'COR弱命中率':<18} {'Δ命中':<8} "
                      f"{'RAW MAE(mm)':<14} {'COR MAE(mm)':<14} {'ΔMAE':<10} "
                      f"{'RAW CSI@0.1':<12} {'COR CSI@0.1':<12} {'ΔCSI':<8}")
                print("-" * 110)
                try:
                    n_lead = min(T, 10)
                    fs_final = forecast_stack
                    cs_final = corrected_stack
                    ts_final = truth_stack
                    # 分bin口径：弱降水 truth 在 [0.1, 1.0) mm，原始/订正 >0.1mm 视为命中
                    thr_csi = 0.1
                    weak_lo, weak_hi = 0.1, 1.0
                    for t in range(n_lead):
                        fc = fs_final[:, t, :, :]
                        co = cs_final[:, t, :, :]
                        tr = ts_final[:, t, :, :]
                        # 弱降水命中率（命中=truth在此区间 且 预报>=0.1）
                        weak_mask = (tr >= weak_lo) & (tr < weak_hi)
                        if weak_mask.sum() > 0:
                            raw_wk_hit = np.mean(fc[weak_mask] >= thr_csi)
                            cor_wk_hit = np.mean(co[weak_mask] >= thr_csi)
                            d_wk_hit = cor_wk_hit - raw_wk_hit
                        else:
                            raw_wk_hit = cor_wk_hit = d_wk_hit = 0
                        # MAE
                        raw_mae = np.mean(np.abs(fc - tr))
                        cor_mae = np.mean(np.abs(co - tr))
                        d_mae = cor_mae - raw_mae
                        # CSI@0.1
                        tru_p = tr >= thr_csi
                        fc_p = fc >= thr_csi
                        co_p = co >= thr_csi
                        rtp = np.sum(tru_p & fc_p)
                        rfp = np.sum((~tru_p) & fc_p)
                        rfn = np.sum(tru_p & (~fc_p))
                        ctp = np.sum(tru_p & co_p)
                        cfp = np.sum((~tru_p) & co_p)
                        cfn = np.sum(tru_p & (~co_p))
                        rcsi = rtp / (rtp + rfp + rfn) if (rtp + rfp + rfn) > 0 else 0
                        ccsi = ctp / (ctp + cfp + cfn) if (ctp + cfp + cfn) > 0 else 0
                        dcsi = ccsi - rcsi
                        wk_tag = "✅" if d_wk_hit > 0.01 else ("⚠️" if d_wk_hit > -0.01 else "❌")
                        mae_tag = "✅" if d_mae < -0.005 else ("⚠️" if d_mae < 0.005 else "❌")
                        csi_tag = "✅" if dcsi > 0.001 else ("⚠️" if dcsi > -0.001 else "❌")
                        print(f"D{t+1:<3} {raw_wk_hit*100:>12.2f}%            "
                              f"{cor_wk_hit*100:>12.2f}%      {d_wk_hit*100:+6.2f}% {wk_tag}  "
                              f"{raw_mae:>12.4f}   {cor_mae:>12.4f}   {d_mae:+8.4f} {mae_tag}  "
                              f"{rcsi:>10.4f}    {ccsi:>10.4f}    {dcsi:>+7.4f} {csi_tag}")
                    print("-" * 110)
                    # 第二部分：残差在降水强度bins上的表现（所有lead_time合并统计）
                    fs_f = fs_final.reshape(-1)
                    cs_f = cs_final.reshape(-1)
                    ts_f = ts_final.reshape(-1)
                    print("[分雨强bin] 所有lead_time合并的MAE分解 / 正残差比例（帮助理解MAE/CSI来源）")
                    bins_def = [
                        ("无雨 0-0.1mm",   0,    0.1),
                        ("极弱 0.1-0.2mm", 0.1,  0.2),
                        ("弱雨 0.2-0.5mm", 0.2,  0.5),
                        ("小雨 0.5-1mm",   0.5,  1.0),
                        ("轻雨 1-2mm",     1,    2),
                        ("中雨 2-5mm",     2,    5),
                        ("大雨 5mm+",      5,    float("inf")),
                    ]
                    print(f"  {'雨强(truth)':<18} {'像素占比':<10} {'RAW MAE':<10} {'COR MAE':<10} {'改进(mm)':<10} "
                          f"{'正残差比例RAW':<14} {'正残差比例COR':<14}")
                    total_pix = ts_f.size
                    res_raw = fs_f - ts_f  # 残差=预测-真值（这里叫residual其实是error）
                    res_cor = cs_f - ts_f
                    for name, lo, hi in bins_def:
                        m = (ts_f >= lo) & (ts_f < hi)
                        cnt = np.sum(m)
                        if cnt == 0:
                            continue
                        rmae = np.mean(np.abs(res_raw[m]))
                        cmae = np.mean(np.abs(res_cor[m]))
                        imp = rmae - cmae
                        # 正残差比例（corrected>raw说明正残差加生效；cor_raw_diff>0 = 订正后抬高了值）
                        diff = cs_f[m] - fs_f[m]
                        p_raw = np.mean(res_raw[m] > 0)  # raw残差正=预报偏高
                        p_cor = np.mean(res_cor[m] > 0)  # cor残差正=订正后仍然偏高
                        ratio = cnt / total_pix * 100
                        print(f"  {name:<18} {ratio:>7.2f}%    {rmae:>8.4f}   {cmae:>8.4f}   {imp:>+8.4f}   "
                              f"{p_raw*100:>10.2f}%      {p_cor*100:>10.2f}%")
                    print("=" * 110 + "\n")
                except Exception as e_diag:
                    print(f"[WARN] final_save诊断计算失败（不影响NC保存）: {e_diag}")
                    import traceback
                    traceback.print_exc()
                print("=" * 110 + "\n")
                
                # 获取经纬度
                output_lats, output_lons = self._get_output_lat_lon(H, W)
                
                print(f"\n[DEBUG] 输出经纬度范围: lat=[{output_lats[0]:.4f}, {output_lats[-1]:.4f}], lon=[{output_lons[0]:.4f}, {output_lons[-1]:.4f}]")
                
                # 与test目录一致的NC格式：(time, lead_time, lat, lon)
                nc_filename = nc_results_dir / "final_prediction_results.nc"
                with nc.Dataset(nc_filename, 'w') as f:
                    f.createDimension('time', n_samples)
                    f.createDimension('lead_time', T)
                    f.createDimension('lat', H)
                    f.createDimension('lon', W)
                    # 新增member维度(10个集合成员，用于CRPS计算)
                    n_members_dim = None
                    if forecast_members_stack is not None:
                        n_members_dim = forecast_members_stack.shape[2]
                        f.createDimension('member', n_members_dim)

                    time_var = f.createVariable('time', 'S10', ('time',))
                    lead_var = f.createVariable('lead_time', 'i4', ('lead_time',))
                    lat_var = f.createVariable('lat', 'f8', ('lat',))
                    lon_var = f.createVariable('lon', 'f8', ('lon',))

                    forecast_var = f.createVariable('forecast', 'f4', ('time', 'lead_time', 'lat', 'lon'))
                    corrected_var = f.createVariable('corrected', 'f4', ('time', 'lead_time', 'lat', 'lon'))
                    truth_var = f.createVariable('truth', 'f4', ('time', 'lead_time', 'lat', 'lon'))
                    # 新增forecast_members变量保存10个集合成员
                    forecast_members_var = None
                    if forecast_members_stack is not None:
                        forecast_members_var = f.createVariable('forecast_members', 'f4', ('time', 'lead_time', 'member', 'lat', 'lon'))
                    # 新增corrected_members变量保存订正后集合成员
                    corrected_members_var = None
                    if corrected_members_stack is not None:
                        corrected_members_var = f.createVariable('corrected_members', 'f4', ('time', 'lead_time', 'member', 'lat', 'lon'))
                    
                    time_var[:] = np.array([date.encode('utf-8') for date in sample_dates])
                    lead_var[:] = np.arange(T)
                    lat_var[:] = output_lats
                    lon_var[:] = output_lons
                    
                    # 应用陆地掩码（与test目录一致：保存前将海洋区域设为NaN）
                    land_mask_for_save = None
                    if hasattr(self, 'dataset') and self.dataset is not None and hasattr(self.dataset, 'land_mask'):
                        land_mask = self.dataset.land_mask
                        if land_mask is not None:
                            if land_mask.shape != (H, W):
                                from scipy.ndimage import zoom
                                land_mask = zoom(land_mask.astype(np.float32),
                                               (H / land_mask.shape[0], W / land_mask.shape[1]),
                                               order=0) > 0.5
                            land_mask_for_save = land_mask
                            print(f"[final_save] 应用陆地掩码: shape={land_mask.shape}, 陆地占比={np.mean(land_mask)*100:.1f}%")
                            
                            # 应用掩码：陆地区域保留，海洋区域设为NaN（与test一致）
                            mask_broadcast = land_mask[np.newaxis, np.newaxis, :, :]
                            forecast_stack = np.where(mask_broadcast, np.nan_to_num(forecast_stack, nan=0.0), np.nan)
                            corrected_stack = np.where(mask_broadcast, np.nan_to_num(corrected_stack, nan=0.0), np.nan)
                            truth_stack = np.where(mask_broadcast, np.nan_to_num(truth_stack, nan=0.0), np.nan)
                            # 集合成员不预先应用陆地掩码，在plot_metrics_by_lead_time.py中统一处理
                            print(f"[final_save] 应用掩码后: forecast NaN数={np.sum(np.isnan(forecast_stack))}, truth NaN数={np.sum(np.isnan(truth_stack))}")
                    
                    forecast_data = np.ascontiguousarray(forecast_stack, dtype=np.float32)
                    corrected_data = np.ascontiguousarray(corrected_stack, dtype=np.float32)
                    truth_data = np.ascontiguousarray(truth_stack, dtype=np.float32)

                    forecast_var[:, :, :, :] = forecast_data
                    corrected_var[:, :, :, :] = corrected_data
                    truth_var[:, :, :, :] = truth_data

                    forecast_var.units = 'mm'
                    corrected_var.units = 'mm'
                    truth_var.units = 'mm'

                    # 保存10个集合成员(用于CRPS计算)
                    if forecast_members_var is not None and forecast_members_stack is not None:
                        forecast_members_data = np.ascontiguousarray(forecast_members_stack, dtype=np.float32)
                        forecast_members_var[:, :, :, :, :] = forecast_members_data
                        forecast_members_var.units = 'mm'
                        print(f"[final_save] forecast_members已保存: shape={forecast_members_data.shape}")

                    # 保存订正后集合成员(用于corrected CRPS计算)
                    if corrected_members_var is not None and corrected_members_stack is not None:
                        corrected_members_data = np.ascontiguousarray(corrected_members_stack, dtype=np.float32)
                        corrected_members_var[:, :, :, :, :] = corrected_members_data
                        corrected_members_var.units = 'mm'
                        print(f"[final_save] corrected_members已保存: shape={corrected_members_data.shape}")
                    
                    # 保存陆地掩码
                    if land_mask_for_save is not None:
                        land_mask_var = f.createVariable('land_mask', 'f4', ('lat', 'lon'))
                        land_mask_var[:] = land_mask_for_save.astype(np.float32)
                        print(f"[final_save] 陆地掩码已保存: shape={land_mask_for_save.shape}")
                    
                    print(f"[final_save] NC文件已保存: {nc_filename}")
                    print(f"  forecast: min={forecast_data.min():.4f}, max={forecast_data.max():.4f}")
                    print(f"  truth: min={truth_data.min():.4f}, max={truth_data.max():.4f}")
                    print(f"  corrected: min={corrected_data.min():.4f}, max={corrected_data.max():.4f}")
            else:
                print("[final_save] 警告：没有收集到任何样本数据！")
            
            # 保存模型
            self.save()
            print(f"✅ 保存最终模型到: {self.results_folder}")
            
            self.writer.flush()
            self.writer.close()
            print(f"✅ TensorBoard日志已关闭")
        
        dist.barrier()
        print("training completed")

