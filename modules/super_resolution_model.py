"""Two-stage SR: 2-D Conv-ResNet prior followed by multiscale residual DDIM."""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def _groups(channels: int) -> int:
    for value in (8, 4, 2, 1):
        if channels % value == 0:
            return value
    return 1


class ConvResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.GroupNorm(_groups(channels), channels), nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(_groups(channels), channels), nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x):
        return x + self.net(x)


class RegressionConvResNet(nn.Module):
    """Phase 1: learned high-resolution residual over bilinear interpolation."""

    def __init__(self, base_channels=64, num_blocks=8, residual_scale=0.5):
        super().__init__()
        self.residual_scale = float(residual_scale)
        self.stem = nn.Conv2d(1, base_channels, 3, padding=1)
        self.body = nn.Sequential(*[ConvResidualBlock(base_channels) for _ in range(num_blocks)])
        self.head = nn.Sequential(nn.GroupNorm(_groups(base_channels), base_channels), nn.SiLU(),
                                  nn.Conv2d(base_channels, 1, 3, padding=1))

    def forward(self, low_resolution, target_size):
        baseline = F.interpolate(low_resolution, size=target_size, mode="bilinear", align_corners=True)
        features = self.stem(baseline)
        residual = torch.tanh(self.head(self.body(features))) * self.residual_scale
        prior = (baseline + residual).clamp(0.0, 1.0)
        return baseline, prior


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        scale = math.log(10000.0) / max(half - 1, 1)
        freq = torch.exp(torch.arange(half, device=t.device) * -scale)
        emb = t.float()[:, None] * freq[None]
        return torch.cat((emb.sin(), emb.cos()), dim=1)


class TimeResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(_groups(in_channels), in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.time = nn.Linear(time_dim, out_channels)
        self.norm2 = nn.GroupNorm(_groups(out_channels), out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.skip = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, time_embedding):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time(time_embedding)[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class ConditionalDenoisingUNet(nn.Module):
    """Phase 2 denoiser. Input is noisy residual concatenated with Phase-1 prior."""

    def __init__(self, base_channels=48, condition_channels=2):
        super().__init__()
        self.condition_channels = int(condition_channels)
        time_dim = base_channels * 4
        self.time_mlp = nn.Sequential(SinusoidalTimeEmbedding(base_channels),
                                      nn.Linear(base_channels, time_dim), nn.SiLU(),
                                      nn.Linear(time_dim, time_dim))
        self.in_conv = nn.Conv2d(1 + self.condition_channels, base_channels, 3, padding=1)
        self.enc1 = TimeResBlock(base_channels, base_channels, time_dim)
        self.down1 = nn.Conv2d(base_channels, base_channels * 2, 4, stride=2, padding=1)
        self.enc2 = TimeResBlock(base_channels * 2, base_channels * 2, time_dim)
        self.down2 = nn.Conv2d(base_channels * 2, base_channels * 4, 4, stride=2, padding=1)
        self.mid = TimeResBlock(base_channels * 4, base_channels * 4, time_dim)
        self.dec2 = TimeResBlock(base_channels * 6, base_channels * 2, time_dim)
        self.dec1 = TimeResBlock(base_channels * 3, base_channels, time_dim)
        self.out = nn.Sequential(nn.GroupNorm(_groups(base_channels), base_channels), nn.SiLU(),
                                 nn.Conv2d(base_channels, 1, 3, padding=1))

    def forward(self, noisy_residual, timestep, condition):
        temb = self.time_mlp(timestep)
        x1 = self.enc1(self.in_conv(torch.cat((noisy_residual, condition), dim=1)), temb)
        x2 = self.enc2(self.down1(x1), temb)
        x3 = self.mid(self.down2(x2), temb)
        x = F.interpolate(x3, size=x2.shape[-2:], mode="bilinear", align_corners=False)
        x = self.dec2(torch.cat((x, x2), dim=1), temb)
        x = F.interpolate(x, size=x1.shape[-2:], mode="bilinear", align_corners=False)
        x = self.dec1(torch.cat((x, x1), dim=1), temb)
        return self.out(x)


def _extract(values, timestep, shape):
    return values.gather(0, timestep).reshape(timestep.shape[0], *((1,) * (len(shape) - 1)))


class TwoStageSuperResolution(nn.Module):
    """Stable Phase-2 controlled multiscale residual diffusion.

    Phase 1 supplies the large-scale deterministic field.  Phase 2 uses
    v-prediction and deterministic DDIM to generate only the unresolved detail.
    A learned confidence gate and an explicit coarse-grid consistency projection
    prevent the diffusion result from erasing or replacing the Phase-1 field.
    """

    def __init__(self, regression: RegressionConvResNet, denoiser: ConditionalDenoisingUNet,
                 timesteps=100, residual_clip=4.0, sampling_steps=25,
                 coarse_residual_fraction=0.35, coarse_residual_max_fraction=0.80,
                 gate_min=0.10, gate_max=0.85):
        super().__init__()
        self.regression, self.denoiser = regression, denoiser
        self.timesteps = int(timesteps)
        self.residual_clip = float(residual_clip)
        self.sampling_steps = min(int(sampling_steps), self.timesteps)
        self.coarse_residual_fraction = float(coarse_residual_fraction)
        self.coarse_residual_max_fraction = float(coarse_residual_max_fraction)
        self.gate_min = float(gate_min)
        self.gate_max = float(gate_max)
        if not 0.0 <= self.gate_min < self.gate_max <= 1.0:
            raise ValueError("Require 0 <= gate_min < gate_max <= 1")
        if not 0.0 <= self.coarse_residual_fraction <= self.coarse_residual_max_fraction <= 1.0:
            raise ValueError("Require 0 <= coarse fraction <= coarse max fraction <= 1")
        self.detail_gate = nn.Sequential(
            nn.Conv2d(3, 24, 3, padding=1), nn.SiLU(),
            nn.Conv2d(24, 1, 3, padding=1),
        )
        # Start conservatively: sigmoid(-2) ~= 0.12, so an untrained Phase 2
        # cannot destroy the already useful Phase-1 forecast.
        nn.init.zeros_(self.detail_gate[-1].weight)
        nn.init.constant_(self.detail_gate[-1].bias, -2.0)
        self.register_buffer("residual_std", torch.tensor(0.05, dtype=torch.float32))

        steps = torch.arange(self.timesteps + 1, dtype=torch.float64)
        curve = torch.cos(((steps / self.timesteps + 0.008) / 1.008) * math.pi / 2).square()
        curve = curve / curve[0]
        beta = (1.0 - curve[1:] / curve[:-1]).clamp(1e-4, 0.999).float()
        alpha_bar = torch.cumprod(1.0 - beta, dim=0)
        self.register_buffer("beta", beta)
        self.register_buffer("alpha_bar", alpha_bar)

    def set_residual_std(self, value):
        value = max(float(value), 1e-4)
        self.residual_std.fill_(value)

    def phase1(self, low, target_size):
        return self.regression(low, target_size)

    def _project_multiscale_detail(self, detail, low_size):
        coarse = F.interpolate(detail, size=low_size, mode="area")
        coarse_up = F.interpolate(coarse, size=detail.shape[-2:], mode="bilinear", align_corners=True)
        high_frequency = detail - coarse_up
        strength = (coarse_up.abs() / (2.0 * self.residual_std)).clamp(0.0, 1.0)
        fraction = (self.coarse_residual_fraction
                    + (self.coarse_residual_max_fraction - self.coarse_residual_fraction) * strength)
        return high_frequency + fraction * coarse_up

    def high_frequency_target(self, prior, truth, low_size):
        # Kept as a public name for training-code compatibility.  The denoiser
        # learns the complete residual; the allowed coarse-scale fraction is
        # applied exactly once when the residual is composed with Phase 1.
        return truth - prior.detach()

    @staticmethod
    def _condition(prior, baseline):
        return torch.cat((prior, baseline), dim=1)

    def q_sample(self, clean_residual, timestep, noise):
        abar = _extract(self.alpha_bar, timestep, clean_residual.shape)
        return abar.sqrt() * clean_residual + (1.0 - abar).sqrt() * noise

    def _v_target(self, clean_residual, noise, timestep):
        abar = _extract(self.alpha_bar, timestep, clean_residual.shape)
        return abar.sqrt() * noise - (1.0 - abar).sqrt() * clean_residual

    def _x0_from_v(self, noisy, predicted_v, timestep):
        abar = _extract(self.alpha_bar, timestep, noisy.shape)
        return abar.sqrt() * noisy - (1.0 - abar).sqrt() * predicted_v

    def _eps_from_v(self, noisy, predicted_v, timestep):
        abar = _extract(self.alpha_bar, timestep, noisy.shape)
        return (1.0 - abar).sqrt() * noisy + abar.sqrt() * predicted_v

    def compose(self, prior, baseline, normalized_detail, low_size):
        normalized_detail = normalized_detail.clamp(-self.residual_clip, self.residual_clip)
        physical_detail = normalized_detail * self.residual_std
        gate_raw = torch.sigmoid(self.detail_gate(torch.cat((prior, baseline, physical_detail.abs()), dim=1)))
        gate = self.gate_min + (self.gate_max - self.gate_min) * gate_raw
        gated_detail = self._project_multiscale_detail(gate * physical_detail, low_size)
        return (prior + gated_detail).clamp(0.0, 1.0), gate, gated_detail

    @staticmethod
    def _soft_fss(prediction, truth, threshold, window, sharpness=24.0):
        pred_event = torch.sigmoid((prediction - threshold) * sharpness)
        truth_event = torch.sigmoid((truth - threshold) * sharpness)
        pred_fraction = F.avg_pool2d(pred_event, window, stride=1, padding=window // 2)
        truth_fraction = F.avg_pool2d(truth_event, window, stride=1, padding=window // 2)
        numerator = (pred_fraction - truth_fraction).square().mean()
        denominator = pred_fraction.square().mean() + truth_fraction.square().mean() + 1e-6
        return numerator / denominator

    def diffusion_loss(self, prior, baseline, truth, low_size, mask=None,
                       x0_weight=0.5, reconstruction_weight=1.0,
                       consistency_weight=0.5, gradient_weight=0.05,
                       structure_weight=0.2, heavy_weight=5.0,
                       fss_weight=1.0, gate_weight=0.25):
        target_detail = self.high_frequency_target(prior, truth, low_size)
        clean = (target_detail / self.residual_std).clamp(-self.residual_clip, self.residual_clip)
        timestep = torch.randint(0, self.timesteps, (truth.shape[0],), device=truth.device)
        noise = torch.randn_like(clean)
        noisy = self.q_sample(clean, timestep, noise)
        condition = self._condition(prior.detach(), baseline.detach())
        predicted_v = self.denoiser(noisy, timestep, condition)
        target_v = self._v_target(clean, noise, timestep)
        predicted_x0 = self._x0_from_v(noisy, predicted_v, timestep).clamp(
            -self.residual_clip, self.residual_clip
        )

        def masked_mean(value):
            if mask is None:
                return value.mean()
            return (value * mask).sum() / mask.sum().clamp_min(1.0)

        # Log-normalized thresholds corresponding to 10, 25 and 50 mm/day.
        threshold_10 = math.log1p(10.0) / math.log1p(250.0)
        threshold_25 = math.log1p(25.0) / math.log1p(250.0)
        threshold_50 = math.log1p(50.0) / math.log1p(250.0)
        diffusion_weight = 1.0 + 2.0 * (truth >= threshold_10).float()
        diffusion_weight = diffusion_weight + heavy_weight * (truth >= threshold_25).float()
        diffusion_weight = diffusion_weight + heavy_weight * (truth >= threshold_50).float()
        # Heavy-rain pixels are scarce. Weight both the diffusion objective and
        # x0 reconstruction so their residual distribution is not overwhelmed
        # by dry/light-rain pixels before the end-to-end losses are evaluated.
        v_loss = masked_mean((predicted_v - target_v).square() * diffusion_weight)
        x0_loss = masked_mean(
            F.smooth_l1_loss(predicted_x0, clean, reduction="none") * diffusion_weight
        )
        final, gate, _ = self.compose(prior.detach(), baseline.detach(), predicted_x0, low_size)
        intensity_weight = 1.0 + 3.0 * truth.square()
        intensity_weight = intensity_weight + heavy_weight * (truth >= threshold_25).float()
        intensity_weight = intensity_weight + 0.5 * heavy_weight * (truth >= threshold_50).float()
        reconstruction = masked_mean(
            F.smooth_l1_loss(final, truth, reduction="none") * intensity_weight
        )
        final_coarse = F.interpolate(final, size=low_size, mode="area")
        prior_coarse = F.interpolate(prior.detach(), size=low_size, mode="area")
        consistency = F.l1_loss(final_coarse, prior_coarse)
        grad_x = (final[..., :, 1:] - final[..., :, :-1]) - (truth[..., :, 1:] - truth[..., :, :-1])
        grad_y = (final[..., 1:, :] - final[..., :-1, :]) - (truth[..., 1:, :] - truth[..., :-1, :])
        gradient = 0.5 * (grad_x.abs().mean() + grad_y.abs().mean())
        structure = 0.0
        for window in (3, 9):
            final_fraction = F.avg_pool2d(final, window, stride=1, padding=window // 2)
            truth_fraction = F.avg_pool2d(truth, window, stride=1, padding=window // 2)
            structure = structure + F.l1_loss(final_fraction, truth_fraction)
        structure = structure / 2.0
        fss = 0.0
        for threshold in (threshold_10, threshold_25, threshold_50):
            for window in (3, 9):
                fss = fss + self._soft_fss(final, truth, threshold, window)
        fss = fss / 6.0
        target_strength = (target_detail.abs() / (2.0 * self.residual_std)).clamp(0.0, 1.0)
        target_gate = torch.maximum(target_strength, 0.80 * (truth >= threshold_25).float())
        target_gate = torch.maximum(target_gate, (truth >= threshold_50).float())
        gate_supervision = masked_mean(F.smooth_l1_loss(gate, target_gate, reduction="none"))
        total = (v_loss + x0_weight * x0_loss + reconstruction_weight * reconstruction
                 + consistency_weight * consistency + gradient_weight * gradient
                 + structure_weight * structure + fss_weight * fss
                 + gate_weight * gate_supervision)
        return total, {"v": v_loss.detach(), "x0": x0_loss.detach(),
                       "reconstruction": reconstruction.detach(), "consistency": consistency.detach(),
                       "structure": structure.detach(), "fss": fss.detach(),
                       "gate_loss": gate_supervision.detach(), "gate": gate.mean().detach()}

    @torch.no_grad()
    def sample_residual_ddim(self, prior, baseline, generator=None):
        sample = torch.randn(prior.shape, device=prior.device, dtype=prior.dtype, generator=generator)
        condition = self._condition(prior, baseline)
        step_indices = torch.linspace(0, self.timesteps - 1, self.sampling_steps, device=prior.device)
        step_indices = torch.unique(step_indices.round().long(), sorted=True)
        for position in reversed(range(len(step_indices))):
            step = int(step_indices[position].item())
            timestep = torch.full((prior.shape[0],), step, device=prior.device, dtype=torch.long)
            predicted_v = self.denoiser(sample, timestep, condition)
            predicted_x0 = self._x0_from_v(sample, predicted_v, timestep).clamp(
                -self.residual_clip, self.residual_clip
            )
            if position == 0:
                sample = predicted_x0
                continue
            previous_step = int(step_indices[position - 1].item())
            previous_abar = self.alpha_bar[previous_step].reshape(1, 1, 1, 1)
            predicted_noise = self._eps_from_v(sample, predicted_v, timestep)
            # DDIM eta=0: deterministic and free of per-step noise amplification.
            sample = previous_abar.sqrt() * predicted_x0 + (1.0 - previous_abar).sqrt() * predicted_noise
        return sample.clamp(-self.residual_clip, self.residual_clip)

    @torch.no_grad()
    def super_resolve(self, low, target_size, generator=None):
        baseline, prior = self.phase1(low, target_size)
        normalized_detail = self.sample_residual_ddim(prior, baseline, generator=generator)
        final, gate, detail = self.compose(prior, baseline, normalized_detail, low.shape[-2:])
        return baseline, prior, final, gate, detail
