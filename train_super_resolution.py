"""Train controlled multiscale conditional DDIM while reusing Phase 1."""
from __future__ import annotations

import json
import math
import os

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

import config_server as config
from data.datasets.super_resolution import SuperResolutionDataset
from modules.super_resolution_model import ConditionalDenoisingUNet, RegressionConvResNet, TwoStageSuperResolution


def _paths(cfg):
    root = os.environ.get("RVD_SR_RESULT_DIR", cfg["result_dir"])
    train_nc = os.environ.get("RVD_SR_TRAIN_CORRECTION", cfg["train_correction_file"])
    test_nc = os.environ.get("RVD_SR_TEST_CORRECTION", cfg["test_correction_file"])
    return root, train_nc, test_nc


def _dataset(cfg, train_nc, test_nc, mode):
    return SuperResolutionDataset(train_nc, test_nc, cfg["truth_dir"], mode,
                                  validation_fraction=cfg["validation_fraction"],
                                  lat_range=cfg["lat_range"], lon_range=cfg["lon_range"],
                                  normalization_max_mm=cfg["normalization_max_mm"])


def _build_model(cfg):
    return TwoStageSuperResolution(
        RegressionConvResNet(cfg["regression_channels"], cfg["regression_blocks"], cfg["regression_residual_scale"]),
        ConditionalDenoisingUNet(cfg["diffusion_channels"], condition_channels=2), cfg["timesteps"],
        cfg["diffusion_residual_clip"], cfg["ddim_sampling_steps"],
        cfg["coarse_residual_fraction"], cfg["coarse_residual_max_fraction"],
        cfg["phase2_gate_min"], cfg["phase2_gate_max"]
    )


def _masked_mean(value, mask):
    return (value * mask).sum() / mask.sum().clamp_min(1.0)


def _phase1_loss(prior, truth, mask):
    intensity_weight = 1.0 + 3.0 * truth.square()
    pixel = F.smooth_l1_loss(prior, truth, reduction="none") * intensity_weight
    grad_x = (prior[..., :, 1:] - prior[..., :, :-1]) - (truth[..., :, 1:] - truth[..., :, :-1])
    grad_y = (prior[..., 1:, :] - prior[..., :-1, :]) - (truth[..., 1:, :] - truth[..., :-1, :])
    return _masked_mean(pixel, mask) + 0.05 * (grad_x.abs().mean() + grad_y.abs().mean())


@torch.no_grad()
def _validate_phase1(model, loader, device):
    model.eval(); total = 0.0
    for batch in loader:
        low, truth, mask = (batch[k].to(device, non_blocking=True) for k in ("low", "truth", "land_mask"))
        _, prior = model.phase1(low, truth.shape[-2:])
        total += _phase1_loss(prior, truth, mask).item()
    return total / max(len(loader), 1)


def _load_regression_only(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state = checkpoint.get("model", checkpoint)
    prefix = "regression."
    regression_state = {key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)}
    if not regression_state:
        raise KeyError(f"No regression.* weights in {checkpoint_path}")
    model.regression.load_state_dict(regression_state, strict=True)
    print(f"[Phase1] reused frozen checkpoint: {checkpoint_path}")


@torch.no_grad()
def _estimate_residual_std(model, loader, device, max_batches):
    model.eval(); sum_sq = 0.0; count = 0.0
    for batch_index, batch in enumerate(tqdm(loader, desc="Residual scale")):
        low, truth, mask = (batch[k].to(device, non_blocking=True) for k in ("low", "truth", "land_mask"))
        _, prior = model.phase1(low, truth.shape[-2:])
        detail = model.high_frequency_target(prior, truth, low.shape[-2:])
        sum_sq += float((detail.square() * mask).sum().item())
        count += float(mask.sum().item())
        if batch_index + 1 >= max_batches:
            break
    value = math.sqrt(sum_sq / max(count, 1.0))
    model.set_residual_std(value)
    print(f"[Phase2] training-only high-frequency residual std={value:.8f}")
    return value


def _diffusion_kwargs(cfg):
    return dict(x0_weight=cfg["sr_x0_loss_weight"],
                reconstruction_weight=cfg["sr_reconstruction_loss_weight"],
                consistency_weight=cfg["sr_consistency_loss_weight"],
                gradient_weight=cfg["sr_gradient_loss_weight"],
                structure_weight=cfg["sr_structure_loss_weight"],
                heavy_weight=cfg["sr_heavy_rain_weight"],
                fss_weight=cfg["sr_fss_loss_weight"],
                gate_weight=cfg["sr_gate_loss_weight"])


@torch.no_grad()
def _validate_diffusion_loss(model, loader, device, cfg):
    model.eval(); total = 0.0; count = 0
    for batch in loader:
        low, truth, mask = (batch[k].to(device, non_blocking=True) for k in ("low", "truth", "land_mask"))
        baseline, prior = model.phase1(low, truth.shape[-2:])
        loss, _ = model.diffusion_loss(prior, baseline, truth, low.shape[-2:], mask, **_diffusion_kwargs(cfg))
        total += loss.item(); count += 1
        if count >= cfg["diffusion_val_batches"]:
            break
    return total / max(count, 1)


@torch.no_grad()
def _validate_end_to_end(model, loader, dataset, device, cfg):
    model.eval()
    generator = torch.Generator(device=device).manual_seed(cfg["sampling_seed"])
    totals = {"phase1_sq": 0.0, "phase2_sq": 0.0, "phase1_abs": 0.0,
              "phase2_abs": 0.0, "phase1_heavy_sq": 0.0, "phase2_heavy_sq": 0.0,
              "heavy_pixels": 0.0, "gate": 0.0, "pixels": 0.0, "batches": 0}
    for label in ("phase1", "phase2"):
        for threshold in (10, 25):
            for term in ("tp", "fp", "fn"):
                totals[f"{label}_{term}_{threshold}"] = 0.0
            for window in (3, 9):
                totals[f"{label}_fss_num_{threshold}_{window}"] = 0.0
                totals[f"{label}_fss_den_{threshold}_{window}"] = 0.0
    for batch in loader:
        low, truth, mask = (batch[k].to(device, non_blocking=True) for k in ("low", "truth", "land_mask"))
        _, prior, final, gate, _ = model.super_resolve(low, truth.shape[-2:], generator)
        truth_mm = dataset.denormalize_tensor(truth)
        prior_mm = dataset.denormalize_tensor(prior)
        final_mm = dataset.denormalize_tensor(final)
        totals["phase1_sq"] += float(((prior_mm - truth_mm).square() * mask).sum().item())
        totals["phase2_sq"] += float(((final_mm - truth_mm).square() * mask).sum().item())
        totals["phase1_abs"] += float(((prior_mm - truth_mm).abs() * mask).sum().item())
        totals["phase2_abs"] += float(((final_mm - truth_mm).abs() * mask).sum().item())
        heavy_mask = mask * (truth_mm >= 10.0).float()
        totals["phase1_heavy_sq"] += float(((prior_mm - truth_mm).square() * heavy_mask).sum().item())
        totals["phase2_heavy_sq"] += float(((final_mm - truth_mm).square() * heavy_mask).sum().item())
        totals["heavy_pixels"] += float(heavy_mask.sum().item())
        for label, prediction in (("phase1", prior_mm), ("phase2", final_mm)):
            for threshold in (10, 25):
                pred_event = prediction >= float(threshold)
                truth_event = truth_mm >= float(threshold)
                valid = mask > 0.5
                totals[f"{label}_tp_{threshold}"] += float((pred_event & truth_event & valid).sum().item())
                totals[f"{label}_fp_{threshold}"] += float((pred_event & (~truth_event) & valid).sum().item())
                totals[f"{label}_fn_{threshold}"] += float(((~pred_event) & truth_event & valid).sum().item())
                for window in (3, 9):
                    pred_fraction = F.avg_pool2d(pred_event.float() * mask, window, stride=1,
                                                 padding=window // 2)
                    truth_fraction = F.avg_pool2d(truth_event.float() * mask, window, stride=1,
                                                  padding=window // 2)
                    totals[f"{label}_fss_num_{threshold}_{window}"] += float(
                        (((pred_fraction - truth_fraction).square()) * mask).sum().item()
                    )
                    totals[f"{label}_fss_den_{threshold}_{window}"] += float(
                        ((pred_fraction.square() + truth_fraction.square()) * mask).sum().item()
                    )
        totals["gate"] += float(gate.mean().item()); totals["pixels"] += float(mask.sum().item())
        totals["batches"] += 1
    pixels = max(totals["pixels"], 1.0)
    result = {"phase1_rmse": math.sqrt(totals["phase1_sq"] / pixels),
              "phase2_rmse": math.sqrt(totals["phase2_sq"] / pixels),
              "phase1_mae": totals["phase1_abs"] / pixels,
              "phase2_mae": totals["phase2_abs"] / pixels,
              "phase1_heavy_rmse": math.sqrt(totals["phase1_heavy_sq"] / max(totals["heavy_pixels"], 1.0)),
              "phase2_heavy_rmse": math.sqrt(totals["phase2_heavy_sq"] / max(totals["heavy_pixels"], 1.0)),
              "gate": totals["gate"] / max(totals["batches"], 1)}
    for label in ("phase1", "phase2"):
        for threshold in (10, 25):
            denominator = sum(totals[f"{label}_{term}_{threshold}"] for term in ("tp", "fp", "fn"))
            result[f"{label}_csi_{threshold}"] = totals[f"{label}_tp_{threshold}"] / max(denominator, 1.0)
            for window in (3, 9):
                numerator = totals[f"{label}_fss_num_{threshold}_{window}"]
                fss_denominator = totals[f"{label}_fss_den_{threshold}_{window}"]
                result[f"{label}_fss_{threshold}_w{window}"] = 1.0 - numerator / max(fss_denominator, 1e-8)
    return result


def main():
    cfg = config.sr_config
    result_dir, train_nc, test_nc = _paths(cfg)
    os.makedirs(result_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg["sampling_seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg["sampling_seed"])
    print(f"[SR v3] device={device}; output={result_dir}")
    train_set = _dataset(cfg, train_nc, test_nc, "train")
    val_set = _dataset(cfg, train_nc, test_nc, "val")
    loader_args = dict(batch_size=cfg["batch_size"], num_workers=cfg["num_workers"],
                       pin_memory=torch.cuda.is_available(), persistent_workers=cfg["num_workers"] > 0)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, **loader_args)
    sample_count = min(cfg["end_to_end_val_samples"], len(val_set))
    if sample_count == 1:
        sample_indices = [0]
    else:
        sample_indices = [round(i * (len(val_set) - 1) / (sample_count - 1)) for i in range(sample_count)]
    e2e_val_loader = DataLoader(Subset(val_set, sample_indices), shuffle=False, **loader_args)
    print(f"[E2E] fixed validation subset: {sample_count} lead-days distributed across the full validation period")
    model = _build_model(cfg).to(device)
    history = {"phase1_train": [], "phase1_val": [], "phase2_train": [],
               "phase2_val_loss": [], "phase2_e2e": []}

    phase1_path = os.environ.get("RVD_SR_PHASE1_CHECKPOINT", os.path.join(result_dir, "best_phase1.pt"))
    if cfg["reuse_phase1"] and os.path.isfile(phase1_path):
        _load_regression_only(model, phase1_path, device)
    else:
        print("[Phase1] no reusable checkpoint; training Conv-ResNet")
        optimizer = AdamW(model.regression.parameters(), lr=cfg["phase1_lr"], weight_decay=1e-4)
        best = float("inf")
        for epoch in range(cfg["phase1_epochs"]):
            model.train(); total = 0.0
            for batch in tqdm(train_loader, desc=f"Phase1 {epoch + 1}/{cfg['phase1_epochs']}"):
                low, truth, mask = (batch[k].to(device, non_blocking=True) for k in ("low", "truth", "land_mask"))
                optimizer.zero_grad(set_to_none=True)
                _, prior = model.phase1(low, truth.shape[-2:])
                loss = _phase1_loss(prior, truth, mask); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.regression.parameters(), 1.0); optimizer.step()
                total += loss.item()
            train_loss = total / max(len(train_loader), 1)
            val_loss = _validate_phase1(model, val_loader, device)
            history["phase1_train"].append(train_loss); history["phase1_val"].append(val_loss)
            print(f"[Phase1] epoch={epoch + 1} train={train_loss:.6f} val={val_loss:.6f}")
            if val_loss < best:
                best = val_loss
                torch.save({"model": model.state_dict(), "phase": 1, "config": cfg}, phase1_path)
        _load_regression_only(model, phase1_path, device)

    model.regression.requires_grad_(False).eval()
    residual_std = _estimate_residual_std(model, train_loader, device, cfg["residual_stat_batches"])
    optimizer = AdamW(list(model.denoiser.parameters()) + list(model.detail_gate.parameters()),
                      lr=cfg["phase2_lr"], weight_decay=1e-4)
    best_score = float("inf"); stale_epochs = 0
    checkpoint_path = os.path.join(result_dir, "best_model_v3.pt")
    for epoch in range(cfg["phase2_epochs"]):
        model.denoiser.train(); model.detail_gate.train(); total = 0.0
        component_sums = {key: 0.0 for key in ("v", "x0", "reconstruction", "consistency",
                                               "structure", "fss", "gate_loss", "gate")}
        for batch in tqdm(train_loader, desc=f"Phase2-v3 {epoch + 1}/{cfg['phase2_epochs']}"):
            low, truth, mask = (batch[k].to(device, non_blocking=True) for k in ("low", "truth", "land_mask"))
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                baseline, prior = model.phase1(low, truth.shape[-2:])
            loss, parts = model.diffusion_loss(prior, baseline, truth, low.shape[-2:], mask, **_diffusion_kwargs(cfg))
            loss.backward(); torch.nn.utils.clip_grad_norm_(list(model.denoiser.parameters()) + list(model.detail_gate.parameters()), 1.0)
            optimizer.step(); total += loss.item()
            for key in component_sums:
                component_sums[key] += float(parts[key].item())
        train_loss = total / max(len(train_loader), 1)
        val_loss = _validate_diffusion_loss(model, val_loader, device, cfg)
        e2e = _validate_end_to_end(model, e2e_val_loader, val_set, device, cfg)
        mean_fss10 = 0.5 * (e2e["phase2_fss_10_w3"] + e2e["phase2_fss_10_w9"])
        mean_fss25 = 0.5 * (e2e["phase2_fss_25_w3"] + e2e["phase2_fss_25_w9"])
        score = (e2e["phase2_rmse"] + 0.20 * e2e["phase2_mae"]
                 + 0.15 * e2e["phase2_heavy_rmse"]
                 - 2.0 * e2e["phase2_csi_10"] - 4.0 * e2e["phase2_csi_25"]
                 - 1.0 * mean_fss10 - 1.5 * mean_fss25)
        history["phase2_train"].append(train_loss); history["phase2_val_loss"].append(val_loss)
        history["phase2_e2e"].append(e2e)
        part_text = " ".join(f"{key}={value/max(len(train_loader),1):.5f}" for key, value in component_sums.items())
        print(f"[Phase2-v3] epoch={epoch + 1} train={train_loss:.6f} val={val_loss:.6f} {part_text}")
        print(f"[E2E] P1 RMSE/MAE={e2e['phase1_rmse']:.4f}/{e2e['phase1_mae']:.4f}; "
              f"P2 RMSE/MAE={e2e['phase2_rmse']:.4f}/{e2e['phase2_mae']:.4f}; gate={e2e['gate']:.3f}")
        print(f"[HEAVY] P1/P2 heavy-RMSE={e2e['phase1_heavy_rmse']:.3f}/{e2e['phase2_heavy_rmse']:.3f}; "
              f"CSI10={e2e['phase1_csi_10']:.3f}/{e2e['phase2_csi_10']:.3f}; "
              f"CSI25={e2e['phase1_csi_25']:.3f}/{e2e['phase2_csi_25']:.3f}")
        print(f"[FSS] P1/P2 @10(w3,w9)="
              f"{e2e['phase1_fss_10_w3']:.3f},{e2e['phase1_fss_10_w9']:.3f}/"
              f"{e2e['phase2_fss_10_w3']:.3f},{e2e['phase2_fss_10_w9']:.3f}; "
              f"@25(w3,w9)={e2e['phase1_fss_25_w3']:.3f},{e2e['phase1_fss_25_w9']:.3f}/"
              f"{e2e['phase2_fss_25_w3']:.3f},{e2e['phase2_fss_25_w9']:.3f}")
        if score < best_score:
            best_score = score; stale_epochs = 0
            torch.save({"model": model.state_dict(), "phase": "controlled_multiscale_v_ddim",
                        "config": cfg, "residual_std": residual_std,
                        "normalization_max_mm": cfg["normalization_max_mm"], "e2e": e2e}, checkpoint_path)
            print(f"[Phase2-v3] saved end-to-end best: {checkpoint_path}")
        else:
            stale_epochs += 1
        if stale_epochs >= cfg["phase2_patience"]:
            print(f"[Phase2-v3] early stop after {stale_epochs} epochs without E2E improvement")
            break

    with open(os.path.join(result_dir, "training_history_v3.json"), "w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)
    print(f"[SR v3] complete: {checkpoint_path}")


if __name__ == "__main__":
    main()
