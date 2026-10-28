"""独立训练 TP(10)+29 个气象辅助变量的 39 通道模型。"""
import argparse
import os
import random

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

import config_server as config
from data.load_dataset_multivar import load_multivar_dataset
from data.transposed_collate import precipitation_collate_fn
from modules.deterministic_correction import DeterministicCorrection
from modules.trainer import Trainer

N_INPUT_CHANNELS = 39  # TP×10 + 2t/MSL/GH/q/sp/u/v/w×29；不含 ttr
MODEL_SUFFIX = "-multivar-tp10-allmet29-ch39"

parser = argparse.ArgumentParser()
parser.add_argument("--ndevice", type=int, required=True)
args = parser.parse_args()


def schedule_func(step):
    return 0.5 * (1 + np.cos(np.pi * step / config.n_step))


def main(rank, world_size):
    seed = 42
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12357"  # 与主模型训练端口分离
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    train_ds, val_ds = load_multivar_dataset(config.data_config)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_ds) if world_size > 1 else None
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_ds, shuffle=False) if world_size > 1 else None
    train_dl = DataLoader(train_ds, batch_size=config.batch_size, shuffle=train_sampler is None,
                          sampler=train_sampler, collate_fn=precipitation_collate_fn, num_workers=0, pin_memory=False)
    val_dl = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, sampler=val_sampler,
                        collate_fn=precipitation_collate_fn, num_workers=0, pin_memory=False)

    model = DeterministicCorrection(
        simple_unet_dim=config.embed_dim, n_ensemble_channels=N_INPUT_CHANNELS,
        n_residual_channels=config.data_config["output_channel"], loss_type=config.loss_type,
        huber_delta=config.huber_delta, use_ensemble_spread=config.use_ensemble_spread,
        pred_mode=config.pred_mode, clip_noise=config.clip_noise, aux_loss=config.aux_loss,
        residual_blend_weight=config.residual_blend_weight, max_abs_residual_norm=config.max_abs_residual_norm,
        positive_residual_clip_value=config.positive_residual_clip_value,
        negative_residual_clip_value=config.negative_residual_clip_value,
        forecast_min=train_ds.forecast_min, forecast_max=train_ds.forecast_max,
        truth_min=train_ds.truth_min, truth_max=train_ds.truth_max,
        use_pimd=config.use_pimd, pimd_num_branches=config.pimd_num_branches,
        pimd_temperature=config.pimd_temperature, pimd_entropy_loss_weight=config.pimd_entropy_loss_weight,
    ).to(rank)
    model = DDP(model, device_ids=[rank])
    model_name = f"{config.model_name}{MODEL_SUFFIX}"
    results_folder = os.path.join(config.current_run_dir, model_name)
    trainer = Trainer(
        rank=rank, diffusion_model=model, train_dl=train_dl, val_dl=val_dl,
        sample_num_of_frame=config.init_num_of_frame + 1, init_num_of_frame=config.init_num_of_frame,
        scheduler_function=schedule_func, scheduler_checkpoint_step=config.scheduler_checkpoint_step,
        ema_decay=config.ema_decay, train_lr=config.lr, train_num_steps=config.n_step,
        step_start_ema=config.ema_start_step, update_ema_every=config.ema_step,
        save_and_sample_every=config.log_checkpoint_step, results_folder=results_folder,
        tensorboard_dir=os.path.join(config.tensorboard_run_dir, model_name), model_name=model_name,
        val_num_of_batch=config.val_num_of_batch, optimizer=config.optimizer,
        dataset=train_ds, val_dataset=val_ds,
    )
    trainer.train()
    if rank == 0:
        np.savez(os.path.join(results_folder, "aux_normalization_params.npz"), **train_ds.get_aux_normalization())
        print("[多变量训练] 已保存仅由训练集得到的辅助变量归一化参数。")


if __name__ == "__main__":
    if args.ndevice == 1:
        main(0, 1)
    else:
        mp.spawn(main, args=(args.ndevice,), nprocs=args.ndevice, join=True)
