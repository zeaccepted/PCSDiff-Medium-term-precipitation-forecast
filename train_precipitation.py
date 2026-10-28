import config_server as config
import argparse
import os
import random
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from modules.deterministic_correction import DeterministicCorrection
from modules.trainer import Trainer
from torch.nn.parallel import DistributedDataParallel as DDP
from data.transposed_collate import precipitation_collate_fn

# 固定随机种子，确保基线模型和新模型训练可复现
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


parser = argparse.ArgumentParser(description="values from bash script")
parser.add_argument("--ndevice", type=int, required=True, help="number of cuda device")
args = parser.parse_args()


def schedule_func(ep):
    # 余弦退火学习率调度（与test目录一致，保持训练收敛风格相同）
    import numpy as np
    return 0.5 * (1 + np.cos(np.pi * ep / config.n_step))


def main(rank, world_size):
    # 手动设置环境变量
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12356'  # 修改端口号，避免冲突
    
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    from data.load_dataset import load_dataset
    train_dataset, val_dataset = load_dataset(config.data_config)

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if world_size > 1 else None
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False) if world_size > 1 else None

    train_data = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=False if train_sampler is not None else True,
        collate_fn=precipitation_collate_fn,
        num_workers=0,
        pin_memory=False,
        sampler=train_sampler,
    )
    val_data = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=precipitation_collate_fn,
        num_workers=0,
        pin_memory=False,
        sampler=val_sampler,
    )

    # ===== Step 2 SimpleUNet + PIMD =====
    # DeterministicCorrection 内部创建 SimpleUNetBackbone + PIMD模块
    model = DeterministicCorrection(
        # SimpleUNet核心超参
        simple_unet_dim=getattr(config, "embed_dim", 64),    # 64
        n_ensemble_channels=getattr(config, "model_input_channels", 10),  # 10
        n_residual_channels=config.data_config["output_channel"],          # 1
        # 训练损失参数
        loss_type=getattr(config, "loss_type", "l2"),
        huber_delta=getattr(config, "huber_delta", 1.0),
        use_ensemble_spread=getattr(config, "use_ensemble_spread", False),
        pred_mode=getattr(config, "pred_mode", "pred_true"),
        clip_noise=getattr(config, "clip_noise", True),
        aux_loss=getattr(config, "aux_loss", False),
        # clip/weight参数（保留传参兼容）
        residual_blend_weight=getattr(config, "residual_blend_weight", 1.0),
        max_abs_residual_norm=getattr(config, "max_abs_residual_norm", 1.0),
        positive_residual_clip_value=getattr(config, "positive_residual_clip_value", 1.0),
        negative_residual_clip_value=getattr(config, "negative_residual_clip_value", 1.0),
        # 归一化参数
        forecast_min=train_dataset.forecast_min,
        forecast_max=train_dataset.forecast_max,
        truth_min=train_dataset.truth_min,
        truth_max=train_dataset.truth_max,
        # ===== PIMD参数 =====
        use_pimd=getattr(config, "use_pimd", True),
        pimd_num_branches=getattr(config, "pimd_num_branches", 3),
        pimd_temperature=getattr(config, "pimd_temperature", 1.0),
        pimd_entropy_loss_weight=getattr(config, "pimd_entropy_loss_weight", 0.01),
    ).to(rank)

    model = DDP(model, device_ids=[rank])

    trainer = Trainer(
        rank=rank,
        diffusion_model=model,
        train_dl=train_data,
        val_dl=val_data,
        sample_num_of_frame=config.init_num_of_frame + 1,
        init_num_of_frame=config.init_num_of_frame,
        scheduler_function=schedule_func,
        scheduler_checkpoint_step=config.scheduler_checkpoint_step,
        ema_decay=config.ema_decay,
        train_lr=config.lr,
        train_num_steps=config.n_step,
        step_start_ema=config.ema_start_step,
        update_ema_every=config.ema_step,
        save_and_sample_every=config.log_checkpoint_step,
        results_folder=os.path.join(config.current_run_dir, f"{config.model_name}/"),
        tensorboard_dir=os.path.join(config.tensorboard_run_dir, f"{config.model_name}/"),
        model_name=config.model_name,
        val_num_of_batch=config.val_num_of_batch,
        optimizer=config.optimizer,
        dataset=train_dataset,  # 传递训练数据集对象
        val_dataset=val_dataset  # 传递验证数据集对象
    )

    if config.load_model:
        trainer.load(load_step=config.load_step)

    trainer.train()


if __name__ == "__main__":
    # 当只有一个GPU时，不使用分布式训练
    if args.ndevice == 1:
        # 直接运行main函数，不使用spawn
        main(0, 1)
    else:
        mp.spawn(main, args=(args.ndevice,), nprocs=args.ndevice, join=True)
        try:
            dist.barrier()
            dist.destroy_process_group()
        except Exception:
            pass
