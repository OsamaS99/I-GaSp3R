# train.py

import os
import sys
import math
import json
import time
import torch
import argparse
import datetime
import numpy as np
import torch.backends.cudnn as cudnn

from pathlib import Path
from torch.utils.data.distributed import DistributedSampler
from torch.optim.lr_scheduler import MultiStepLR
from torch.cuda.amp import GradScaler
from torch.optim import Adam

# Import necessary modules
from model import InGaS  # Import the model from model.py
from spann3r.datasets import build_dataset  # Ensure this module is available
import utils.misc as misc  # Assuming you have a misc utils module
import utils.loss_mask as loss_mask

import wandb  # Import wandb for logging

def init_distributed_mode(args):
    """
    Initialize the distributed training environment.
    """
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ['RANK'])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    else:
        print('Not using distributed mode')
        args.distributed = False
        args.world_size = 1
        args.rank = 0
        args.gpu = 0
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(
        backend=args.dist_backend, init_method=args.dist_url,
        world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    misc.setup_for_distributed(args.rank == 0)

def train_one_epoch(model, data_loader, optimizer, device, epoch, loss_scaler, args):
    model.train()
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = f'Epoch: [{epoch}]'

    for batch in metric_logger.log_every(data_loader, args.print_freq, header):
        # Move data to device
        frames = batch['frames']
        frames = [frame.to(device, non_blocking=True) for frame in frames]
        batch = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        # Forward pass
        preds, preds_all = model(frames)

        # Decode and render predictions
        _, _, h, w = frames[0]['img'].shape
        color, _ = model.decoder(batch, preds_all, (h, w))

        # Calculate losses
        mask = loss_mask.calculate_loss_mask(batch)
        loss, mse_loss, lpips_loss = model.calculate_loss(
            batch, preds_all, color, mask,
            apply_mask=args.config['loss']['apply_mask'],
            average_over_mask=args.config['loss']['average_over_mask'],
            calculate_ssim=False
        )

        # Backward pass and optimization
        loss_value = loss.item()
        optimizer.zero_grad()
        loss_scaler.scale(loss).backward()
        loss_scaler.step(optimizer)
        loss_scaler.update()

        # Logging
        if args.rank == 0:  # Log only from the main process
            wandb.log({
                'train/loss': loss_value,
                'train/mse': mse_loss.item(),
                'train/lpips': lpips_loss.item(),
                'train/epoch': epoch + batch['idx'] / len(data_loader),
            })

        metric_logger.update(loss=loss_value, mse=mse_loss.item(), lpips=lpips_loss.item())

    # Gather stats and return
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def validate(model, data_loader, device, epoch, args):
    model.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = f'Validation Epoch: [{epoch}]'

    with torch.no_grad():
        for batch in metric_logger.log_every(data_loader, args.print_freq, header):
            # Move data to device
            frames = batch['frames']
            frames = [frame.to(device, non_blocking=True) for frame in frames]
            batch = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            # Forward pass
            preds, preds_all = model(frames)

            # Decode and render predictions
            _, _, h, w = frames[0]['img'].shape
            color, _ = model.decoder(batch, preds_all, (h, w))

            # Calculate losses
            mask = loss_mask.calculate_loss_mask(batch)
            loss, mse_loss, lpips_loss, ssim_val = model.calculate_loss(
                batch, preds_all, color, mask,
                apply_mask=args.config['loss']['apply_mask'],
                average_over_mask=args.config['loss']['average_over_mask'],
                calculate_ssim=True
            )

            # Logging
            if args.rank == 0:  # Log only from the main process
                wandb.log({
                    'val/loss': loss.item(),
                    'val/mse': mse_loss.item(),
                    'val/lpips': lpips_loss.item(),
                    'val/ssim': ssim_val.item(),
                    'val/epoch': epoch,
                })

            metric_logger.update(loss=loss.item(), mse=mse_loss.item(), lpips=lpips_loss.item(), ssim=ssim_val.item())

    # Gather stats and return
    metric_logger.synchronize_between_processes()
    print("Validation stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def main(args):
    # Initialize distributed mode
    init_distributed_mode(args)

    # Initialize device and seed
    device = torch.device(f'cuda:{args.gpu}')
    seed = args.seed + args.rank
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    # Initialize wandb (only on main process)
    if args.rank == 0:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=args.config,
            dir=args.output_dir,
        )

    # Initialize model
    model = InGaS(
        spann3r_checkpoint_path=args.spann3r_checkpoint_path,
        dust3r_checkpoint_path=args.dust3r_checkpoint_path,
        config=args.config
    )
    model.to(device)

    # Wrap model with DDP
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)

    # Prepare data loaders with DistributedSampler
    dataset_train = build_dataset(args.train_dataset, args, test=False)
    dataset_val = build_dataset(args.val_dataset, args, test=True)

    sampler_train = DistributedSampler(dataset_train, num_replicas=args.world_size, rank=args.rank, shuffle=True)
    sampler_val = DistributedSampler(dataset_val, num_replicas=args.world_size, rank=args.rank, shuffle=False)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Set up optimizer and loss scaler
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    scheduler = MultiStepLR(optimizer, milestones=[args.epochs // 2], gamma=0.1)
    loss_scaler = GradScaler()

    # Training loop
    for epoch in range(args.epochs):
        data_loader_train.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            model, data_loader_train, optimizer, device, epoch, loss_scaler, args
        )

        scheduler.step()

        # Validation step
        if epoch % args.eval_freq == 0 or epoch == args.epochs - 1:
            val_stats = validate(model, data_loader_val, device, epoch, args)

        # Save checkpoint (only from main process)
        if args.rank == 0 and args.output_dir:
            checkpoint_path = os.path.join(args.output_dir, f'checkpoint_epoch_{epoch}.pth')
            torch.save({
                'model': model.module.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
            }, checkpoint_path)

    if args.rank == 0:
        wandb.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='InGaS Training Script with Multi-GPU Support and wandb Logging')
    parser.add_argument('--spann3r_checkpoint_path', type=str, required=True, help='Path to Spann3R checkpoint')
    parser.add_argument('--dust3r_checkpoint_path', type=str, required=True, help='Path to Dust3R checkpoint')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--train_dataset', type=str, required=True, help='Training dataset')
    parser.add_argument('--val_dataset', type=str, required=True, help='Validation dataset')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size per GPU')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers per GPU')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--output_dir', type=str, default='./output', help='Directory to save checkpoints and logs')
    parser.add_argument('--eval_freq', type=int, default=1, help='Evaluation frequency (in epochs)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--print_freq', type=int, default=10, help='Print frequency')
    parser.add_argument('--dist_url', default='env://', help='URL used to set up distributed training')
    parser.add_argument('--wandb_project', type=str, default='InGaS_Project', help='wandb project name')
    parser.add_argument('--wandb_run_name', type=str, default='InGaS_Run', help='wandb run name')
    args = parser.parse_args()

    # Load config file
    with open(args.config, 'r') as f:
        args.config = json.load(f)

    main(args)
