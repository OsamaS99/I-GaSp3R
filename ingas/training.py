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
from ingas.model import InGaS  # Import the model from model.py
from spann3r.datasets import build_dataset  # Ensure this module is available
import croco.utils.misc as misc  # Assuming you have a misc utils module
import utils.loss_mask as loss_mask

import wandb  # Import wandb for logging

def get_args_parser():
    parser = argparse.ArgumentParser(description='InGaS Training Script', add_help=False)
    
    # Model & Checkpoints
    parser.add_argument('--spann3r_checkpoint_path', type=str, default = "/home/team13/Workspace/ingas_osama/checkpoints/spann3r.pth", help='Path to Spann3R checkpoint')
    parser.add_argument('--dust3r_checkpoint_path', type=str, default="/home/team13/Workspace/ingas_osama/checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth", help='Path to Dust3R checkpoint')

    # Datasets
    parser.add_argument('--train_dataset', type=str,default= "10000 @ Scannetpp(split='train', ROOT='/mnt/hdd/scannetpp/data', resolution=224, transform=ImgNorm)" , help='Training dataset')
    parser.add_argument('--val_dataset', type=str, default="Scannetpp(split='val', ROOT='/mnt/hdd/scannetpp/data', resolution=224, num_seq=1, kf_every=10, seed=777, full_video=True)", help='Validation dataset')

    # Experiment parameters
    parser.add_argument('--seed', default=0, type=int, help="Random seed")
    parser.add_argument('--output_dir', type=str, default='./output', help='Directory to save checkpoints and logs')

    # Training Hyperparameters
    parser.add_argument('--batch_size', default=1, type=int, help='Batch size per GPU')
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.05, help="Weight decay")
    parser.add_argument('--gradient_clip_val', type=float, default=0.5, help="Gradient clipping value")
    parser.add_argument('--warmup_epochs', type=int, default=10, help='Epochs to warmup LR')
    parser.add_argument('--eval_freq', type=int, default=1, help='Evaluation frequency (in epochs)')
    parser.add_argument('--save_freq', type=int, default=1, help='Save frequency (in epochs)')
    parser.add_argument('--print_freq', type=int, default=10, help='Print frequency')
    parser.add_argument('--accum_iter', default=1, type=int, help="Accumulate gradient iterations")

    # Loss Parameters
    parser.add_argument('--apply_mask', action='store_true', help='Apply mask during loss computation')
    parser.add_argument('--average_over_mask', action='store_true', help='Average loss over mask')
    parser.add_argument('--mse_loss_weight', type=float, default=1.0, help='MSE loss weight')
    parser.add_argument('--lpips_loss_weight', type=float, default=0.25, help='LPIPS loss weight')
    parser.add_argument('--spann3r_loss_weight', type=float, default=None, help='Spann3r loss weight')


    # Distributed training
    parser.add_argument('--dist_url', default='env://', help='URL used to set up distributed training')
    parser.add_argument('--world_size', default=1, type=int, help='Number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    
    # Logging
    parser.add_argument('--wandb_project', type=str, default='InGaS', help='wandb project name')
    parser.add_argument('--wandb_run_name', type=str, default='InGaS_Demo', help='wandb run name')

    return parser

def init_distributed_mode(args):
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
        for view in batch:
            for name in 'img original_img pts3d valid_mask camera_pose camera_intrinsics F_matrix corres depthmap'.split():  # pseudo_focal
                if name not in view:
                    continue
                view[name] = view[name].to(device, non_blocking=True)

        # Forward pass
        preds, preds_all = model(batch)

        # Decode and render predictions
        _, _, h, w = batch[0]['img'].shape
        color, _ = model.decoder(batch, preds, (h, w))

        # Calculate losses with mask
        mask = loss_mask.calculate_loss_mask(batch)
        loss, mse_loss, lpips_loss = model.calculate_loss(
            batch, preds_all, color, mask,
            apply_mask=args.apply_mask,
            average_over_mask=args.average_over_mask,
            calculate_ssim=False
        )

        # Backward pass and optimization
        loss_value = loss.item()
        optimizer.zero_grad()
        loss_scaler.scale(loss).backward()
        loss_scaler.step(optimizer)
        loss_scaler.update()

        # Logging
        if args.rank == 0:
            wandb.log({
                'train/loss': loss_value,
                'train/mse': mse_loss.item(),
                'train/lpips': lpips_loss.item(),
                # 'train/epoch': epoch + batch['idx'] / len(data_loader),
            })

        metric_logger.update(loss=loss_value, mse=mse_loss.item(), lpips=lpips_loss.item())

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def validate(model, data_loader, device, epoch, args):
    model.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = f'Validation Epoch: [{epoch}]'

    with torch.no_grad():
        for batch in metric_logger.log_every(data_loader, args.print_freq, header):
            for view in batch:
                for name in 'img original_img pts3d valid_mask camera_pose camera_intrinsics F_matrix corres depthmap'.split():  # pseudo_focal
                    if name not in view:
                        continue
                    view[name] = view[name].to(device, non_blocking=True)

            # Forward pass
            preds, preds_all = model(batch)

            # Decode and render predictions
            _, _, h, w = batch[0]['img'].shape
            color, _ = model.decoder(batch, preds, (h, w))

            # Calculate losses with mask
            mask = loss_mask.calculate_loss_mask(batch)
            loss, mse_loss, lpips_loss, ssim_val = model.calculate_loss(
                batch, preds_all, color, mask,
                apply_mask=args.apply_mask,
                average_over_mask=args.average_over_mask,
                calculate_ssim=True
            )

            # Logging
            if args.rank == 0:
                wandb.log({
                    'val/loss': loss.item(),
                    'val/mse': mse_loss.item(),
                    'val/lpips': lpips_loss.item(),
                    'val/ssim': ssim_val.item(),
                    'val/epoch': epoch,
                })

            metric_logger.update(loss=loss.item(), mse=mse_loss.item(), lpips=lpips_loss.item(), ssim=ssim_val.item())

    metric_logger.synchronize_between_processes()
    print("Validation stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def main(args):
    init_distributed_mode(args)
    global_rank = misc.get_rank()
    world_size = misc.get_world_size()

    device = torch.device(f'cuda:{args.gpu}')
    torch.manual_seed(args.seed + args.rank)
    np.random.seed(args.seed + args.rank)
    cudnn.benchmark = True

    if args.rank == 0:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=args,
            dir=args.output_dir,
        )

    model = InGaS(
        spann3r_checkpoint_path=args.spann3r_checkpoint_path,
        dust3r_checkpoint_path=args.dust3r_checkpoint_path,
        config=args
    )
    model.to(device)
    print(args.distributed)
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)

    dataset_train = build_dataset(args.train_dataset, 1, test=False, num_workers=1)
    dataset_val = build_dataset(args.val_dataset, 1, test=True, num_workers=1)

    dataset_train.dataset.set_epoch(1)
    dataset_train.dataset.set_ratio(1.0)

    dataset_val.dataset.set_epoch(1)
    dataset_val.dataset.set_ratio(1.0)


    data_loader_train = dataset_train
    data_loader_val = dataset_val

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = MultiStepLR(optimizer, milestones=[args.epochs // 2], gamma=0.1)
    loss_scaler = GradScaler()

    for epoch in range(args.epochs):
        data_loader_train.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            model, data_loader_train, optimizer, device, epoch, loss_scaler, args
        )
        scheduler.step()

        if epoch % args.eval_freq == 0 or epoch == args.epochs - 1:
            val_stats = validate(model, data_loader_val, device, epoch, args)

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
    parser = argparse.ArgumentParser(description='InGaS Training Script', parents=[get_args_parser()])
    args = parser.parse_args()

    main(args)
