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
import random

import wandb
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import io
from collections import defaultdict
from pathlib import Path
from torch.utils.data.distributed import DistributedSampler
from torch.optim.lr_scheduler import MultiStepLR
from torch.cuda.amp import GradScaler
from torch.optim import Adam
import torchvision.utils as vutils

from PIL import Image
from ingas.model import InGaS
from spann3r.datasets import build_dataset
import croco.utils.misc as misc
import utils.loss_mask as loss_mask

class Visualizer:
    def __init__(self, log_dir, use_wandb=True):
        self.writer = SummaryWriter(log_dir)
        self.use_wandb = use_wandb

    def log_images(self, tag, images, step, nrow=8):
        """Log images to both TensorBoard and W&B"""
        # Convert images to grid
        if isinstance(images, list):
            images = torch.stack([torch.from_numpy(img) if isinstance(img, np.ndarray) 
                                else img for img in images])
        grid = make_grid(images, nrow=nrow, normalize=True)
        
        # Log to TensorBoard
        self.writer.add_image(tag, grid, step)
        
        # Log to W&B
        if self.use_wandb:
            wandb.log({tag: wandb.Image(grid)}, step=step)

    def log_metric(self, tag, value, step):
        """Log scalar metrics to both TensorBoard and W&B"""
        self.writer.add_scalar(tag, value, step)
        if self.use_wandb:
            wandb.log({tag: value}, step=step)

    def log_point_cloud(self, tag, points, colors, step):
        """Log 3D point cloud to both TensorBoard and W&B"""
        if self.use_wandb:
            point_cloud_data = np.hstack((points, colors))
            wandb.log({tag: wandb.Object3D(point_cloud_data)}, step=step)
        
        # For TensorBoard, we'll log a different format
        self.writer.add_embedding(
            points,
            tag=f"{tag}_points",
            global_step=step,
            metadata_header=['x', 'y', 'z']
        )

    def close(self):
        self.writer.close()

def get_args_parser():
    parser = argparse.ArgumentParser(description='InGaS Training Script', add_help=False)
    
    # Model & Checkpoints
    parser.add_argument('--spann3r_checkpoint_path', type=str, 
                       default="/home/team13/Workspace/ingas_osama/checkpoints/spann3r.pth")
    parser.add_argument('--dust3r_checkpoint_path', type=str,
                       default="/home/team13/Workspace/ingas_osama/checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth")

    # Datasets
    parser.add_argument('--train_dataset', type=str,
                       default="1 @ Scannetpp(split='train', ROOT='/mnt/hdd/scannetpp/data', resolution=224, transform=ImgNorm)")
    parser.add_argument('--val_dataset', type=str,
                       default="Scannetpp(split='val', ROOT='/mnt/hdd/scannetpp/data', resolution=224, num_seq=1, kf_every=10, seed=777, full_video=True)")

    # Experiment parameters
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--output_dir', type=str, default='./output')
    parser.add_argument('--log_dir', type=str, default='./logs')

    # Training Hyperparameters
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--lr', type=float, default=5e-6)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--gradient_clip_val', type=float, default=0.5)
    parser.add_argument('--warmup_epochs', type=int, default=10)
    parser.add_argument('--eval_freq', type=int, default=1)
    parser.add_argument('--save_freq', type=int, default=1)
    parser.add_argument('--print_freq', type=int, default=10)
    parser.add_argument('--accum_iter', default=1, type=int)
    parser.add_argument('--num_workers', default=4, type=int)

    # Loss Parameters
    parser.add_argument('--apply_mask', action='store_true')
    parser.add_argument('--average_over_mask', action='store_true')
    parser.add_argument('--mse_loss_weight', type=float, default=1.0)
    parser.add_argument('--lpips_loss_weight', type=float, default=0.25)
    parser.add_argument('--spann3r_loss_weight', type=float, default=None)

    # Distributed training
    parser.add_argument('--dist_url', default='env://')
    parser.add_argument('--world_size', default=1, type=int)
    parser.add_argument('--local_rank', default=-1, type=int)
    
    # Logging
    parser.add_argument('--wandb_project', type=str, default='InGaS')
    parser.add_argument('--wandb_run_name', type=str, default='InGaS_Demo')
    parser.add_argument('--no_wandb', action='store_false', dest='use_wandb', default=True, help='Disable W&B logging')

    return parser

def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ['RANK'])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ.get('LOCAL_RANK', 0))
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
    print(f'| distributed init (rank {args.rank}): {args.dist_url}', flush=True)
    torch.distributed.init_process_group(
        backend=args.dist_backend, init_method=args.dist_url,
        world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    misc.setup_for_distributed(args.rank == 0)

def process_batch(batch, device):
    """Move batch data to device"""
    for view in batch:
        for name in ['img', 'original_img', 'pts3d', 'valid_mask', 'camera_pose',
                    'camera_intrinsics', 'F_matrix', 'corres', 'depthmap']:
            if name in view:
                view[name] = view[name].to(device, non_blocking=True)
    return batch

def train_one_epoch(model, data_loader, optimizer, device, epoch, loss_scaler, args, visualizer):
    model.train()
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = f'Epoch: [{epoch}]'

    for batch_idx, batch in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
        batch = process_batch(batch, device)
        
        # Forward pass
        preds, preds_all = model(batch[1:-1])
        _, _, h, w = batch[0]['img'].shape

        color, _ = model.decoder(batch, preds, (h, w))

        # Calculate losses
        mask = loss_mask.calculate_loss_mask(batch)
        loss, mse_loss, lpips_loss = model.calculate_loss(
            batch, preds_all, color, mask,
            apply_mask=args.apply_mask,
            average_over_mask=args.average_over_mask,
            calculate_ssim=False
        )

        # Optimization
        loss_value = loss.item()
        optimizer.zero_grad()
        loss_scaler.scale(loss).backward()
        loss_scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip_val)
        loss_scaler.step(optimizer)
        loss_scaler.update()

        # Logging
        if args.rank == 0:
            global_step = epoch * len(data_loader) + batch_idx
            visualizer.log_metric('train/loss', loss_value, global_step)
            visualizer.log_metric('train/mse', mse_loss.item(), global_step)
            visualizer.log_metric('train/lpips', lpips_loss.item(), global_step)
            
            # Log sample images every 100 steps
            if batch_idx % 100 == 0:
                input_images = [view['img'][0].cpu() for view in batch]
                visualizer.log_images(
                    'train/input_frames',
                    input_images,
                    global_step,
                    nrow=len(input_images)
                )
                
                rendered_images = [color[0, i].cpu() for i in range(color.shape[1])]
                visualizer.log_images(
                    'train/rendered_frames',
                    rendered_images,
                    global_step,
                    nrow=len(rendered_images)
                )

        metric_logger.update(loss=loss_value, mse=mse_loss.item(), lpips=lpips_loss.item())

    metric_logger.synchronize_between_processes()
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def validate(model, data_loader, device, epoch, args, visualizer):
    model.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = f'Validation Epoch: [{epoch}]'

    with torch.no_grad():
        for batch_idx, batch in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
            batch = process_batch(batch, device)
            
            # Forward pass
            preds, preds_all = model(batch)
            _, _, h, w = batch[0]['img'].shape
            color, _ = model.decoder(batch, preds, (h, w))

            # Calculate losses
            mask = loss_mask.calculate_loss_mask(batch)
            loss, mse_loss, lpips_loss, ssim_val = model.calculate_loss(
                batch, preds_all, color, mask,
                apply_mask=args.apply_mask,
                average_over_mask=args.average_over_mask,
                calculate_ssim=True
            )

            if args.rank == 0:
                global_step = epoch * len(data_loader) + batch_idx
                
                # Log metrics
                visualizer.log_metric('val/loss', loss.item(), global_step)
                visualizer.log_metric('val/mse', mse_loss.item(), global_step)
                visualizer.log_metric('val/lpips', lpips_loss.item(), global_step)
                visualizer.log_metric('val/ssim', ssim_val.item(), global_step)

                # Log visualizations for first batch only
                if batch_idx == 0:
                    # Input frames
                    input_images = [view['img'][0].cpu() for view in batch]
                    visualizer.log_images(
                        f'val/input_frames',
                        input_images,
                        global_step,
                        nrow=len(input_images)
                    )

                    # Rendered frames
                    rendered_images = [color[0, i].cpu() for i in range(color.shape[1])]
                    visualizer.log_images(
                        f'val/rendered_frames',
                        rendered_images,
                        global_step,
                        nrow=len(rendered_images)
                    )

                    # Point cloud visualization
                    for j, view in enumerate(batch):
                        if 'pts3d' in preds[j]:
                            depthmap = view['depthmap'][0].cpu().numpy()
                            mask = depthmap != 0
                            img = view['img'][0].cpu().permute(1, 2, 0).numpy()
                            image = img[mask]
                            pts = preds[j]['pts3d'][0].cpu().numpy()[mask]
                            
                            visualizer.log_point_cloud(
                                f'val/point_cloud_view_{j}',
                                pts,
                                (image + 1.0) / 2.0,
                                global_step
                            )

            metric_logger.update(
                loss=loss.item(),
                mse=mse_loss.item(),
                lpips=lpips_loss.item(),
                ssim=ssim_val.item()
            )

    metric_logger.synchronize_between_processes()
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def main(args):
    init_distributed_mode(args)
    
    # Set up directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # Setup device
    device = torch.device(f'cuda:{args.gpu}')
    torch.manual_seed(args.seed + args.rank)
    np.random.seed(args.seed + args.rank)
    random.seed(args.seed + args.rank)
    cudnn.benchmark = True

    # Initialize logging
    visualizer = None
    if args.rank == 0:
        if args.use_wandb:
            wandb.init(
                project=args.wandb_project,
                config=vars(args),
                dir=args.output_dir,
                name=f"InGaS_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
        visualizer = Visualizer(args.log_dir, use_wandb=args.use_wandb)

    # Create model
    model = InGaS(
        spann3r_checkpoint_path=args.spann3r_checkpoint_path,
        dust3r_checkpoint_path=args.dust3r_checkpoint_path,
        config=args
    ).to(device)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True
        )

    # Setup data loaders
    dataset_train = build_dataset(args.train_dataset, test=False, num_workers=1, batch_size=1)
    dataset_val = build_dataset(args.val_dataset, test=True, num_workers=1, batch_size=1)
    data_loader_train = dataset_train
    data_loader_val = dataset_val

    # Initialize optimizer and schedulers
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    scheduler = MultiStepLR(
        optimizer,
        milestones=[args.epochs // 2],
        gamma=0.1
    )
    
    loss_scaler = GradScaler()

    # Resume from checkpoint if exists
    start_epoch = 0
    if args.output_dir:
        latest_checkpoint = None
        checkpoint_files = sorted(Path(args.output_dir).glob('checkpoint_epoch_*.pth'))
        if checkpoint_files:
            latest_checkpoint = str(checkpoint_files[-1])
            if args.rank == 0:
                print(f"Resuming from checkpoint: {latest_checkpoint}")
            
            checkpoint = torch.load(latest_checkpoint, map_location='cpu')
            if args.distributed:
                model.module.load_state_dict(checkpoint['model'])
            else:
                model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch'] + 1
            scheduler.last_epoch = checkpoint['epoch']
            if 'loss_scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['loss_scaler'])

    # Training loop
    print(f"Start training for {args.epochs} epochs")
    for epoch in range(start_epoch, args.epochs):
        try:
            # Set epoch for dataset if method exists
            if hasattr(data_loader_train.dataset, 'set_epoch'):
                data_loader_train.dataset.set_epoch(1)

            # Update dataset ratio if supported
            epoch_ratio = epoch / args.epochs
            if hasattr(data_loader_train.dataset, 'set_ratio'):
                if epoch_ratio < 0.75:
                    active_ratio = min(1, epoch / args.epochs * 2.0)
                else:
                    active_ratio = max(0.5, 1 - (epoch_ratio - 0.75) / 0.25)
                data_loader_train.dataset.set_ratio(0.25)

            # Training
            train_stats = train_one_epoch(
                model=model,
                data_loader=data_loader_train,
                optimizer=optimizer,
                device=device,
                epoch=epoch,
                loss_scaler=loss_scaler,
                args=args,
                visualizer=visualizer if args.rank == 0 else None
            )

            # Validation
            if epoch % args.eval_freq == 0 or epoch == args.epochs - 1:
                val_stats = validate(
                    model=model,
                    data_loader=data_loader_train,
                    device=device,
                    epoch=epoch,
                    args=args,
                    visualizer=visualizer if args.rank == 0 else None
                )

            # Update learning rate
            scheduler.step()

            # Save checkpoint
            if args.rank == 0 and args.output_dir:
                checkpoint_path = os.path.join(args.output_dir, f'checkpoint_epoch_{epoch}.pth')
                model_state = model.module.state_dict() if args.distributed else model.state_dict()
                
                save_obj = {
                    'model': model_state,
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'args': args,
                    'loss_scaler': loss_scaler.state_dict(),
                }
                
                torch.save(save_obj, checkpoint_path)
                
                # Remove old checkpoints to save space (keep last 3)
                checkpoint_files = sorted(Path(args.output_dir).glob('checkpoint_epoch_*.pth'))
                for checkpoint_file in checkpoint_files[:-3]:
                    checkpoint_file.unlink()

            if args.rank == 0 and visualizer is not None:
                epoch_summary = {
                    'epoch': epoch,
                    'lr': optimizer.param_groups[0]['lr'],
                    **{f'train_{k}': v for k, v in train_stats.items()},
                    **({f'val_{k}': v for k, v in val_stats.items()} if 'val_stats' in locals() else {})
                }
                
                if args.use_wandb:
                    wandb.log(epoch_summary)
                
                # Also log to TensorBoard
                for k, v in epoch_summary.items():
                    visualizer.log_metric(f'epoch/{k}', v, epoch)

        except Exception as e:
            if args.rank == 0:
                print(f"Error during training: {str(e)}")
                import traceback
                traceback.print_exc()
            raise e

    # Clean up
    if args.rank == 0:
        if visualizer is not None:
            visualizer.close()
        if args.use_wandb:
            wandb.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser('InGaS Training Script', parents=[get_args_parser()])
    args = parser.parse_args()
    
    # Create output and log directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Save training configuration
    if misc.get_rank() == 0:
        config_path = os.path.join(args.output_dir, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(vars(args), f, indent=2)
    
    try:
        main(args)
    except Exception as e:
        if misc.get_rank() == 0:
            print(f"Training failed with error: {str(e)}")
            import traceback
            traceback.print_exc()
        raise e
