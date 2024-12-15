import os
import json
import torch
import argparse
import numpy as np
import torch.backends.cudnn as cudnn
import random
import datetime

import imageio

import wandb
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
import io
from pathlib import Path
from torch.optim.lr_scheduler import MultiStepLR
from torch.cuda.amp import GradScaler

from ingas.model import InGaS
from spann3r.datasets import build_dataset
import croco.utils.misc as misc
import utils.loss_mask as loss_mask

import torch.distributed as dist

class Visualizer:
    def __init__(self, log_dir, use_wandb=True):
        self.writer = SummaryWriter(log_dir)
        self.use_wandb = use_wandb
        self.log_buffer = {}

    def log_images(self, tag, images, step, nrow=8):
        """Log images to TensorBoard and W&B."""
        if isinstance(images, list):
            images = torch.stack([torch.from_numpy(img) if isinstance(img, np.ndarray) 
                                  else img for img in images])
        grid = make_grid(images, nrow=nrow, normalize=True)
        
        # Log to TensorBoard
        self.writer.add_image(tag, grid, step)
        
        # Log to W&B
        if self.use_wandb:
            self.log_buffer[tag] = wandb.Image(grid)

    def log_gif_images(self, tag, images, step, fps=2):
        """Log GIF images."""
        frames = []
        for img in images:
            if isinstance(img, torch.Tensor):
                img = img.cpu().permute(1, 2, 0).numpy()
                img = ((img + 1.0) / 2.0 * 255).astype(np.uint8)  # Assuming images are normalized [-1,1]
            elif isinstance(img, np.ndarray):
                img = img
            else:
                raise TypeError("Images should be either torch.Tensor or numpy.ndarray")
            frames.append(img)
        
        gif_bytes = io.BytesIO()
        imageio.mimsave(gif_bytes, frames, format='gif', fps=fps)
        gif_bytes.seek(0)
                
        # Log to W&B
        if self.use_wandb:
            self.log_buffer[tag] = wandb.Video(gif_bytes, format="gif")

    def log_metric(self, tag, value, step):
        """Log scalar metrics."""
        self.writer.add_scalar(tag, value, step)
        if self.use_wandb:
            self.log_buffer[tag] = value

    def log_point_cloud(self, tag, points, colors, step):
        """Log point cloud."""
        if self.use_wandb:
            point_cloud_data = np.hstack((points, colors))
            self.log_buffer[tag] = wandb.Object3D(point_cloud_data)

    def flush(self, step):
        """Flush logs."""
        if self.use_wandb:
            wandb.log(self.log_buffer, step=step)
        self.log_buffer.clear()

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
                       default="Scannetpp(split='train', ROOT='/mnt/hdd/scannetpp/data', resolution=224, num_frames=6, transform=ImgNorm)")
    parser.add_argument('--val_dataset', type=str,
                       default="Scannetpp(split='val', ROOT='/mnt/hdd/scannetpp/data', resolution=224, num_seq=1, kf_every=10, seed=777, full_video=True)")

    # Experiment parameters
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--output_dir', type=str, default='./output')
    parser.add_argument('--log_dir', type=str, default='./logs')

    # Training Hyperparameters
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--epochs', default=500, type=int)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.001)
    parser.add_argument('--gradient_clip_val', type=float, default=0.5)
    parser.add_argument('--warmup_epochs', type=int, default=10)
    parser.add_argument('--eval_freq', type=int, default=5)
    parser.add_argument('--save_freq', type=int, default=5)
    parser.add_argument('--print_freq', type=int, default=10)
    parser.add_argument('--accum_iter', default=4, type=int, help='For gradient accumulation')
    parser.add_argument('--num_workers', default=1, type=int)

    # Loss Parameters
    parser.add_argument('--apply_mask', action='store_false')
    parser.add_argument('--average_over_mask', action='store_false')
    parser.add_argument('--use_frequency_weighting', action='store_false')

    parser.add_argument('--mse_loss_weight', type=float, default=1.0)
    parser.add_argument('--lpips_loss_weight', type=float, default=0.1)
    parser.add_argument('--opacity_reg_weight', type=float, default=0.005)
    parser.add_argument('--spann3r_loss_weight', type=float, default=0.05)

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
    dist.init_process_group(
        backend=args.dist_backend, init_method=args.dist_url,
        world_size=args.world_size, rank=args.rank
    )
    dist.barrier()
    misc.setup_for_distributed(args.rank == 0)


def process_batch(batch, device):
    """Move batch data to device."""
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
        if len(batch) < (args.num_context_frames + args.num_target_frames):
            # Skip if batch doesn't have enough frames
            continue

        batch = process_batch(batch, device)
        context_views = batch[:len(batch)//2]
        target_views = batch[len(batch)//2:]

        with torch.cuda.amp.autocast(enabled=False):
            # Forward pass and render predictions
            predicted_color, splat_freq, predicted_opacity, preds_all = model.render_predictions(context_views, target_views, device)
            # Compute mask
            mask = loss_mask.calculate_loss_mask(target_views, context_views)
            # Compute losses
            loss, mse_loss, lpips_loss, opacity_reg, spann3r_loss, _ = model.calculate_loss(
                context_views, target_views, preds_all, predicted_color, splat_freq, predicted_opacity, mask,
                apply_mask=args.apply_mask,
                average_over_mask=args.average_over_mask,
                use_frequency_weighting=args.use_frequency_weighting,
                calculate_ssim=False
            )

        loss_value = loss.item()

        # Backprop with gradient scaling
        loss = loss / args.accum_iter
        loss_scaler.scale(loss).backward()

        # Gradient accumulation
        if (epoch * len(data_loader) + batch_idx + 1) % args.accum_iter == 0:
            loss_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip_val)
            loss_scaler.step(optimizer)
            loss_scaler.update()
            optimizer.zero_grad(set_to_none=True)

        # Logging
        if args.rank == 0:
            global_step = epoch * len(data_loader) + batch_idx
            visualizer.log_metric('train/loss', loss_value, global_step)
            visualizer.log_metric('train/mse', mse_loss.item(), global_step)
            visualizer.log_metric('train/lpips', lpips_loss.item(), global_step)
            if opacity_reg:
                visualizer.log_metric('train/opacity_reg', opacity_reg.item(), global_step)
            if spann3r_loss:
                visualizer.log_metric('train/spann3r_loss', spann3r_loss.item(), global_step)

            if global_step % 100 == 0:
                # Log some images
                context_images = [view['img'][0].cpu() for view in context_views]
                visualizer.log_images('train/context_frames', context_images, global_step, nrow=len(context_images))

                target_images = [view['img'][0].cpu() for view in target_views]
                visualizer.log_images('train/target_frames', target_images, global_step, nrow=len(target_images))

                rendered_images = [predicted_color[0, i].cpu() for i in range(predicted_color.shape[1])]
                visualizer.log_images('train/rendered_frames', rendered_images, global_step, nrow=len(rendered_images))

            visualizer.flush(global_step)

        metric_logger.update(loss=loss_value, mse=mse_loss.item(), lpips=lpips_loss.item())
        if opacity_reg:
            metric_logger.update(opacity_reg=opacity_reg.item())

        if spann3r_loss:
            metric_logger.update(spann3r_loss=spann3r_loss.item())

    metric_logger.synchronize_between_processes()
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def validate(model, data_loader, device, epoch, args, visualizer):
    model.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = f'Validation Epoch: [{epoch}]'
    global_step = epoch * len(data_loader)

    with torch.no_grad():
        for batch_idx, batch in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
            if len(batch) < (args.num_context_frames + args.num_target_frames):
                continue

            batch = process_batch(batch, device)
            context_views = batch
            target_views = batch

            predicted_color, preds_all = model.render_predictions(context_views, target_views, device)
            mask = loss_mask.calculate_loss_mask(target_views, context_views)
            loss, mse_loss, lpips_loss, ssim_val = model.calculate_loss(
                context_views, target_views, preds_all, predicted_color, mask,
                apply_mask=args.apply_mask,
                average_over_mask=args.average_over_mask,
                calculate_ssim=True
            )

            if args.rank == 0:
                step = global_step + batch_idx
                visualizer.log_metric('val/loss', loss.item(), step)
                visualizer.log_metric('val/mse', mse_loss.item(), step)
                visualizer.log_metric('val/lpips', lpips_loss.item(), step)
                visualizer.log_metric('val/ssim', ssim_val.item(), step)

                if batch_idx == 0:
                    # Log as GIF
                    target_images = [view['img'][0].cpu() for view in target_views]
                    visualizer.log_gif_images('val/target_frames', target_images, step, fps=2)

                    rendered_images = [predicted_color[0, i].cpu() for i in range(predicted_color.shape[1])]
                    visualizer.log_gif_images('val/rendered_frames', rendered_images, step, fps=2)

                visualizer.flush(step)

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

    # Setup device and seeds
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
                name=args.wandb_run_name + f"_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
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
    dataset_train = build_dataset(args.train_dataset, test=False, num_workers=args.num_workers, batch_size=args.batch_size)
    dataset_val = build_dataset(args.val_dataset, test=True, num_workers=args.num_workers, batch_size=args.batch_size)
    data_loader_train = dataset_train
    data_loader_val = dataset_val

    # Initialize lists to hold parameters for each group
    params_with_standard_lr = []
    params_with_reduced_lr = []

    # Iterate through all named parameters in the model
    for name, param in model.named_parameters():
        # Exclude parameters containing "lpips" and ensure they require gradients
        if "lpips" not in name and param.requires_grad:
            if "gaussian_dpt.dpt" in name:
                params_with_standard_lr.append(param)
            else:
                params_with_reduced_lr.append(param)

    # Optimizer and Scheduler
    optimizer = torch.optim.Adam(
        [
            {'params': params_with_standard_lr, 'lr': args.lr},
            {'params': params_with_reduced_lr, 'lr': 0.1 * args.lr}
        ],
    )

    scheduler = MultiStepLR(
        optimizer,
        milestones=range(args.epochs // 10, args.epochs, args.epochs // 10),  
        gamma=0.9
    )

    loss_scaler = GradScaler()

    start_epoch = 0
    # Resume if checkpoint found
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

    print(f"Start training for {args.epochs} epochs")
    for epoch in range(start_epoch, args.epochs):
        if hasattr(data_loader_train.dataset, 'set_epoch'):
            data_loader_train.dataset.set_epoch(epoch)
        max_ratio = 0.8
        min_ratio = 0.4

        if args.epochs > 1:
            current_epoch = epoch - start_epoch
            progress = current_epoch / (args.epochs - 1)
            ratio = max_ratio - progress * (max_ratio - min_ratio)
        else:
            ratio = min_ratio  # or max_ratio, depending on your preference

        data_loader_train.dataset.set_ratio(ratio)

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

        if (epoch % args.eval_freq == 0 or epoch == args.epochs - 1) and data_loader_val is not None:
            val_stats = validate(
                model=model,
                data_loader=data_loader_val,
                device=device,
                epoch=epoch,
                args=args,
                visualizer=visualizer if args.rank == 0 else None
            )
        else:
            val_stats = {}

        scheduler.step()

        if args.rank == 0 and args.output_dir:
            checkpoint_path = os.path.join(args.output_dir, f'checkpoint_epoch_{epoch}.pth')
            model_state = model.module.state_dict() if args.distributed else model.state_dict()
            
            save_obj = {
                'model': model_state,
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'args': vars(args),
                'loss_scaler': loss_scaler.state_dict(),
            }
            torch.save(save_obj, checkpoint_path)

            # Keep last 3 checkpoints
            checkpoint_files = sorted(Path(args.output_dir).glob('checkpoint_epoch_*.pth'))
            for ckpt in checkpoint_files[:-3]:
                ckpt.unlink()

            # Log epoch summary
            epoch_summary = {
                'epoch': epoch,
                'lr': optimizer.param_groups[0]['lr'],
            }

            if args.use_wandb:
                wandb.log(epoch_summary)

            # TensorBoard logging for epoch-level metrics
            for k, v in epoch_summary.items():
                visualizer.log_metric(f'epoch/{k}', v, epoch)

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
