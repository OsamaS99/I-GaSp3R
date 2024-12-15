import os
import json
import torch
import argparse
import datetime
import numpy as np
import torch.backends.cudnn as cudnn
import random
import imageio
import wandb
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
import io
from collections import defaultdict
from pathlib import Path
from torch.optim.lr_scheduler import MultiStepLR
from torch.cuda.amp import GradScaler

from ingas.model import InGaS  # Import the InGaS model from model.py
from spann3r.datasets import build_dataset
import croco.utils.misc as misc
import utils.loss_mask as loss_mask


class Visualizer:
    def __init__(self, log_dir, use_wandb=True):
        self.writer = SummaryWriter(log_dir)
        self.use_wandb = use_wandb
        self.log_buffer = {}

    def log_images(self, tag, images, step, nrow=8):
        if isinstance(images, list):
            images = torch.stack([
                torch.from_numpy(img) if isinstance(img, np.ndarray) else img 
                for img in images
            ])
        grid = make_grid(images, nrow=nrow, normalize=True)
        self.writer.add_image(tag, grid, step)
        if self.use_wandb:
            self.log_buffer[tag] = wandb.Image(grid)

    def log_gif_images(self, tag, images, step, fps=2):
        frames = []
        for img in images:
            if isinstance(img, torch.Tensor):
                img = img.cpu().permute(1, 2, 0).numpy()
                img = ((img + 1.0) / 2.0 * 255).astype(np.uint8)
            elif isinstance(img, np.ndarray):
                img = img
            else:
                raise TypeError("Images should be either torch.Tensor or numpy.ndarray")
            frames.append(img)
        
        
        # Create GIF in memory

        # Create GIF in memory
        gif_bytes = io.BytesIO()
        imageio.mimsave(gif_bytes, frames, format='gif', fps=fps)
        gif_bytes.seek(0)
                
                
        # Log to W&B as GIF

        # Log to W&B as GIF
        if self.use_wandb:
            self.log_buffer[tag] = wandb.Video(gif_bytes, format="gif")

    def log_metric(self, tag, value, step):
        self.writer.add_scalar(tag, value, step)
        if self.use_wandb:
            self.log_buffer[tag] = value

    def flush(self, step):
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

    # Datasets (adjust these based on your actual dataset setup)
    parser.add_argument('--train_dataset', type=str,
                       default="Scannetpp(split='train', ROOT='/mnt/hdd/scannetpp/data', resolution=224, num_seq=1, num_frames=5, kf_every=10, seed=777, full_video=False, transform=ImgNorm)")
    #parser.add_argument('--val_dataset', type=str,
     #                  default="Scannetpp(split='val', ROOT='/mnt/hdd/scannetpp/data', resolution=224, num_seq=1, kf_every=10, seed=777, full_video=True)")
    parser.add_argument('--val_dataset', type=str,
                       default="Scannetpp(split='val', ROOT='/mnt/hdd/scannetpp/data', resolution=224, num_seq=1, num_frames=5, kf_every=10, seed=777, full_video=False, transform=ImgNorm)")
    # Experiment parameters
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--output_dir', type=str, default='./output')
    parser.add_argument('--log_dir', type=str, default='./logs')

    # Training Hyperparameters
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--epochs', default=1, type=int)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--gradient_clip_val', type=float, default=0.5)
    parser.add_argument('--warmup_epochs', type=int, default=10)
    parser.add_argument('--eval_freq', type=int, default=1)
    parser.add_argument('--save_freq', type=int, default=1)
    parser.add_argument('--print_freq', type=int, default=10)
    parser.add_argument('--accum_iter', default=16, type=int)
    parser.add_argument('--num_workers', default=8, type=int)

    # Loss Parameters
    parser.add_argument('--apply_mask', action='store_false')
    parser.add_argument('--average_over_mask', action='store_false')
    parser.add_argument('--mse_loss_weight', type=float, default=1.0)
    parser.add_argument('--lpips_loss_weight', type=float, default=0.05)
    parser.add_argument('--spann3r_loss_weight', type=float, default=None)

    # Distributed training
    parser.add_argument('--dist_url', default='env://')
    parser.add_argument('--world_size', default=1, type=int)
    parser.add_argument('--local_rank', default=-1, type=int)
    
    # Logging
    parser.add_argument('--wandb_project', type=str, default='InGaS')
    parser.add_argument('--wandb_run_name', type=str, default='InGaS_Demo')
    parser.add_argument('--no_wandb', action='store_false', dest='use_wandb', default=True, help='Disable W&B logging')

    # Add new argument for using train dataset during evaluation
    parser.add_argument('--eval_on_train', action='store_true', default=False,
                       help='Use training dataset for evaluation instead of validation dataset')
    
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



def validate(model, data_loader, device, epoch, args, visualizer):
    model.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = f'Validation Epoch: [{epoch}]'
    
    # Add metric accumulation
    total_metrics = defaultdict(float)
    num_batches = 0
    
    torch.cuda.empty_cache()
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
            try:
                # Process smaller batches during validation
                if args.batch_size > 1:
                    for key in batch:
                        batch[key] = batch[key][:1]  # Process only first item in batch
                
                batch = process_batch(batch, device)
                torch.cuda.empty_cache()
                
                color, preds_all, context_views, target_views = model.render_frames(batch, device)
                
                mask = loss_mask.calculate_loss_mask(target_views, context_views)
                loss, mse_loss, lpips_loss, ssim_val = model.calculate_loss(
                    context_views, target_views, preds_all, color, mask,
                    apply_mask=args.apply_mask,
                    average_over_mask=args.average_over_mask,
                    calculate_ssim=True
                )

                # Accumulate metrics
                total_metrics['loss'] += loss.item()
                total_metrics['mse'] += mse_loss.item()
                total_metrics['lpips'] += lpips_loss.item()
                total_metrics['ssim'] += ssim_val.item()
                num_batches += 1

                if args.rank == 0:
                    # Use epoch * total_batches + batch_idx for consistent step counting
                    global_step = (epoch - 1) * len(data_loader) + batch_idx
                    visualizer.log_metric('val/loss', loss.item(), global_step)
                    visualizer.log_metric('val/mse', mse_loss.item(), global_step)
                    visualizer.log_metric('val/lpips', lpips_loss.item(), global_step)
                    visualizer.log_metric('val/ssim', ssim_val.item(), global_step)
                    visualizer.flush(global_step)

                    # Log target and rendered frames as GIFs
                    target_images = [view['img'][0].cpu() for view in target_views]
                    visualizer.log_images('val/target_frames', target_images, global_step)

                    context_images = [view['img'][0].cpu() for view in context_views]
                    visualizer.log_images('val/context_frames', context_images, global_step)

                    rendered_images = [color[0, i].cpu() for i in range(color.shape[1])]
                    visualizer.log_images('val/rendered_frames', rendered_images, global_step)

                metric_logger.update(loss=loss.item(), mse=mse_loss.item(), lpips=lpips_loss.item(), ssim=ssim_val.item())

            except torch.cuda.OutOfMemoryError:
                # Handle OOM gracefully
                torch.cuda.empty_cache()
                print(f"WARNING: Out of memory in validation batch {batch_idx}. Skipping batch.")
                continue

    metric_logger.synchronize_between_processes()
    
    # Calculate and log mean metrics using the last global_step
    if args.rank == 0 and num_batches > 0:
        mean_metrics = {k: v / num_batches for k, v in total_metrics.items()}
        if visualizer is not None:
            final_step = (epoch - 1) * len(data_loader) + len(data_loader) - 1
            for metric_name, value in mean_metrics.items():
                visualizer.log_metric(f'val/epoch_mean_{metric_name}', value, final_step)
            visualizer.flush(final_step)
    
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

    # Setup data loader for evaluation only
    if args.eval_on_train:
        eval_loader = build_dataset(args.train_dataset, test=False, num_workers=args.num_workers, batch_size=args.batch_size)
    else:
        eval_loader = build_dataset(args.val_dataset, test=True, num_workers=args.num_workers, batch_size=args.batch_size)
    
    # Initialize train_ratio for validation
    eval_loader.dataset.train_ratio = 0.7   # Fixed ratio for validation
    # Resume from checkpoint if exists
    start_epoch = 0
    latest_checkpoint = None
    checkpoint_files = sorted(Path(args.output_dir).glob('checkpoint_epoch_*.pth'))
    if checkpoint_files:
        latest_checkpoint = str(checkpoint_files[-1])
        # Extract epoch number from filename
        epoch_num = latest_checkpoint.split('epoch_')[-1].split('.')[0]
        if args.rank == 0:
            print(f"Resuming from checkpoint: {latest_checkpoint}")
            print(f"Checkpoint epoch: {epoch_num}")

        checkpoint = torch.load(latest_checkpoint, map_location='cpu')
        if args.distributed:
            model.module.load_state_dict(checkpoint['model'])
        else:
            model.load_state_dict(checkpoint['model'])


    # Training loop

    try:
        val_stats = {}
        val_stats = validate(
            model=model,
            data_loader=eval_loader,  # Use the selected loader
            device=device,
            epoch=1,
            args=args,
            visualizer=visualizer if args.rank == 0 else None
        )


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

    # Create output and log directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    # Save training configuration

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
