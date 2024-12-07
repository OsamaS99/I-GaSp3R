import torch
import einops
import lpips
import torch.nn.functional as F

from spann3r.model import SpatialMemory
from spann3r.model import Spann3R
from spann3r.loss import Regr3D_t, ConfLoss_t, normalize_pointcloud_t
from dust3r.inference import get_pred_means
from dust3r.losses import L21
from dust3r.utils.geometry import inv, geotrf

import pixelsplat.decoder_splatting_cuda as pixelsplat_decoder
import utils.compute_ssim as compute_ssim
import utils.sh_utils as sh_utils
from utils.geometry import adjust_extrinsic_matrix, rotate_sh, build_covariance, get_scale_multiplier


def reg_dense_scales(scales):
    """
    Apply a sigmoid activation to scale offsets so that they are within (0, 1).
    """
    scales = scales.sigmoid()
    return scales

def reg_dense_opacities(opacities):
    """
    Apply a sigmoid activation to opacities so that they are within (0, 1).
    """
    opacities = opacities.sigmoid()
    return opacities

def reg_dense_rotation(rotations, eps=1e-8):
    """
    Normalize rotations to have unit length.
    """
    rotations = rotations / (rotations.norm(dim=-1, keepdim=True) + eps)
    return rotations


class InGaS(Spann3R):
    def __init__(self, spann3r_checkpoint_path, dust3r_checkpoint_path, config=None,
                 use_feat=False, mem_pos_enc=False, memory_dropout=0.15):
        # Initialize the Spann3R model with Gaussian heads
        super(InGaS, self).__init__(
            dus3r_name=dust3r_checkpoint_path,
            use_feat=use_feat,
            mem_pos_enc=mem_pos_enc,
            memory_dropout=memory_dropout,
            is_gaussian=True
        )
        self.config = config

        # Load Spann3R weights with strict=False
        state_dict = torch.load(spann3r_checkpoint_path, map_location='cpu')
        if 'model' in state_dict:
            state_dict = state_dict['model']

        self.load_state_dict(state_dict, strict=False)

        # Freeze DUSt3R backbone, except Gaussian heads
        self.dust3r.requires_grad_(False)
        self.dust3r.downstream_head1.requires_grad_(True)
        self.dust3r.downstream_head2.requires_grad_(True)

        # Initialize decoder
        self.decoder = pixelsplat_decoder.DecoderSplattingCUDA(
            background_color=[0.0, 0.0, 0.0]
        )

        # LPIPS criterion
        if self.config is not None and self.config.average_over_mask:
            self.lpips_criterion = lpips.LPIPS(net='vgg', spatial=True)
        else:
            self.lpips_criterion = lpips.LPIPS(net='vgg')

        if self.config is not None and self.config.spann3r_loss_weight is not None:
            self.spann3r_criterion = ConfLoss_t(Regr3D_t(L21, norm_mode='avg_dis', fix_first=False), alpha=0.4)
            self.dust3r.downstream_head1.requires_grad_(True)
            self.dust3r.downstream_head2.requires_grad_(True)

    def forward(self, frames, return_memory=False):
        """
        Forward pass through the model.
        
        Args:
            frames (list): A list of frames/dicts with 'img', 'original_img', 'pts3d', 'valid_mask', 'camera_pose', and optionally 'camera_intrinsics'.
            return_memory (bool): If True, return the spatial memory.

        Returns:
            preds (list): List of predictions for each frame.
            preds_all (list): Detailed predictions including paired frames.
            sp_mem (SpatialMemory, optional): Returned if return_memory is True.
        """
        if self.training:
            sp_mem = SpatialMemory(self.norm_q, self.norm_k, self.norm_v, mem_dropout=self.mem_dropout, attn_thresh=0)
        else:
            sp_mem = SpatialMemory(self.norm_q, self.norm_k, self.norm_v)

        feat1, feat2, pos1, pos2, shape1, shape2 = None, None, None, None, None, None
        feat_k1, feat_k2 = None, None
        preds = None
        preds_all = []

        for i in range(len(frames) - 1):
            view1 = frames[i]
            view2 = frames[i + 1]

            # Encode frames
            feat1, feat2, pos1, pos2, shape1, shape2 = self.encode_frames(
                view1, view2, feat1, feat2, pos1, pos2, shape1, shape2
            )

            # Memory readout
            if feat_k2 is not None:
                feat_fuse = sp_mem.memory_read(feat_k2, res=True)
            else:
                feat_fuse = feat1

            # Decode features
            dec1, dec2 = self.decode(feat_fuse, pos1, feat2, pos2)

            # Encode feat key
            feat_k1 = self.encode_feat_key(feat1, dec1[-1], 1)
            feat_k2 = self.encode_feat_key(feat2, dec2[-1], 2)

            # Regress point maps
            with torch.cuda.amp.autocast(enabled=False):
                res1 = self.downstream_head(dec1, shape1, 1)
                res2 = self.downstream_head(dec2, shape2, 2)

            # Initialize SH from images
            new_sh1 = torch.zeros_like(res1['sh'])
            new_sh2 = torch.zeros_like(res2['sh'])
            new_sh1[..., 0] = sh_utils.RGB2SH(einops.rearrange(view1['original_img'], 'b c h w -> b h w c'))
            new_sh2[..., 0] = sh_utils.RGB2SH(einops.rearrange(view2['original_img'], 'b c h w -> b h w c'))
            res1['sh'] = res1['sh'] + new_sh1
            res2['sh'] = res2['sh'] + new_sh2

            cur_v = self.encode_cur_value(res1, dec1, pos1, shape1)

            if self.training:
                sp_mem.add_mem(feat_k1, cur_v + feat_k1)
            else:
                sp_mem.add_mem_check(feat_k1, cur_v + feat_k1)

            res2['pts3d_in_other_view'] = res2.pop('pts3d')
            res2['means_in_other_view'] = res2.pop('means')

            if preds is None:
                preds = [res1]
                preds_all = [(res1, res2)]
            else:
                res1['pts3d_in_other_view'] = res1.pop('pts3d')
                res1['means_in_other_view'] = res1.pop('means')
                preds.append(res1)
                preds_all.append((res1, res2))

        preds.append(res2)

        if return_memory:
            return preds, preds_all, sp_mem

        return preds, preds_all

    def render_frames(self, batch, device):
        """
        Consolidates the logic to render frames from a given batch.

        Args:
            batch (list): Batch of frames (each element is a dict with keys like 'img', 'camera_pose', 'camera_intrinsics', etc.)
            device (torch.device): The device on which to perform computations.

        Returns:
            color (torch.Tensor): The rendered frames of shape [B, V, C, H, W].
            context_views (list): The extracted context views.
            target_views (list): The extracted target views.
        """
        # Extract context and target views
        context_views = batch[1:-1]
        target_views = [batch[1]] + [batch[-1]]
        _, preds_all = self.forward(context_views)

        # Compute necessary parameters
        _, combined_pr_pts, gt_factor, _, _, _ = self.get_all_means(context_views, preds_all)
        pred_sh = self.get_all_sh(preds_all)
        pred_opacities = self.get_all_opacities(preds_all)
        pred_rotations = self.get_all_rotations(preds_all)
        pred_scales = self.get_all_scales(preds_all)

        # Get height and width from the first context view
        _, _, h, w = context_views[0]['img'].shape

        base_pose = adjust_extrinsic_matrix(context_views[0]['camera_pose'], gt_factor)
        inv_base_pose = torch.inverse(base_pose)

        target_extrinsics = inv_base_pose[:, None, :, :] @ adjust_extrinsic_matrix(
            torch.stack([t['camera_pose'] for t in target_views], dim=1), gt_factor)
        context_extrinsics = inv_base_pose[:, None, :, :] @ adjust_extrinsic_matrix(
            torch.stack([c['camera_pose'] for c in context_views], dim=1), gt_factor)

        target_intrinsics = torch.stack([t['camera_intrinsics'] for t in target_views], dim=1)
        context_intrinsics = torch.stack([c['camera_intrinsics'] for c in context_views], dim=1)

        harmonics = pred_sh['average']
        context_camera_rotations = context_extrinsics[..., :3, :3]
        harmonics = rotate_sh(harmonics, context_camera_rotations[..., None, None, None, :, :])

        rotations = pred_rotations["average"]
        scales = pred_scales["average"]
        scales = 5.0 + (50.0 - 5.0) * scales

        pixel_size = 1 / torch.tensor((w, h), dtype=torch.float32, device=device)
        multiplier = get_scale_multiplier(context_intrinsics, pixel_size, 0.5)
        means = combined_pr_pts["average"]
        scales = scales * multiplier[..., None, None, None]
        covariances = build_covariance(scales, rotations)
        covariances = context_camera_rotations[..., None, None, :, :] @ covariances @ \
                      context_camera_rotations[..., None, None, :, :].transpose(-1, -2)
        opacities = pred_opacities['average']

        color, _ = self.decoder(target_extrinsics, target_intrinsics, means, harmonics, covariances, opacities, (h, w))

        return color, preds_all, context_views, target_views

    def calculate_loss(self, context_views, target_views, preds_all, color, mask, apply_mask=True, average_over_mask=True, calculate_ssim=False):
        """
        Calculate the loss (MSE, LPIPS, optionally SSIM and Spann3r loss).

        Args:
            context_views (list): The context frames used to produce predictions.
            target_views (list): The target frames.
            preds_all (list): All predictions from the forward pass.
            color (torch.Tensor): The rendered frames [B, V, C, H, W].
            mask (torch.Tensor): The mask indicating valid pixels.
            apply_mask (bool): Whether to mask the loss computation.
            average_over_mask (bool): Whether to average the loss only over masked pixels.
            calculate_ssim (bool): Whether to compute SSIM as well.

        Returns:
            loss, mse_loss, lpips_loss, (ssim_val if calculate_ssim=True)
        """
        target_color = torch.stack([view['original_img'] for view in target_views], dim=1)
        predicted_color = color

        if apply_mask:
            target_color = target_color * mask[..., None, :, :]
            predicted_color = predicted_color * mask[..., None, :, :]

        flattened_color = einops.rearrange(predicted_color, 'b v c h w -> (b v) c h w')
        flattened_target_color = einops.rearrange(target_color, 'b v c h w -> (b v) c h w')
        flattened_mask = einops.rearrange(mask, 'b v h w -> (b v) h w')

        rgb_l2_loss = (flattened_color - flattened_target_color) ** 2
        if average_over_mask:
            mse_loss = (rgb_l2_loss * flattened_mask[:, None, ...]).sum() / flattened_mask.sum()
        else:
            mse_loss = rgb_l2_loss.mean()

        lpips_loss = self.lpips_criterion(flattened_target_color, flattened_color, normalize=True)
        if average_over_mask:
            lpips_loss = (lpips_loss * flattened_mask[:, None, ...]).sum() / flattened_mask.sum()
        else:
            lpips_loss = lpips_loss.mean()

        loss = self.config.mse_loss_weight * mse_loss + self.config.lpips_loss_weight * lpips_loss

        if self.config.spann3r_loss_weight is not None:
            spann3r_loss = self.spann3r_criterion.compute_frame_loss(context_views, preds_all)
            loss += self.config.spann3r_loss_weight * spann3r_loss

        if calculate_ssim:
            if average_over_mask:
                ssim_val = compute_ssim.compute_ssim(flattened_target_color, flattened_color, full=True)
                ssim_val = (ssim_val * flattened_mask[:, None, ...]).sum() / flattened_mask.sum()
            else:
                ssim_val = compute_ssim.compute_ssim(flattened_target_color, flattened_color, full=False)
                ssim_val = ssim_val.mean()
            return loss, mse_loss, lpips_loss, ssim_val

        return loss, mse_loss, lpips_loss

    @staticmethod
    def get_all_means(gts, preds, norm_mode='avg_dis', fix_first=False, dist_clip=None, gt_scale=False):
        in_camera1 = inv(gts[0]['camera_pose'])

        gt_pts = []
        valids = []
        pr_pts_l = []
        pr_pts_r = []
        num_frames = len(gts)

        for i, gt in enumerate(gts):
            gt_pts.append(geotrf(in_camera1, gt['pts3d']))
            valid = gt['valid_mask'].clone()

            if dist_clip is not None:
                dis = gt['pts3d'].norm(dim=-1)
                valid = valid & (dis <= dist_clip)
            valids.append(valid)

            if i != num_frames - 1:
                pred_left = get_pred_means(preds[i][0], use_pose=(i != 0))
                pr_pts_l.append(pred_left)
            if i != 0:
                pred_right = get_pred_means(preds[i-1][1], use_pose=(i != 0))
                pr_pts_r.append(pred_right)

        pr_pts = (pr_pts_l, pr_pts_r)

        if norm_mode:
            pr_pts, pr_factor = normalize_pointcloud_t(pr_pts, norm_mode, valids, fix_first=fix_first, gt=False)
        else:
            pr_factor = None

        if norm_mode and not gt_scale:
            gt_pts, gt_factor = normalize_pointcloud_t(gt_pts, norm_mode, valids, fix_first=fix_first, gt=True)
        else:
            gt_factor = None

        pr_pts_l, pr_pts_r = pr_pts
        pr_pts_avg_normalized = [pr_pts_l[0]] + [
            (pr_pts_l[i + 1] + pr_pts_r[i]) / 2 for i in range(num_frames - 2)
        ] + [pr_pts_r[-1]]
        pr_pts_avg_normalized = torch.stack(pr_pts_avg_normalized, dim=1)

        combined_pr_pts = {
            'left': pr_pts_l,
            'right': pr_pts_r,
            'average': pr_pts_avg_normalized,
        }

        return gt_pts, combined_pr_pts, gt_factor, pr_factor, valids, {}

    @staticmethod
    def get_all_sh(preds):
        num_frames = len(preds)
        pr_pts_l = [preds[i][0]['sh'] for i in range(num_frames)]
        pr_pts_r = [preds[i][1]['sh'] for i in range(num_frames)]
        pr_pts_avg = [pr_pts_l[0]] + [
            (pr_pts_l[i + 1] + pr_pts_r[i]) / 2 for i in range(num_frames - 1)
        ] + [pr_pts_r[-1]]

        pr_pts_avg = torch.stack(pr_pts_avg, dim=1)
        combined_pr_pts = {
            'left': pr_pts_l,
            'right': pr_pts_r,
            'average': pr_pts_avg
        }

        return combined_pr_pts

    @staticmethod
    def get_all_scales(preds):
        num_frames = len(preds)
        pr_pts_l = [preds[i][0]['scales'] for i in range(num_frames)]
        pr_pts_r = [preds[i][1]['scales'] for i in range(num_frames)]
        pr_pts_avg = [pr_pts_l[0]] + [
            (pr_pts_l[i + 1] + pr_pts_r[i]) / 2 for i in range(num_frames - 1)
        ] + [pr_pts_r[-1]]

        pr_pts_avg = torch.stack(pr_pts_avg, dim=1)
        pr_pts_avg = reg_dense_scales(pr_pts_avg)

        combined_pr_pts = {
            'left': pr_pts_l,
            'right': pr_pts_r,
            'average': pr_pts_avg
        }

        return combined_pr_pts

    @staticmethod
    def get_all_rotations(preds):
        num_frames = len(preds)
        pr_pts_l = [preds[i][0]['rotations'] for i in range(num_frames)]
        pr_pts_r = [preds[i][1]['rotations'] for i in range(num_frames)]
        pr_pts_avg = [pr_pts_l[0]] + [
            (pr_pts_l[i + 1] + pr_pts_r[i]) / 2 for i in range(num_frames - 1)
        ] + [pr_pts_r[-1]]
        pr_pts_avg = torch.stack(pr_pts_avg, dim=1)

        pr_pts_avg = reg_dense_rotation(pr_pts_avg)

        combined_pr_pts = {
            'left': pr_pts_l,
            'right': pr_pts_r,
            'average': pr_pts_avg
        }

        return combined_pr_pts

    @staticmethod
    def get_all_opacities(preds):
        num_frames = len(preds)
        pr_pts_l = [preds[i][0]['opacities'] for i in range(num_frames)]
        pr_pts_r = [preds[i][1]['opacities'] for i in range(num_frames)]
        pr_pts_avg = [pr_pts_l[0]] + [
            (pr_pts_l[i + 1] + pr_pts_r[i]) / 2 for i in range(num_frames - 1)
        ] + [pr_pts_r[-1]]

        pr_pts_avg = torch.stack(pr_pts_avg, dim=1)
        pr_pts_avg = reg_dense_opacities(pr_pts_avg)

        combined_pr_pts = {
            'left': pr_pts_l,
            'right': pr_pts_r,
            'average': pr_pts_avg
        }

        return combined_pr_pts
