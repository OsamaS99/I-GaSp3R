import torch
import einops
import lpips

from spann3r.model import SpatialMemory, Spann3R
from spann3r.loss import Regr3D_t, ConfLoss_t, normalize_pointcloud_t
from dust3r.inference import get_pred_pts3d
from dust3r.losses import L21
from dust3r.utils.geometry import inv, geotrf

import pixelsplat.decoder_splatting_cuda as pixelsplat_decoder
import utils.compute_ssim as compute_ssim
import utils.sh_utils as sh_utils
from utils.geometry import get_scale_multiplier, build_covariance, adjust_extrinsic_matrix, rotate_sh


def reg_dense_scales(scales):
    """Regularize scales into a [0,1] range and then map to desired range later."""
    return scales.sigmoid()

def reg_dense_opacities(opacities):
    """Regularize opacities to a [0,1] range."""
    return opacities.sigmoid()

def reg_dense_rotation(rotations, eps=1e-8):
    """Normalize rotations to unit length."""
    return rotations / (rotations.norm(dim=-1, keepdim=True) + eps)

class CustomMeanFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor1, tensor2):
        # Save tensors for backward pass
        ctx.save_for_backward(tensor1, tensor2)
        # Compute forward pass: (tensor1 + tensor2) / 2
        return (tensor1 + tensor2) / 2

    @staticmethod
    def backward(ctx, grad_output):
        # Retrieve saved tensors
        tensor1, tensor2 = ctx.saved_tensors
        # Return gradients without dividing by 2
        return grad_output, grad_output



class InGaS(Spann3R):
    def __init__(self, spann3r_checkpoint_path, dust3r_checkpoint_path, config=None,
                 use_feat=False, mem_pos_enc=False, memory_dropout=0.15):
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

        # Freeze dust3r except heads
        self.dust3r.requires_grad_(False)
        self.dust3r.downstream_head1.requires_grad_(True)
        self.dust3r.downstream_head2.requires_grad_(True)

        # Initialize decoder
        self.decoder = pixelsplat_decoder.DecoderSplattingCUDA(
            background_color=[0.0, 0.0, 0.0]
        )

        # LPIPS loss
        if self.config and self.config.average_over_mask:
            self.lpips_criterion = lpips.LPIPS(net='vgg', spatial=True)
        else:
            self.lpips_criterion = lpips.LPIPS(net='vgg')
        
        self.lpips_criterion.requires_grad_(False)

        if self.config and self.config.spann3r_loss_weight is not None:
            self.spann3r_criterion = ConfLoss_t(Regr3D_t(L21, norm_mode='avg_dis', fix_first=False), alpha=0.1)

    def forward(self, frames, return_memory=False):
        """
        Forward pass of the InGaS model.
        frames: list of views (dicts)
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

            # Memory read
            if feat_k2 is not None:
                feat_fuse = sp_mem.memory_read(feat_k2, res=True)
            else:
                feat_fuse = feat1

            # Decode features
            dec1, dec2 = self.decode(feat_fuse, pos1, feat2, pos2)

            # Encode feat key
            feat_k1 = self.encode_feat_key(feat1, dec1[-1], 1)
            feat_k2 = self.encode_feat_key(feat2, dec2[-1], 2)

            # Regress pointmaps
            with torch.cuda.amp.autocast(enabled=False):
                res1 = self.downstream_head(dec1, shape1, 1)
                res2 = self.downstream_head(dec2, shape2, 2)

            # Residual SH initialization
            new_sh1 = torch.zeros_like(res1['sh'])
            new_sh2 = torch.zeros_like(res2['sh'])
            new_sh1[..., 0] = sh_utils.RGB2SH(einops.rearrange(view1['original_img'], 'b c h w -> b h w c'))
            new_sh2[..., 0] = sh_utils.RGB2SH(einops.rearrange(view2['original_img'], 'b c h w -> b h w c'))
            res1['sh'] = res1['sh'] + new_sh1
            res2['sh'] = res2['sh'] + new_sh2

            # Memory update
            cur_v = self.encode_cur_value(res1, dec1, pos1, shape1)
            if self.training:
                sp_mem.add_mem(feat_k1, cur_v + feat_k1)
            else:
                sp_mem.add_mem_check(feat_k1, cur_v + feat_k1)

            # Prepare preds
            res2['pts3d_in_other_view'] = res2.pop('pts3d')

            if res1 is None:
                raise ValueError('res1 is None')
            if preds is None:
                preds = [res1]
                preds_all = [(res1, res2)]
            else:
                res1['pts3d_in_other_view'] = res1.pop('pts3d')
                preds.append(res1)
                preds_all.append((res1, res2))

        preds.append(res2)

        if return_memory:
            return preds, preds_all, sp_mem
        return preds, preds_all

    def render_predictions(self, context_views, target_views, device):
        """
        Given context and target views, run model forward, compute gaussian splats,
        and render the predicted images for target views.
        Returns:
            predicted_color: tensor [B, T, C, H, W]
            preds_all: the detailed predictions structure
        """
        # Forward pass through model using context + target (for the sequential nature)
        # We assume all frames together: context then target
        with torch.no_grad() if not self.training else torch.enable_grad():
            preds, preds_all = self(context_views)

        # Compute all necessary attributes from preds_all
        gt_pts, combined_pr_pts, gt_factor, pr_factor, valids, _ = self.get_all_means(context_views, preds_all)
        pred_sh = self.get_all_sh(preds_all)
        pred_opacities = self.get_all_opacities(preds_all)
        pred_rotations = self.get_all_rotations(preds_all)
        pred_scales = self.get_all_scales(preds_all)

        # We now need to compute extrinsics/intrinsics for target poses and render
        _, _, h, w = target_views[0]['img'].shape

        base_pose = adjust_extrinsic_matrix(context_views[0]['camera_pose'], gt_factor)
        inv_base_pose = torch.inverse(base_pose)

        # Create extrinsics for target views
        target_extrinsics = inv_base_pose[:, None, :, :] @ adjust_extrinsic_matrix(
            torch.stack([tv['camera_pose'] for tv in target_views], dim=1), gt_factor
        )

        context_extrinsics = inv_base_pose[:, None, :, :] @ adjust_extrinsic_matrix(
            torch.stack([cv['camera_pose'] for cv in context_views], dim=1), gt_factor
        )

        target_intrinsics = torch.stack([tv['camera_intrinsics'] for tv in target_views], dim=1)
        context_intrinsics = torch.stack([cv['camera_intrinsics'] for cv in context_views], dim=1)

        harmonics = pred_sh['average']
        context_camera_rotations = context_extrinsics[..., :3, :3]
        harmonics = rotate_sh(harmonics, context_camera_rotations[..., None, None, None, :, :])

        rotations = pred_rotations["average"]
        scales = pred_scales["average"]
        scales = 50.0 * scales

        pixel_size = 1 / torch.tensor((w, h), dtype=torch.float32, device=device)
        multiplier = get_scale_multiplier(context_intrinsics, pixel_size, 0.2) 
        means = combined_pr_pts["average"]
        scales = scales * multiplier[..., None, None, None] / pr_factor.view(-1)

        covariances = build_covariance(scales, rotations)
        covariances = context_camera_rotations[..., None, None, :, :] @ covariances @ context_camera_rotations[..., None, None, :, :].transpose(-1, -2)

        opacities = pred_opacities['average']
        # Render
        color, freq = self.decoder(
            target_extrinsics, target_intrinsics,
            means, harmonics, 
            covariances, opacities,
            (h, w)
        )

        return color, freq, opacities, preds_all

    def calculate_loss(self, context_views, target_views, preds_all, predicted_color, splat_freq, predicted_opacity, mask,
                    apply_mask=True, average_over_mask=True, calculate_ssim=False, epsilon=1e-8, use_frequency_weighting=True):
        """
        Calculate the losses with optional frequency-based weighting.
        Here, we first aggregate splat_freq over its 'c' dimension to match the color images (which have c=3).
        """

        # Extract target colors: (b, v, c, h, w)
        target_color = torch.stack([view['original_img'] for view in target_views], dim=1)

        # Apply mask if requested
        if apply_mask:
            target_color = target_color * mask[..., None, :, :]         # (b, v, c, h, w)
            predicted_color = predicted_color * mask[..., None, :, :]   # (b, v, c, h, w)
            
        # Flatten (b,v) -> (b*v)
        flattened_color = einops.rearrange(predicted_color, 'b v c h w -> (b v) c h w')  # (b*v,c,h,w)
        flattened_target_color = einops.rearrange(target_color, 'b v c h w -> (b v) c h w')  # (b*v,c,h,w)
        flattened_mask = einops.rearrange(mask, 'b v h w -> (b v) h w')  # (b*v,h,w)

        if use_frequency_weighting:
            # splat_freq: (b, v, c_freq, h, w)
            # Aggregate over c_freq dimension
            splat_freq_mean = splat_freq.to(torch.float).mean(dim=2)  # now (b, v, h, w)

            # Flatten along b and v
            flattened_splat_freq = einops.rearrange(splat_freq_mean, 'b v h w -> (b v) h w')  # (b*v,h,w)

            # Compute weights: 1/(freq+epsilon), and zero out where freq=0
            weight_map = 1.0 / (flattened_splat_freq + epsilon)
            weight_map[flattened_splat_freq == 0] = 0.0  # (b*v,h,w)

        else:
            # No frequency weighting: just use mask if available
            # Create a uniform weight map of ones
            weight_map = torch.ones((flattened_color.shape[0], flattened_color.shape[2], flattened_color.shape[3]),
                                    dtype=flattened_color.dtype,
                                    device=flattened_color.device)  # (b*v,h,w)
            if apply_mask:
                weight_map = weight_map * flattened_mask  # (b*v,h,w)
            # This will broadcast similarly during multiplication

        # ============ MSE Loss ============
        rgb_l2_loss = (flattened_color - flattened_target_color) ** 2  # (b*v,c,h,w)

        if apply_mask and average_over_mask and not use_frequency_weighting:
            # Original mask-only averaging
            numerator = (rgb_l2_loss * flattened_mask[:, None, ...]).sum()
            denominator = flattened_mask.sum() + epsilon
            mse_loss = numerator / denominator
        else:
            # With frequency weighting or simplified approach
            # (b*v,c,h,w) * (b*v,1,h,w) via broadcasting
            numerator = (rgb_l2_loss * weight_map[:, None, ...]).sum()
            denominator = (weight_map[:, None, ...]).sum() + epsilon
            mse_loss = numerator / denominator

        # ============ LPIPS Loss ============
        lpips_loss_map = self.lpips_criterion(flattened_target_color, flattened_color, normalize=True)  # (b*v,c,h,w)

        if apply_mask and average_over_mask and not use_frequency_weighting:
            numerator = (lpips_loss_map * flattened_mask[:, None, ...]).sum()
            denominator = flattened_mask.sum() + epsilon
            lpips_loss = numerator / denominator
        else:
            numerator = (lpips_loss_map * weight_map[:, None, ...]).sum()
            denominator = (weight_map[:, None, ...]).sum() + epsilon
            lpips_loss = numerator / denominator

        # Combine weighted losses
        loss = self.config.mse_loss_weight * mse_loss + \
            self.config.lpips_loss_weight * lpips_loss

        # ============ Opacity Regularization ============
        opacity_reg_loss = None
        if self.config.opacity_reg_weight is not None:
            if apply_mask:
                masked_opacity = predicted_opacity * mask[..., None]  # (b,v,h,w)
                opacity_reg_loss = torch.abs(masked_opacity).mean()
            else:
                opacity_reg_loss = torch.abs(predicted_opacity).mean()
            loss += self.config.opacity_reg_weight * opacity_reg_loss

        # ============ Optional Spann3r Loss ============
        spann3r_loss = None
        if self.config.spann3r_loss_weight is not None:
            spann3r_loss, _, _ = self.spann3r_criterion.compute_frame_loss(context_views, preds_all)
            loss += self.config.spann3r_loss_weight * spann3r_loss

        # ============ SSIM (Optional) ============
        ssim_val = None
        if calculate_ssim:
            ssim_map = compute_ssim.compute_ssim(flattened_target_color, flattened_color, full=False)  # (b*v,c,h,w)
            if apply_mask and average_over_mask and not use_frequency_weighting:
                numerator = (ssim_map * flattened_mask[:, None, ...]).sum()
                denominator = flattened_mask.sum() + epsilon
                ssim_val = numerator / denominator
            else:
                numerator = (ssim_map * weight_map[:, None, ...]).sum()
                denominator = (weight_map[:, None, ...]).sum() + epsilon
                ssim_val = numerator / denominator

        return loss, mse_loss, lpips_loss, opacity_reg_loss, spann3r_loss, ssim_val

    @staticmethod
    def get_all_means(gts, preds, norm_mode='avg_dis', fix_first=False, dist_clip=None, gt_scale=False):
        """
        Computes normalized points and cameras.
        """
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
                pred_left = get_pred_pts3d(gts, preds[i][0], use_pose=(i != 0))
                pr_pts_l.append(pred_left)
            if i != 0:
                pred_right = get_pred_pts3d(gts, preds[i - 1][1], use_pose=(i != 0))
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
        pr_pts_avg_normalized = [pr_pts_l[0]] + [CustomMeanFunction.apply(pr_pts_l[i+1], pr_pts_r[i]) for i in range(num_frames - 2)] + [pr_pts_r[-1]]
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
        pr_pts_avg = [pr_pts_l[0]] + [CustomMeanFunction.apply(pr_pts_l[i+1], pr_pts_r[i]) for i in range(num_frames - 1)] + [pr_pts_r[-1]]
        pr_pts_avg = torch.stack(pr_pts_avg, dim=1)
        return {'left': pr_pts_l, 'right': pr_pts_r, 'average': pr_pts_avg}

    @staticmethod
    def get_all_scales(preds):
        num_frames = len(preds)
        pr_pts_l = [preds[i][0]['scales'] for i in range(num_frames)]
        pr_pts_r = [preds[i][1]['scales'] for i in range(num_frames)]
        pr_pts_avg = [pr_pts_l[0]] + [CustomMeanFunction.apply(pr_pts_l[i+1], pr_pts_r[i]) for i in range(num_frames - 1)] + [pr_pts_r[-1]]
        pr_pts_avg = torch.stack(pr_pts_avg, dim=1)
        pr_pts_avg = reg_dense_scales(pr_pts_avg)
        return {'left': pr_pts_l, 'right': pr_pts_r, 'average': pr_pts_avg}

    @staticmethod
    def get_all_rotations(preds):
        num_frames = len(preds)
        pr_pts_l = [preds[i][0]['rotations'] for i in range(num_frames)]
        pr_pts_r = [preds[i][1]['rotations'] for i in range(num_frames)]
        pr_pts_avg = [pr_pts_l[0]] + [CustomMeanFunction.apply(pr_pts_l[i+1], pr_pts_r[i]) for i in range(num_frames - 1)] + [pr_pts_r[-1]]
        pr_pts_avg = torch.stack(pr_pts_avg, dim=1)
        pr_pts_avg = reg_dense_rotation(pr_pts_avg)
        return {'left': pr_pts_l, 'right': pr_pts_r, 'average': pr_pts_avg}

    @staticmethod
    def get_all_opacities(preds):
        num_frames = len(preds)
        pr_pts_l = [preds[i][0]['opacities'] for i in range(num_frames)]
        pr_pts_r = [preds[i][1]['opacities'] for i in range(num_frames)]
        pr_pts_avg = [pr_pts_l[0]] + [CustomMeanFunction.apply(pr_pts_l[i+1], pr_pts_r[i]) for i in range(num_frames - 1)] + [pr_pts_r[-1]]
        pr_pts_avg = torch.stack(pr_pts_avg, dim=1)
        pr_pts_avg = reg_dense_opacities(pr_pts_avg)
        return {'left': pr_pts_l, 'right': pr_pts_r, 'average': pr_pts_avg}
