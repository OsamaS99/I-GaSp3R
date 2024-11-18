import torch
import einops
import lpips

from spann3r.model import SpatialMemory

from spann3r.model import Spann3R
from spann3r.loss import Regr3D_t, ConfLoss_t, normalize_pointcloud_t
from dust3r.inference import get_pred_means
from dust3r.losses import L21

import pixelsplat.decoder_splatting_cuda as pixelsplat_decoder
import utils.compute_ssim as compute_ssim
import utils.geometry as geometry
import utils.sh_utils as sh_utils

class InGaS(Spann3R):
    def __init__(self, spann3r_checkpoint_path, dust3r_checkpoint_path, config=None,
                 use_feat=False, mem_pos_enc=False, memory_dropout=0.15, ):
        # Initialize the base Spann3R model with Gaussian heads
        super(InGaS, self).__init__(
            dus3r_name=dust3r_checkpoint_path,
            use_feat=use_feat,
            mem_pos_enc=mem_pos_enc,
            memory_dropout=memory_dropout,
            is_gaussian=True  # Use Gaussian heads
        )
        self.config = config

        # Load Spann3R weights with strict=False
        state_dict = torch.load(spann3r_checkpoint_path)

        # If the state dictionary is saved under a 'model' key, extract it
        if 'model' in state_dict:
            state_dict = state_dict['model']
        else:
            state_dict = state_dict

        missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)

        # Filter out Gaussian head keys from missing_keys
        # gaussian_keys = [key for key in missing_keys if 'gaussian' in key]
        # other_missing_keys = [key for key in missing_keys if 'gaussian' not in key]

        # if other_missing_keys:
        #     print("Warning: Other missing keys:", other_missing_keys)

        # print("Missing Gaussian keys (expected):", gaussian_keys)
        # print("Unexpected keys (should be empty):", unexpected_keys)

        # Freeze all weights except Gaussian heads
        self.requires_grad_(False)
        self.dust3r.downstream_head1.gaussian_dpt.dpt.requires_grad_(True)
        self.dust3r.downstream_head2.gaussian_dpt.dpt.requires_grad_(True)


        # Initialize decoder and benchmarker
        self.decoder = pixelsplat_decoder.DecoderSplattingCUDA(
            background_color=[0.0, 0.0, 0.0]
        )

        # Loss criteria
        if self.config.average_over_mask:
            self.lpips_criterion = lpips.LPIPS('vgg', spatial=True)
        else:
            self.lpips_criterion = lpips.LPIPS('vgg')

        if config.spann3r_loss_weight is not None:
            self.spann3r_criterion = ConfLoss_t(Regr3D_t(L21, norm_mode='avg_dis', fix_first=False), alpha=0.4)
            self.dust3r.downstream_head1.requires_grad_(True)
            self.dust3r.downstream_head2.requires_grad_(True)


    def forward(self, frames, return_memory=False):
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

            ##### Encode frames
            feat1, feat2, pos1, pos2, shape1, shape2 = self.encode_frames(
                view1, view2, feat1, feat2, pos1, pos2, shape1, shape2
            )

            ##### Memory readout
            if feat_k2 is not None:
                feat_fuse = sp_mem.memory_read(feat_k2, res=True)
            else:
                feat_fuse = feat1
            
            ##### Decode features
            dec1, dec2 = self.decode(feat_fuse, pos1, feat2, pos2)
            
            ##### Encode feat key
            feat_k1 = self.encode_feat_key(feat1, dec1[-1], 1)
            feat_k2 = self.encode_feat_key(feat2, dec2[-1], 2)

            ##### Regress pointmaps
            with torch.cuda.amp.autocast(enabled=False):
                res1 = self.downstream_head(dec1, shape1, 1)
                res2 = self.downstream_head(dec2, shape2, 2)

            res1['covariances'] = geometry.build_covariance(res1['scales'], res1['rotations'])
            res2['covariances'] = geometry.build_covariance(res2['scales'], res2['rotations'])

            learn_residual = True
            if learn_residual:
                new_sh1 = torch.zeros_like(res1['sh'])
                new_sh2 = torch.zeros_like(res2['sh'])
                new_sh1[..., 0] = sh_utils.RGB2SH(einops.rearrange(view1['original_img'], 'b c h w -> b h w c'))
                new_sh2[..., 0] = sh_utils.RGB2SH(einops.rearrange(view2['original_img'], 'b c h w -> b h w c'))
                res1['sh'] = res1['sh'] + new_sh1
                res2['sh'] = res2['sh'] + new_sh2

            ##### Memory update
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
    
    def calculate_loss(self, batch, preds_all, color, mask, apply_mask=True, average_over_mask=True, calculate_ssim=False):
        # Define the target frames as the first and last frames
        target_views = batch #[batch[0], batch[-1]]
        target_color = torch.stack([view['img'] for view in target_views], dim=1)
        predicted_color = color

        if apply_mask:
            assert mask.sum() > 0, "There are no valid pixels in the mask!"
            # Apply the mask to both target and predicted colors
            target_color = target_color * mask[..., None, :, :]
            predicted_color = predicted_color * mask[..., None, :, :]

        # Flatten the color tensors for loss computation
        flattened_color = einops.rearrange(predicted_color, 'b v c h w -> (b v) c h w')
        flattened_target_color = einops.rearrange(target_color, 'b v c h w -> (b v) c h w')
        flattened_mask = einops.rearrange(mask, 'b v h w -> (b v) h w')

        # Compute MSE loss
        rgb_l2_loss = (flattened_color - flattened_target_color) ** 2
        if average_over_mask:
            mse_loss = (rgb_l2_loss * flattened_mask[:, None, ...]).sum() / flattened_mask.sum()
        else:
            mse_loss = rgb_l2_loss.mean()

        # Compute LPIPS loss
        lpips_loss = self.lpips_criterion(flattened_target_color, flattened_color, normalize=True)
        if average_over_mask:
            lpips_loss = (lpips_loss * flattened_mask[:, None, ...]).sum() / flattened_mask.sum()
        else:
            lpips_loss = lpips_loss.mean()

        # Total loss
        loss = self.config.mse_loss_weight * mse_loss + self.config.lpips_loss_weight * lpips_loss

        # Optionally, include other losses if needed
        if self.config.spann3r_loss_weight is not None:
            # Adjust dust3r loss calculation based on the spann3r structure
            # For spann3r, we're assuming frames are organized in sequence with target as first and last frame
            spann3r_loss = self.spann3r_criterion.compute_frame_loss(batch, preds_all)
            loss += self.config.spann3r_criterion * spann3r_loss

        # Masked SSIM
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
    def get_all_means(gts, preds, norm_mode='avg_dis', fix_first=False, dist_clip=None):
        """
        Processes ground truths and predictions to compute left, right, and average 'means' predictions.
        The average is computed after normalization.

        Args:
            gts (list): List of ground truth dictionaries for each frame.
            preds (list): List of predictions for each frame. Each prediction is a list or tuple
                          containing two dictionaries (left and right head outputs).
            norm_mode (str, optional): Normalization mode. Defaults to 'avg_dis'.
            fix_first (bool, optional): Whether to fix the first prediction during normalization. Defaults to False.
            dist_clip (float, optional): Distance threshold to clip predictions. Points beyond this distance are invalid. Defaults to None.

        Returns:
            tuple: 
                - combined_pr_pts (dict): Dictionary containing 'left', 'right', and 'average' 'means' predictions.
                - pr_factor (various): Normalization factor returned by `normalize_pointcloud_t`.
                - valids (list): List of valid masks for each frame.
                - additional_info (dict): Additional information (empty in this implementation).
        """
        valids = []
        pr_pts_l = []
        pr_pts_r = []
        num_frames = len(gts)

        for i, gt in enumerate(gts):
            valid = gt['valid_mask'].clone()

            if dist_clip is not None:
                dis = gt['means'].norm(dim=-1)
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
            pr_pts_normalized, pr_factor = normalize_pointcloud_t(
                pr_pts, norm_mode, valids, fix_first=fix_first, gt=False
            )
            pr_pts_l_normalized, pr_pts_r_normalized = pr_pts_normalized
            pr_pts_avg_normalized = []

            for i in range(num_frames):
                if i == 0:
                    avg_pt = pr_pts_l_normalized[0] if pr_pts_l_normalized else None
                elif i == num_frames - 1:
                    avg_pt = pr_pts_r_normalized[-1] if pr_pts_r_normalized else None
                else:
                    left_pt = pr_pts_l_normalized[i] if i < len(pr_pts_l_normalized) else None
                    right_pt = pr_pts_r_normalized[i-1] if (i-1) < len(pr_pts_r_normalized) else None
                    if left_pt is not None and right_pt is not None:
                        avg_pt = (left_pt + right_pt) / 2
                    elif left_pt is not None:
                        avg_pt = left_pt
                    elif right_pt is not None:
                        avg_pt = right_pt
                    else:
                        avg_pt = None
                pr_pts_avg_normalized.append(avg_pt)
        else:
            pr_pts_normalized = pr_pts
            pr_pts_avg_normalized = []
            for i in range(num_frames):
                if i == 0:
                    avg_pt = pr_pts_l[0] if pr_pts_l else None
                elif i == num_frames - 1:
                    avg_pt = pr_pts_r[-1] if pr_pts_r else None
                else:
                    left_pt = pr_pts_l[i] if i < len(pr_pts_l) else None
                    right_pt = pr_pts_r[i-1] if (i-1) < len(pr_pts_r) else None
                    if left_pt is not None and right_pt is not None:
                        avg_pt = (left_pt + right_pt) / 2
                    elif left_pt is not None:
                        avg_pt = left_pt
                    elif right_pt is not None:
                        avg_pt = right_pt
                    else:
                        avg_pt = None
                pr_pts_avg_normalized.append(avg_pt)
            pr_factor = None

        pr_pts_avg_normalized = [pt for pt in pr_pts_avg_normalized if pt is not None]

        combined_pr_pts = {
            'left': pr_pts_normalized[0] if norm_mode else pr_pts_normalized[0],
            'right': pr_pts_normalized[1] if norm_mode else pr_pts_normalized[1],
            'average': pr_pts_avg_normalized
        }

        return combined_pr_pts, pr_factor, valids, {}

    @staticmethod
    def get_all_sh(preds):
        num_frames = len(preds)
        pr_pts_l = [preds[i][0]['sh'] for i in range(num_frames)]
        pr_pts_r = [preds[i][1]['sh'] for i in range(num_frames)]
        pr_pts_avg = [pr_pts_l[0]] + [(pr_pts_l[i + 1] + pr_pts_r[i]) / 2 for i in range(num_frames - 1)] + [pr_pts_r[-1]]
                
        combined_pr_pts = {
            'left': pr_pts_l,
            'right': pr_pts_r,
            'average': pr_pts_avg
        }
        
        return combined_pr_pts


    @staticmethod
    def get_all_covariances(preds):
        num_frames = len(preds)
        pr_pts_l = [preds[i][0]['covariances'] for i in range(num_frames)]
        pr_pts_r = [preds[i][1]['covariances'] for i in range(num_frames)]
        pr_pts_avg = [pr_pts_l[0]] + [(pr_pts_l[i + 1] + pr_pts_r[i]) / 2 for i in range(num_frames - 1)] + [pr_pts_r[-1]]
        
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
        pr_pts_avg = [pr_pts_l[0]] + [(pr_pts_l[i + 1] + pr_pts_r[i]) / 2 for i in range(num_frames - 1)] + [pr_pts_r[-1]]
        
        combined_pr_pts = {
            'left': pr_pts_l,
            'right': pr_pts_r,
            'average': pr_pts_avg
        }
        
        return combined_pr_pts
