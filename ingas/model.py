import torch
import einops
import lpips
import torch.nn as nn

from dust3r.losses import L21, ConfLoss, Regr3D
from spann3r.model import SpatialMemory

from spann3r.model import Spann3R
import pixelsplat.benchmarker as benchmarker
import pixelsplat.decoder_splatting_cuda as pixelsplat_decoder
import utils.compute_ssim as compute_ssim
from dust3r.model import AsymmetricCroCo3DStereo
import utils.geometry as geometry
import utils.sh_utils as sh_utils

class InGaS(Spann3R):
    def __init__(self, spann3r_checkpoint_path, dust3r_checkpoint_path, config=None,
                 use_feat=False, mem_pos_enc=False, memory_dropout=0.15):
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
        gaussian_keys = [key for key in missing_keys if 'gaussian' in key]
        other_missing_keys = [key for key in missing_keys if 'gaussian' not in key]

        if other_missing_keys:
            print("Warning: Other missing keys:", other_missing_keys)

        print("Missing Gaussian keys (expected):", gaussian_keys)
        print("Unexpected keys (should be empty):", unexpected_keys)

        # Freeze all weights except Gaussian heads
        for name, param in self.named_parameters():
            if 'gaussian' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        # Initialize decoder and benchmarker
        self.decoder = pixelsplat_decoder.DecoderSplattingCUDA(
            background_color=[0.0, 0.0, 0.0]
        )

        # Loss criteria
        if self.config.loss.average_over_mask:
            self.lpips_criterion = lpips.LPIPS('vgg', spatial=True)
        else:
            self.lpips_criterion = lpips.LPIPS('vgg')

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
        target_color = torch.stack([target_view['original_img'] for target_view in batch['target']], dim=1)
        predicted_color = color

        if apply_mask:
            assert mask.sum() > 0, "There are no valid pixels in the mask!"
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
        loss = self.config.loss.mse_loss_weight * mse_loss + self.config.loss.lpips_loss_weight * lpips_loss

        # Optionally, include other losses if needed
        if self.config.loss.dust3r_loss_weight is not None:
            mast3r_loss = self.mast3r_criterion(batch['frames'][0], batch['frames'][1], preds_all[0][0], preds_all[0][1])[0]
            loss += self.config.loss.dust3r_loss_weight * mast3r_loss

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

    def log_metrics(self, prefix, loss, mse, lpips, ssim=None):
        values = {
            f'{prefix}/loss': loss,
            f'{prefix}/mse': mse,
            f'{prefix}/psnr': -10.0 * mse.log10(),
            f'{prefix}/lpips': lpips,
        }

        if ssim is not None:
            values[f'{prefix}/ssim'] = ssim

        # Implement logging using your preferred method
        # For example, using TensorBoard:
        if self.log_writer is not None:
            for key, value in values.items():
                self.log_writer.add_scalar(key, value, self.current_epoch)
