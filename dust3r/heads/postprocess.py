# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# post process function for all heads: extract 3D points/confidence from output
# --------------------------------------------------------
import torch
from einops import rearrange



def postprocess(out, depth_mode, conf_mode):
    """
    extract 3D points/confidence from prediction head output
    """
    fmap = out.permute(0, 2, 3, 1)  # B,H,W,3
    res = dict(pts3d=reg_dense_depth(fmap[:, :, :, 0:3], mode=depth_mode))

    if conf_mode is not None:
        res['conf'] = reg_dense_conf(fmap[:, :, :, 3], mode=conf_mode)
    return res


def reg_dense_depth(xyz, mode):
    """
    extract 3D points from prediction head output
    """
    mode, vmin, vmax = mode

    no_bounds = (vmin == -float('inf')) and (vmax == float('inf'))
    assert no_bounds

    if mode == 'linear':
        if no_bounds:
            return xyz  # [-inf, +inf]
        return xyz.clip(min=vmin, max=vmax)

    # distance to origin
    d = xyz.norm(dim=-1, keepdim=True)
    xyz = xyz / d.clip(min=1e-8)

    if mode == 'square':
        return xyz * d.square()

    if mode == 'exp':
        return xyz * torch.expm1(d)

    raise ValueError(f'bad {mode=}')


def reg_dense_conf(x, mode):
    """
    extract confidence from prediction head output
    """
    mode, vmin, vmax = mode
    if mode == 'exp':
        return vmin + x.exp().clip(max=vmax-vmin)
    if mode == 'sigmoid':
        return (vmax - vmin) * torch.sigmoid(x) + vmin
    raise ValueError(f'bad {mode=}')



# @MODIFIED
def reg_dense_offsets(xyz, shift=6.0):
    """
    Apply an activation function to the offsets so that they are small at initialization
    """
    d = xyz.norm(dim=-1, keepdim=True)
    xyz = xyz / d.clip(min=1e-8)
    offsets = xyz * (torch.exp(d - shift) - torch.exp(torch.zeros_like(d) - shift))
    return offsets

# @MODIFIED
def reg_dense_scales(scales):
    """
    Apply an activation function to the offsets so that they are small at initialization
    """
    scales = scales.exp()
    return scales

# @MODIFIED
def reg_dense_rotation(rotations, eps=1e-8):
    """
    Apply PixelSplat's rotation normalization
    """
    return rotations / (rotations.norm(dim=-1, keepdim=True) + eps)

# @MODIFIED
def reg_dense_sh(sh):
    """
    Apply PixelSplat's spherical harmonic postprocessing
    """
    sh = rearrange(sh, '... (xyz d_sh) -> ... xyz d_sh', xyz=3)
    return sh

# @MODIFIED
def reg_dense_opacities(opacities):
    """
    Apply PixelSplat's opacity postprocessing
    """
    return opacities.sigmoid()

# @MODIFIED
def gaussian_postprocess(out, depth_mode, conf_mode, use_offsets=False, sh_degree=1):
    
    
    fmap = out.permute(0, 2, 3, 1)  # B,H,W,D
    assert conf_mode is not None, "Confidence mode must be provided for Gaussian head"
    
    pts3d, conf, offset, scales, rotations, sh, opacities = torch.split(fmap, [3, 1, 3, 3, 4, 3 * sh_degree, 1], dim=-1)
    
    pts3d = reg_dense_depth(pts3d, mode=depth_mode)
    conf = reg_dense_conf(conf.squeeze(-1), mode=conf_mode)
    offset = reg_dense_offsets(offset)
    scales = reg_dense_scales(scales)
    rotations = reg_dense_rotation(rotations)
    sh = reg_dense_sh(sh)
    opacities = reg_dense_opacities(opacities)

    res = {
        'pts3d': pts3d,
        'conf': conf,
        'scales': scales,
        'rotations': rotations,
        'sh': sh,
        'opacities': opacities
    }

    if use_offsets:
        res['means'] = pts3d.detach() + offset
    else:
        res['means'] = pts3d.detach()

    return res
