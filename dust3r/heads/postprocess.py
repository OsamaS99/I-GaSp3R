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
    # d = xyz.norm(dim=-1, keepdim=True)
    # xyz = xyz / d.clip(min=1e-8)
    # offsets = xyz * (torch.exp(d - shift) - torch.exp(torch.zeros_like(d) - shift))
    return xyz

# @MODIFIED
def reg_dense_scales(scales):
    """
    Apply an activation function to the offsets so that they are small at initialization
    """
    # scales = scales.sigmoid()
    return scales

# @MODIFIED
def reg_dense_rotation(rotations, eps=1e-8):
    """
    Apply PixelSplat's rotation normalization
    """
    # rotations = rotations / (rotations.norm(dim=-1, keepdim=True) + eps)
    return rotations 

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
    return opacities

# @MODIFIED
def gaussian_postprocess(out, depth_mode, conf_mode, use_offsets=True, sh_degree=2):
    fmap = out.permute(0, 2, 3, 1)  # B, H, W, D

    # Calculate the expected number of channels
    # pts3d (3) + [conf (1)] + offset (3) + scales (3) + rotations (4) + sh (3 * sh_degree) + opacities (1)
    num_gaussian_channels = 3 + 3 + 4 + 3 * ((sh_degree + 1) ** 2) + 1  # Excluding conf

    if conf_mode is not None:
        expected_channels = 3 + 1 + num_gaussian_channels  # pts3d + conf + gaussian params
    else:
        expected_channels = 3 + num_gaussian_channels  # pts3d + gaussian params

    num_channels = fmap.shape[-1]
    assert num_channels == expected_channels, f"Unexpected number of channels: got {num_channels}, expected {expected_channels}"

    # Split the fmap into components
    split_sizes = [3]  # pts3d
    if conf_mode is not None:
        split_sizes.append(1)  # conf

    split_sizes.extend([
        3,             # offset
        3,             # scales
        4,             # rotations
        3 * ((sh_degree + 1) ** 2), # spherical harmonics
        1              # opacities 
    ])

    components = torch.split(fmap, split_sizes, dim=-1)
    idx = 0

    pts3d = components[idx]; idx += 1
    if conf_mode is not None:
        conf = components[idx].squeeze(-1); idx += 1
    else:
        conf = None

    offset = components[idx]; idx += 1
    scales = components[idx]; idx += 1
    rotations = components[idx]; idx += 1
    sh = components[idx]; idx += 1
    opacities = components[idx]; idx += 1

    # Apply post-processing to each component
    pts3d = reg_dense_depth(pts3d, mode=depth_mode)
    if conf is not None:
        conf = reg_dense_conf(conf, mode=conf_mode)
    offset = reg_dense_offsets(offset)
    scales = reg_dense_scales(scales)
    rotations = reg_dense_rotation(rotations)
    sh = reg_dense_sh(sh)
    opacities = reg_dense_opacities(opacities)

    # Prepare the result dictionary
    res = {
        'pts3d': pts3d,
        'scales': scales,
        'rotations': rotations,
        'sh': sh,
        'opacities': opacities,
        'offsets': offset
    }
    if conf is not None:
        res['conf'] = conf

    if use_offsets:
        res['means'] = pts3d.detach()
    else:
        res['means'] = pts3d.detach()

    return res
