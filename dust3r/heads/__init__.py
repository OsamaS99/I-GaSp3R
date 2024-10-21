# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# head factory
# --------------------------------------------------------
from .linear_head import LinearPts3d
from .dpt_head import create_dpt_head
from .gaussian_head import create_gaussian_head


def head_factory(head_type, output_mode, net, has_conf=False):
    """" build a prediction head for the decoder 
    """
    if head_type == 'linear' and output_mode == 'pts3d':
        return LinearPts3d(net, has_conf)
    elif head_type == 'dpt' and output_mode == 'pts3d':
        return create_dpt_head(net, has_conf=has_conf)
    elif head_type == 'gaussian_head' and output_mode == 'pts3d+gaussian':
        create_gaussian_head(net, local_feat_dim=8)
    else:
        raise NotImplementedError(f"unexpected {head_type=} and {output_mode=}")
