import torch
import torch.nn.functional as F

from .dpt_head import PixelwiseTaskWithDPT  # noqa
from .postprocess import gaussian_postprocess

from models.blocks import Mlp


class GaussianHead(PixelwiseTaskWithDPT):
    """Model that outputs 3D points and Gaussian parameters, without features (descriptors)"""

    def __init__(self, net, has_conf=False, hooks_idx=None, dim_tokens=None, postprocess=None, feature_dim=256, last_dim=32,
                 depth_mode=None, conf_mode=None, head_type="regression",
                 use_offsets=True, sh_degree=1, **kwargs):
        # Set num_channels for pts3d (and optional confidence)
        num_pts3d_channels = 3  # For x, y, z coordinates
        if has_conf:
            num_pts3d_channels += 1  # Add one channel for confidence

        super().__init__(num_channels=num_pts3d_channels, feature_dim=feature_dim, last_dim=last_dim,
                         hooks_idx=hooks_idx, dim_tokens=dim_tokens, depth_mode=depth_mode,
                         postprocess=None, conf_mode=conf_mode, head_type=head_type)
        self.use_offsets = use_offsets
        self.sh_degree = sh_degree
        self.depth_mode = depth_mode
        self.conf_mode = conf_mode
        self.postprocess = postprocess  # Assign the postprocess function

        # Initialize Gaussian DPT
        gaussian_num_channels = 3 + 3 + 4 + 3 * sh_degree + 1  # Total channels for Gaussian parameters
        self.gaussian_dpt = PixelwiseTaskWithDPT(
            num_channels=gaussian_num_channels, feature_dim=feature_dim, last_dim=last_dim,
            hooks_idx=hooks_idx, dim_tokens=dim_tokens, depth_mode=depth_mode,
            postprocess=None, conf_mode=conf_mode, head_type=head_type
        )

        # Initialize Gaussian parameters
        final_conv_layer = self.gaussian_dpt.dpt.head[-1]
        splits_and_inits = [
            (3, 0.001, 0.001),     # 3D mean offsets
            (3, 0.00003, -7.0),    # Scales
            (4, 1.0, 0.0),         # Rotations
            (3 * sh_degree, 1.0, 0.0),  # Spherical Harmonics
            (1, 1.0, -2.0)         # Opacity
        ]
        start_channels = 0
        for out_channels, gain, bias in splits_and_inits:
            torch.nn.init.xavier_uniform_(
                final_conv_layer.weight[start_channels:start_channels+out_channels, :, :, :],
                gain=gain
            )
            torch.nn.init.constant_(
                final_conv_layer.bias[start_channels:start_channels+out_channels],
                bias
            )
            start_channels += out_channels

    def forward(self, decout, img_shape):
        # Compute 3D points (and optional confidence)
        pts3d = self.dpt(decout, image_size=(img_shape[0], img_shape[1]))

        # Compute Gaussian parameters
        gaussian_features = self.gaussian_dpt.dpt(decout, image_size=(img_shape[0], img_shape[1]))

        # Concatenate outputs
        out = torch.cat([pts3d, gaussian_features], dim=1)

        # Post-process outputs
        if self.postprocess:
            out = self.postprocess(
                out,
                depth_mode=self.depth_mode,
                conf_mode=self.conf_mode,
                use_offsets=self.use_offsets,
                sh_degree=self.sh_degree
            )
        return out


def create_gaussian_head(net, has_conf=True, use_offsets=True, sh_degree=1, postprocess=None):
    """
    return PixelwiseTaskWithDPT for given net params
    """
    assert net.dec_depth > 9, "Decoder depth must be greater than 9"
    l2 = net.dec_depth
    feature_dim = 256
    last_dim = feature_dim // 2
    out_nchan = 3  # For pts3d
    if has_conf:
        out_nchan += 1  # Add one channel for confidence
    ed = net.enc_embed_dim
    dd = net.dec_embed_dim
    return GaussianHead(net, has_conf=has_conf,
                        num_channels=out_nchan,
                        feature_dim=feature_dim,
                        last_dim=last_dim,
                        hooks_idx=[0, l2 * 2 // 4, l2 * 3 // 4, l2],
                        dim_tokens=[ed, dd, dd, dd],
                        postprocess=gaussian_postprocess,
                        depth_mode=net.depth_mode,
                        conf_mode=net.conf_mode,
                        head_type='regression',
                        use_offsets=use_offsets,
                        sh_degree=sh_degree)
