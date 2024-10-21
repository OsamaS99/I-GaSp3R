import torch
import torch.nn.functional as F

from dpt_head import PixelwiseTaskWithDPT  # noqa
from postprocess import gaussian_postprocess

from models.blocks import Mlp


class GaussianHead(PixelwiseTaskWithDPT):
    """Version of the above, modified to also output Gaussian parameters"""

    def __init__(self, net, has_conf=False, local_feat_dim=16, hidden_dim_factor=4., hooks_idx=None, dim_tokens=None,
                 num_channels=1, postprocess=None, feature_dim=256, last_dim=32, depth_mode=None, conf_mode=None, head_type="regression", use_offsets=False, sh_degree=1, **kwargs):
        super().__init__(num_channels=num_channels, feature_dim=feature_dim, last_dim=last_dim, hooks_idx=hooks_idx,
                         dim_tokens=dim_tokens, depth_mode=depth_mode, postprocess=postprocess, conf_mode=conf_mode, head_type=head_type)
        self.local_feat_dim = local_feat_dim

        patch_size = net.patch_embed.patch_size
        if isinstance(patch_size, tuple):
            assert len(patch_size) == 2 and isinstance(patch_size[0], int) and isinstance(
                patch_size[1], int), "What is your patchsize format? Expected a single int or a tuple of two ints."
            assert patch_size[0] == patch_size[1], "Error, non square patches not managed"
            patch_size = patch_size[0]
        self.patch_size = patch_size

        self.desc_mode = net.desc_mode
        self.has_conf = has_conf
        self.two_confs = net.two_confs  # independent confs for 3D regr and descs
        self.desc_conf_mode = net.desc_conf_mode
        idim = net.enc_embed_dim + net.dec_embed_dim

        self.head_local_features = Mlp(in_features=idim,
                                       hidden_features=int(hidden_dim_factor * idim),
                                       out_features=(self.local_feat_dim + self.two_confs) * self.patch_size**2)

        # Gaussian Num Channels =
            # 3D mean offsets (3) +
            # Scales (3) +
            # Rotations (4) +
            # Spherical Harmonics (3 * sh_degree) +
            # Opacity (1)
        gaussian_num_channels = 3 + 3 + 4 + 3 * sh_degree + 1
        self.gaussian_dpt = PixelwiseTaskWithDPT(
            num_channels=gaussian_num_channels, feature_dim=feature_dim, last_dim=last_dim, hooks_idx=hooks_idx,
            dim_tokens=dim_tokens, depth_mode=depth_mode, postprocess=postprocess, conf_mode=conf_mode, head_type=head_type
        )
        
        final_conv_layer = self.gaussian_dpt.dpt.head[-1]
        splits_and_inits = [
            (3, 0.001, 0.001),  # 3D mean offsets
            (3, 0.00003, -7.0),  # Scales
            (4, 1.0, 0.0),  # Rotations
            (3 * sh_degree, 1.0, 0.0),  # Spherical Harmonics
            (1, 1.0, -2.0)  # Opacity
        ]
        start_channels = 0
        for out_channel, s, b in splits_and_inits:
            torch.nn.init.xavier_uniform_(
                final_conv_layer.weight[start_channels:start_channels+out_channel, :, :, :],
                s
            )
            torch.nn.init.constant_(
                final_conv_layer.bias[start_channels:start_channels+out_channel],
                b
            )
            start_channels += out_channel

        self.use_offsets = use_offsets
        self.sh_degree = sh_degree


    def forward(self, decout, img_shape):
        # pass through the heads
        pts3d = self.dpt(decout, image_size=(img_shape[0], img_shape[1]))

        # recover encoder and decoder outputs
        enc_output, dec_output = decout[0], decout[-1]
        cat_output = torch.cat([enc_output, dec_output], dim=-1)  # concatenate
        H, W = img_shape
        B, S, D = cat_output.shape

        # extract local_features
        local_features = self.head_local_features(cat_output)  # B,S,D
        local_features = local_features.transpose(-1, -2).view(B, -1, H // self.patch_size, W // self.patch_size)
        local_features = F.pixel_shuffle(local_features, self.patch_size)  # B,d,H,W

        # extract gaussian_features
        gaussian_features = self.gaussian_dpt.dpt(decout, image_size=(img_shape[0], img_shape[1]))
        # gaussian_features = self.gaussian_local_features(cat_output)  # B,S,D
        # gaussian_features = gaussian_features.transpose(-1, -2).view(B, -1, H // self.patch_size, W // self.patch_size)
        # gaussian_features = F.pixel_shuffle(gaussian_features, self.patch_size)  # B,d,H,W

        # post process 3D pts, descriptors and confidences
        out = torch.cat([pts3d, local_features, gaussian_features], dim=1)
        if self.postprocess:
            out = self.postprocess(out,
                                   depth_mode=self.depth_mode,
                                   conf_mode=self.conf_mode,
                                   desc_dim=self.local_feat_dim,
                                   desc_mode=self.desc_mode,
                                   two_confs=self.two_confs,
                                   desc_conf_mode=self.desc_conf_mode,
                                   use_offsets=self.use_offsets,
                                   sh_degree=self.sh_degree)
        return out


def create_gaussian_head(net, local_feat_dim, has_conf=False, use_offsets=False, sh_degree=1):
    """
    return PixelwiseTaskWithDPT for given net params
    """
    assert net.dec_depth > 9
    l2 = net.dec_depth
    feature_dim = 256
    last_dim = feature_dim // 2
    out_nchan = 3
    ed = net.enc_embed_dim
    dd = net.dec_embed_dim
    return GaussianHead(net, local_feat_dim=local_feat_dim, has_conf=has_conf,
                                            num_channels=out_nchan + has_conf,
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
