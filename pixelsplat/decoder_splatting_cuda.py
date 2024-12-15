import torch
from einops import rearrange, repeat

from .cuda_splatting import render_cuda
from utils.geometry import normalize_intrinsics

class DecoderSplattingCUDA(torch.nn.Module):

    def __init__(self, background_color):
        super().__init__()
        self.register_buffer(
            "background_color",
            torch.tensor(background_color, dtype=torch.float32),
            persistent=False,
        )

    def forward(self, extrinsics, intrinsics, means, harmonics, covariances, opacities, image_shape):
        

        intrinsics = normalize_intrinsics(intrinsics, image_shape)[..., :3, :3]
        b, v, _, _ = extrinsics.shape
        b, c, h, w, _= opacities.shape

        near = torch.full((b, v), 0.01, device=means.device)
        far = torch.full((b, v), 100.0, device=means.device)

        color, freq = render_cuda(
            rearrange(extrinsics, "b v i j -> (b v) i j"),
            rearrange(intrinsics, "b v i j -> (b v) i j"),
            rearrange(near, "b v -> (b v)"),
            rearrange(far, "b v -> (b v)"),
            image_shape,
            repeat(self.background_color, "c -> (b v) c", b=b, v=v),
            repeat(
                rearrange(means, "b v h w xyz -> b (v h w) xyz"), "b g xyz -> (b v) g xyz", v=v
            ),
            repeat(
                rearrange(covariances, "b v h w i j -> b (v h w) i j"),
                "b g i j -> (b v) g i j",
                v=v,
            ),
            repeat(
                rearrange(harmonics, "b v h w c d_sh -> b (v h w) c d_sh"),
                "b g c d_sh -> (b v) g c d_sh",
                v=v,
            ),
            repeat(
                rearrange(opacities, "b v h w 1 -> b (v h w)"),
                "b g -> (b v) g",
                v=v,
            ),
        )
        color = rearrange(color, "(b v) c h w -> b v c h w", b=b, v=v)
        freq = rearrange(freq, "(b v) (c h w) -> b v c h w", b=b, v=v, c=c, h=h, w=w)
        return color, freq
