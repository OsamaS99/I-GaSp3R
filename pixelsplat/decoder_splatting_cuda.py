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

    def forward(self, batch, preds, image_shape):

        base_pose = batch['context'][0]['camera_pose']  # [b, 4, 4]
        inv_base_pose = torch.inverse(base_pose)

        extrinsics = torch.stack(
            [target_view['camera_pose'] for target_view in batch['target']], dim=1
        )
        intrinsics = torch.stack(
            [target_view['camera_intrinsics'] for target_view in batch['target']], dim=1
        )
        intrinsics = normalize_intrinsics(intrinsics, image_shape)[..., :3, :3]

        # Rotate the ground truth extrinsics into the coordinate system used
        extrinsics = inv_base_pose[:, None, :, :] @ extrinsics

        # Stack the predicted data from all views
        # For 'means', the first pred uses 'means', others use 'means_in_other_view'
        means_list = [preds[0]["means"]] + [pred["means_in_other_view"] for pred in preds[1:]]
        means = torch.stack(means_list, dim=1)
        
        covariances = torch.stack([pred["covariances"] for pred in preds], dim=1)
        harmonics = torch.stack([pred["sh"] for pred in preds], dim=1)
        opacities = torch.stack([pred["opacities"] for pred in preds], dim=1)

        b, v, _, _ = extrinsics.shape
        near = torch.full((b, v), 0.1, device=means.device)
        far = torch.full((b, v), 1000.0, device=means.device)

        color = render_cuda(
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
        return color, None
