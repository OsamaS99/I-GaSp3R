import PIL
import numpy as np
import torch
import torchvision

from dust3r.datasets.base.easy_dataset import EasyDataset
from dust3r.datasets.utils.transforms import ImgNorm
from dust3r.utils.geometry import depthmap_to_absolute_camera_coordinates
import dust3r.datasets.utils.cropping as cropping


class BaseStereoViewDataset(EasyDataset):
    """Base class for datasets with multiple views.

    Subclasses should override the `_get_views` method.

    Usage:
        class MyDataset(BaseStereoViewDataset):
            def _get_views(self, idx, resolution, rng):
                # Implementation here
                views = []
                views.append(dict(img=..., ...))
                return views
    """

    def __init__(self, *,  # only keyword arguments
                 split=None,
                 resolution=None,  # int or (width, height) or list of (width, height)
                 transform=ImgNorm,
                 aug_crop=False,
                 seed=None,
                 num_views=2):
        super().__init__()
        self.num_views = num_views
        self.split = split
        self._set_resolutions(resolution)

        self.transform = transform
        self.original_transform = torchvision.transforms.ToTensor()
        if isinstance(transform, str):
            self.transform = eval(transform)

        self.aug_crop = aug_crop
        self.seed = seed

    def __len__(self):
        raise NotImplementedError("Subclasses should implement __len__ method.")

    def get_stats(self):
        return f"{len(self)} samples"

    def __repr__(self):
        resolutions_str = '[' + ';'.join(f'{w}x{h}' for w, h in self._resolutions) + ']'
        return f"""{type(self).__name__}({self.get_stats()},
            split={self.split},
            seed={self.seed},
            resolutions={resolutions_str},
            transform={self.transform})""".replace('\n', '').replace('   ', '')

    def _get_views(self, idx, resolution, rng):
        raise NotImplementedError("This method should be overridden by subclasses.")

    def __getitem__(self, idx):
        if isinstance(idx, tuple):
            # The idx is specifying the aspect ratio index
            idx, ar_idx = idx
        else:
            assert len(self._resolutions) == 1, "Multiple resolutions provided but no aspect ratio index."
            ar_idx = 0

        # Set up the RNG
        if self.seed is not None:
            self._rng = np.random.default_rng(seed=self.seed)
        elif not hasattr(self, '_rng'):
            seed = torch.initial_seed()
            self._rng = np.random.default_rng(seed=seed)

        # Overridden code
        resolution = self._resolutions[ar_idx]
        views = self._get_views(idx, resolution, self._rng)

        # Check data types and process each view
        for v, view in enumerate(views):
            assert 'pts3d' not in view, f"pts3d should not be in the view; it will be computed later for view {view_name(view)}"
            view['idx'] = (idx, ar_idx, v)

            # Encode the image
            width, height = view['img'].size
            view['true_shape'] = np.int32((height, width))
            view['original_img'] = self.original_transform(view['img'])
            view['img'] = self.transform(view['img'])

            assert 'camera_intrinsics' in view, f"camera_intrinsics missing in view {view_name(view)}"
            if 'camera_pose' not in view:
                view['camera_pose'] = np.full((4, 4), np.nan, dtype=np.float32)
            else:
                assert np.isfinite(view['camera_pose']).all(), f'NaN in camera pose for view {view_name(view)}'
            assert np.isfinite(view['depthmap']).all(), f'NaN in depthmap for view {view_name(view)}'
            pts3d, valid_mask = depthmap_to_absolute_camera_coordinates(**view)

            view['pts3d'] = pts3d
            view['valid_mask'] = valid_mask & np.isfinite(pts3d).all(axis=-1)

            # Check all data types
            for key, val in view.items():
                res, err_msg = is_good_type(key, val)
                assert res, f"{err_msg} with {key}={val} for view {view_name(view)}"

            K = view['camera_intrinsics']
            assert K.shape == (3, 3), f"Invalid camera_intrinsics shape for view {view_name(view)}"

        # Adjust image orientation if necessary
        for view in views:
            transpose_to_landscape(view)
            # Store RNG state for debugging
            view['rng'] = int.from_bytes(self._rng.bytes(4), 'big')

        return views

    def _set_resolutions(self, resolutions):
        """Set the resolution(s) of the dataset.

        Params:
            - resolutions: int or tuple or list of tuples
        """
        assert resolutions is not None, 'Resolution must be defined.'

        if not isinstance(resolutions, list):
            resolutions = [resolutions]

        self._resolutions = []
        for resolution in resolutions:
            if isinstance(resolution, int):
                width = height = resolution
            else:
                width, height = resolution
            assert isinstance(width, int), f'Invalid type for width={width}, expected int.'
            assert isinstance(height, int), f'Invalid type for height={height}, expected int.'
            self._resolutions.append((width, height))

    def _crop_resize_if_necessary(self, image, depthmap, intrinsics, resolution, rng=None, info=None):
        """Crop and resize the image and depth map if necessary.

        This function:
            - First downsizes the image with Lanczos interpolation,
              which is better than bilinear interpolation in preserving details.
            - Then crops the image centered on the principal point.
        """
        if not isinstance(image, PIL.Image.Image):
            image = PIL.Image.fromarray(image)

        # Downscale with Lanczos interpolation
        W, H = image.size
        cx, cy = intrinsics[:2, 2].round().astype(int)

        # Calculate min distance to margin
        min_margin_x = min(cx, W - cx)
        min_margin_y = min(cy, H - cy)
        assert min_margin_x > W / 5, f'Bad principal point in view={info}'
        assert min_margin_y > H / 5, f'Bad principal point in view={info}'

        # Center crop
        l, t = cx - min_margin_x, cy - min_margin_y
        r, b = cx + min_margin_x, cy + min_margin_y
        crop_bbox = (l, t, r, b)
        image, depthmap, intrinsics = cropping.crop_image_depthmap(image, depthmap, intrinsics, crop_bbox)

        # Transpose the resolution if necessary
        W, H = image.size  # New size
        if H > W:
            resolution = resolution[::-1]

        # High-quality Lanczos down-scaling
        target_resolution = np.array(resolution)
        if self.aug_crop > 1:
            target_resolution += rng.integers(0, self.aug_crop)

        # Rescale
        image, depthmap, intrinsics = cropping.rescale_image_depthmap(image, depthmap, intrinsics, target_resolution)

        # Actual cropping (if necessary) with bilinear interpolation
        intrinsics2 = cropping.camera_matrix_of_crop(intrinsics, image.size, resolution, offset_factor=0.5)
        crop_bbox = cropping.bbox_from_intrinsics_in_out(intrinsics, intrinsics2, resolution)
        image, depthmap, intrinsics2 = cropping.crop_image_depthmap(image, depthmap, intrinsics, crop_bbox)

        return image, depthmap, intrinsics2


def is_good_type(key, v):
    """Checks if the data type of a value is acceptable.

    Returns:
        (is_good, err_msg)
    """
    if isinstance(v, (str, int, tuple)):
        return True, None
    if isinstance(v, np.ndarray) and v.dtype in (np.float32, np.int32, np.int64, np.uint8, bool):
        return True, None
    if isinstance(v, torch.Tensor) and v.dtype in (torch.float32, torch.int32, torch.int64, torch.uint8, torch.bool):
        return True, None
    return False, f"Unsupported data type {type(v)} with dtype {getattr(v, 'dtype', None)}"


def view_name(view, batch_index=None):
    def sel(x): return x[batch_index] if batch_index not in (None, slice(None)) else x
    db = sel(view.get('dataset', 'unknown'))
    label = sel(view.get('label', 'unknown'))
    instance = sel(view.get('instance', 'unknown'))
    return f"{db}/{label}/{instance}"


def transpose_to_landscape(view):
    height, width = view['true_shape']

    if width < height:
        # Rectify portrait to landscape
        assert view['img'].shape == (3, height, width), f"Unexpected image shape {view['img'].shape}"
        view['img'] = view['img'].swapaxes(1, 2)

        assert view['valid_mask'].shape == (height, width), f"Unexpected valid_mask shape {view['valid_mask'].shape}"
        view['valid_mask'] = view['valid_mask'].swapaxes(0, 1)

        assert view['depthmap'].shape == (height, width), f"Unexpected depthmap shape {view['depthmap'].shape}"
        view['depthmap'] = view['depthmap'].swapaxes(0, 1)

        assert view['pts3d'].shape == (height, width, 3), f"Unexpected pts3d shape {view['pts3d'].shape}"
        view['pts3d'] = view['pts3d'].swapaxes(0, 1)

        # Transpose x and y pixels in intrinsics
        K = view['camera_intrinsics']
        K = np.array([[K[1, 1], K[1, 0], K[1, 2]],
                      [K[0, 1], K[0, 0], K[0, 2]],
                      [K[2, 1], K[2, 0], K[2, 2]]], dtype=K.dtype)
        view['camera_intrinsics'] = K
