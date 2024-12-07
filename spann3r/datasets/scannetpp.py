import os
import json
import cv2
import numpy as np
import os.path as osp
from collections import deque

from dust3r.utils.image import imread_cv2
from .base_many_view_dataset import BaseManyViewDataset


def orthonormalize_rotation_np(cam_pose: np.ndarray) -> np.ndarray:
    """
    Orthonormalize the rotation component of a 4x4 camera pose matrix to ensure det(R)=1.
    Uses pure NumPy SVD.
    """
    R = cam_pose[:3, :3]  # Extract rotation
    U, S, Vt = np.linalg.svd(R)  # R = U S V^T
    R_fixed = U @ Vt

    # Ensure a positive determinant; if negative, flip the last column of U
    if np.linalg.det(R_fixed) < 0:
        U[:, -1] *= -1
        R_fixed = U @ Vt

    cam_pose[:3, :3] = R_fixed
    return cam_pose


class Scannetpp(BaseManyViewDataset):
    def __init__(self, num_seq=100, num_frames=5,
                 test_id=None, full_video=False,
                 kf_every=1, *args, ROOT, **kwargs):
        self.ROOT = ROOT
        super().__init__(num_frames=num_frames, *args, **kwargs)
        self.num_seq = num_seq
        self.test_id = test_id
        self.full_video = full_video
        self.kf_every = kf_every

        # Load all scenes and exclude bad ones
        self.load_all_scenes(ROOT)

        # Load coverage data for overlap-based sampling
        self.coverage = {}
        self.load_coverage_data()

    def __len__(self):
        return len(self.scene_list) * self.num_seq

    def load_all_scenes(self, base_dir):
        if self.test_id is None:
            meta_split = osp.join(base_dir, 'splits', f'nvs_sem_{self.split}.txt')

            if not osp.exists(meta_split):
                raise FileNotFoundError(f"Split file {meta_split} not found")

            with open(meta_split, 'r') as f:
                sequences = f.read().splitlines()

            # Exclude bad scenes
            if self.split == 'train':
                bad_scenes = ['303745abc7']
            elif self.split in ['val', 'test']:
                bad_scenes = ['cc5237fd77']
            else:
                bad_scenes = []

            sequences = [s for s in sequences if s not in bad_scenes]

            # Exclude sequences with no images or missing required files
            valid_sequences = []
            scenes_with_no_good_frames = []

            for sequence in sequences:
                images_dir = osp.join(base_dir, 'data', sequence, 'dslr', 'undistorted_images')
                train_test_lists_path = osp.join(base_dir, 'data', sequence, 'dslr', 'train_test_lists.json')

                if not osp.exists(images_dir) or not osp.exists(train_test_lists_path):
                    continue

                # Check if there are any good frames in the sequence
                if not self.sequence_has_good_frames(base_dir, sequence):
                    scenes_with_no_good_frames.append(sequence)
                    continue

                valid_sequences.append(sequence)

            self.scene_list = valid_sequences
            print(f"Found {len(self.scene_list)} valid scenes in split {self.split}")
            if scenes_with_no_good_frames:
                print(f"Excluded scenes with no good frames: {scenes_with_no_good_frames}")

        else:
            if isinstance(self.test_id, list):
                sequences = self.test_id
            else:
                sequences = [self.test_id]

            # Apply the same exclusion criteria to the test_id(s)
            if self.split == 'train':
                bad_scenes = ['303745abc7']
            elif self.split in ['val', 'test']:
                bad_scenes = ['cc5237fd77']
            else:
                bad_scenes = []

            sequences = [s for s in sequences if s not in bad_scenes]

            valid_sequences = []
            scenes_with_no_good_frames = []

            for sequence in sequences:
                images_dir = osp.join(base_dir, 'data', sequence, 'dslr', 'undistorted_images')
                train_test_lists_path = osp.join(base_dir, 'data', sequence, 'dslr', 'train_test_lists.json')

                if not osp.exists(images_dir) or not osp.exists(train_test_lists_path):
                    continue

                # Check if there are any good frames in the sequence
                if not self.sequence_has_good_frames(base_dir, sequence):
                    scenes_with_no_good_frames.append(sequence)
                    continue

                valid_sequences.append(sequence)

            self.scene_list = valid_sequences
            print(f"Test scenes: {self.scene_list}")
            if scenes_with_no_good_frames:
                print(f"Excluded scenes with no good frames: {scenes_with_no_good_frames}")

    def sequence_has_good_frames(self, base_dir, sequence):
        # Check if the sequence has any good frames (frames not marked as bad)
        input_raw_folder = osp.join(base_dir, 'data', sequence)
        input_processed_folder = osp.join(base_dir, 'data', sequence)

        # Load Train & Test Splits
        frame_file = osp.join(input_raw_folder, "dslr", "train_test_lists.json")
        with open(frame_file, "r") as f:
            train_test_list = json.load(f)

        # Load the nerfstudio/transforms.json file to check whether each image is blurry
        nerfstudio_transforms_path = osp.join(input_raw_folder, 'dslr', 'nerfstudio', 'transforms.json')
        with open(nerfstudio_transforms_path, "r") as f:
            nerfstudio_transforms = json.load(f)

        # Create a mapping from image name to the 'is_bad' flag
        file_path_to_is_bad = {}
        for frame in nerfstudio_transforms["frames"]:
            file_path_to_is_bad[frame["file_path"]] = frame.get("is_bad", False)

        # Check if there is at least one good frame in the train split
        for train_file_name in train_test_list["train"]:
            is_bad = file_path_to_is_bad.get(train_file_name, False)
            if not is_bad:
                return True  # Found at least one good frame

        return False  # No good frames found

    def load_coverage_data(self):
        for sequence in self.scene_list:
            coverage_file = osp.join(self.ROOT, 'coverage', f'{sequence}.json')
            if not osp.exists(coverage_file):
                raise FileNotFoundError(f"Coverage file {coverage_file} not found")
            with open(coverage_file, 'r') as f:
                sequence_coverage = json.load(f)
            self.coverage[sequence] = sequence_coverage[sequence]

    def _get_views(self, idx, resolution, rng, attempts=0):
        scene_id = self.scene_list[idx // self.num_seq]

        # Load camera metadata
        cams_metadata_path = osp.join(self.ROOT, 'data', scene_id, 'dslr', 'nerfstudio', 'transforms_undistorted.json')
        with open(cams_metadata_path, "r") as f:
            cams_meta_data = json.load(f)
        fx, fy, cx, cy = cams_meta_data['fl_x'], cams_meta_data['fl_y'], cams_meta_data['cx'], cams_meta_data['cy']

        frame_meta_data = cams_meta_data['frames']

        # Load the nerfstudio/transforms.json file to check for 'is_bad' frames
        nerfstudio_transforms_path = osp.join(self.ROOT, 'data', scene_id, 'dslr', 'nerfstudio', 'transforms.json')
        with open(nerfstudio_transforms_path, "r") as f:
            nerfstudio_transforms = json.load(f)

        # Create a mapping from image name to the 'is_bad' flag
        file_path_to_is_bad = {}
        for frame in nerfstudio_transforms["frames"]:
            file_path_to_is_bad[frame["file_path"]] = frame.get("is_bad", False)

        train_info_path = osp.join(self.ROOT, 'data', scene_id, 'dslr', 'train_test_lists.json')
        with open(train_info_path, "r") as f:
            train_info = json.load(f)

        # Filter out bad frames
        imgs_idxs_ = [
            img_idx for img_idx in sorted(train_info['train'])
            if not file_path_to_is_bad.get(img_idx, False)
        ]

        if len(imgs_idxs_) < self.num_frames:
            # Not enough good frames, try a different scene
            if attempts >= 5:
                new_idx = rng.integers(0, self.__len__() - 1)
                return self._get_views(new_idx, resolution, rng)
            return self._get_views(idx, resolution, rng, attempts=attempts + 1)

        imgs_idxs = self.sample_frame_idx(imgs_idxs_, scene_id, rng, full_video=self.full_video)
        imgs_idxs = deque(imgs_idxs)

        filepath_index_mapping = {frame["file_path"]: index for index, frame in enumerate(frame_meta_data)}
        P = np.array([
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1]]
        ).astype(np.float32)

        views = []
        while len(imgs_idxs) > 0:
            im_idx = imgs_idxs.popleft()

            # Load image data
            impath = osp.join(self.ROOT, 'data', scene_id, 'dslr', 'undistorted_images', im_idx)
            depthpath = osp.join(self.ROOT, 'data', scene_id, 'dslr', 'undistorted_depths', im_idx.replace('.JPG', '.png'))

            rgb_image = imread_cv2(impath)
            depthmap = imread_cv2(depthpath, cv2.IMREAD_UNCHANGED)
            depthmap = np.nan_to_num(depthmap.astype(np.float32), 0.0) / 1000.0  # Convert depth to meters

            # Load camera parameters
            frame_metadata = frame_meta_data[filepath_index_mapping.get(im_idx)]
            intrinsics = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

            camera_pose = np.array(frame_metadata["transform_matrix"], dtype=np.float32)
            # Convert from OpenGL to OpenCV coordinate system
            
            camera_pose = P @ camera_pose @ P.T

            # camera_pose = orthonormalize_rotation_np(camera_pose)

            # Perform cropping and resizing
            rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
                rgb_image, depthmap, intrinsics, resolution, rng=rng, info=impath)

            num_valid = (depthmap > 0.0).sum()
            if num_valid == 0 or (not np.isfinite(camera_pose).all()):
                if self.full_video:
                    print(f"Warning: No valid depthmap found for {impath}")
                    continue
                else:
                    if attempts >= 5:
                        new_idx = rng.integers(0, self.__len__() - 1)
                        return self._get_views(new_idx, resolution, rng)
                    return self._get_views(idx, resolution, rng, attempts=attempts + 1)

            views.append(dict(
                img=rgb_image,
                depthmap=depthmap,
                camera_pose=camera_pose,
                camera_intrinsics=intrinsics,
                dataset='scannetpp',
                label=osp.join(scene_id, im_idx),
                instance=osp.split(impath)[1],
            ))

        return views
