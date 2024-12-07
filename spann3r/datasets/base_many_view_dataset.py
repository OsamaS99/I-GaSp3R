from dust3r.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset
import numpy as np


class BaseManyViewDataset(BaseStereoViewDataset):
    def __init__(self, *args, num_frames=5, context_overlap_threshold=0.8, target_overlap_threshold=0.6, **kwargs):
        self.num_frames = num_frames  # Total number of frames including targets and context
        self.num_context_frames = self.num_frames - 2  # Subtracting two target frames
        self.context_overlap_threshold = context_overlap_threshold
        self.target_overlap_threshold = target_overlap_threshold
        super().__init__(*args, num_views=self.num_frames, **kwargs)

    def sample_frame_idx(self, img_idxs, sequence, rng, full_video=False):
        if not full_video:
            img_idxs = self.sample_frames(img_idxs, sequence, rng)
        else:
            img_idxs = img_idxs[::self.kf_every]
        return img_idxs

    def sample_frames(self, img_idxs, sequence, rng, attempts=0):
        max_attempts = 5
        img_indices = list(range(len(img_idxs)))
        coverage = self.coverage[sequence]

        # Step 1: Select the first target frame
        first_target_idx = rng.choice(img_indices)
        selected_indices = [first_target_idx]

        # Step 2: Select context frames
        current_idx = first_target_idx
        for _ in range(self.num_context_frames):
            valid_context_frames = [
                idx for idx in img_indices
                if idx not in selected_indices and
                coverage[current_idx][idx] > self.context_overlap_threshold
            ]

            if valid_context_frames:
                next_idx = rng.choice(valid_context_frames)
            else:
                overlaps = [
                    (idx, coverage[current_idx][idx])
                    for idx in img_indices if idx not in selected_indices
                ]
                if not overlaps:
                    break
                next_idx = max(overlaps, key=lambda x: x[1])[0]

            selected_indices.append(next_idx)
            current_idx = next_idx

        # Step 3: Select the last target frame
        last_context_idx = selected_indices[-1] if len(selected_indices) > 1 else first_target_idx
        valid_target_frames = [
            idx for idx in img_indices
            if idx not in selected_indices and
            coverage[last_context_idx][idx] > self.target_overlap_threshold
        ]

        if valid_target_frames:
            last_target_idx = rng.choice(valid_target_frames)
        else:
            overlaps = [
                (idx, coverage[last_context_idx][idx])
                for idx in img_indices if idx not in selected_indices
            ]
            if not overlaps:
                remaining_indices = [idx for idx in img_indices if idx not in selected_indices]
                if not remaining_indices:
                    if attempts >= max_attempts:
                        raise RuntimeError("Unable to sample enough frames with sufficient overlap.")
                    return self.sample_frames(img_idxs, sequence, rng, attempts=attempts+1)
                last_target_idx = rng.choice(remaining_indices)
            else:
                last_target_idx = max(overlaps, key=lambda x: x[1])[0]

        selected_indices.append(last_target_idx)

        if len(selected_indices) < self.num_frames:
            if attempts >= max_attempts:
                raise RuntimeError("Unable to sample enough frames with sufficient overlap.")
            return self.sample_frames(img_idxs, sequence, rng, attempts=attempts+1)

        # Map indices back to image file names
        selected_img_ids = [img_idxs[i] for i in selected_indices]

        return selected_img_ids
