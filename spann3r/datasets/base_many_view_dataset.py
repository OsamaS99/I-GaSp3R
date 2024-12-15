from dust3r.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset
import numpy as np


class BaseManyViewDataset(BaseStereoViewDataset):
    def __init__(self, *args, num_frames=6, context_overlap_threshold=0.7, target_overlap_threshold=0.7, **kwargs):
        self.num_frames = num_frames  # Total number of frames including targets and context
        self.num_context_frames = self.num_frames // 2  # Subtracting two target frames
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
        
        # Set thresholds (can be the same or different for context/target)
        self.context_overlap_threshold = self.train_ratio
        self.target_overlap_threshold = self.train_ratio
        
        # Determine how many context and target frames we need
        num_context = self.num_frames // 2
        num_target = self.num_frames - num_context
        
        img_indices = list(range(len(img_idxs)))
        coverage = self.coverage[sequence]
        
        # Step 1: Select context frames
        # Start with a random context frame
        if not img_indices:
            raise ValueError("No frames available in the sequence.")
        context_frames = [rng.choice(img_indices)]
        
        # For the remaining context frames, pick those that have enough overlap with at least one chosen context frame
        for _ in range(num_context - 1):
            # Find valid context candidates
            valid_context_candidates = []
            for idx in img_indices:
                if idx in context_frames:
                    continue
                
                # Check if this candidate overlaps well with at least one of the already chosen context frames
                if any(coverage[cf][idx] > self.context_overlap_threshold for cf in context_frames):
                    valid_context_candidates.append(idx)
            
            if valid_context_candidates:
                next_context = rng.choice(valid_context_candidates)
            else:
                # If none meet the threshold, pick the best overlapping frame with respect to any context frame
                overlaps = []
                for idx in img_indices:
                    if idx in context_frames:
                        continue
                    best_overlap_with_context = max(coverage[cf][idx] for cf in context_frames)
                    overlaps.append((idx, best_overlap_with_context))
                if not overlaps:
                    # No more frames to choose from
                    if attempts >= max_attempts:
                        raise RuntimeError("Unable to find enough context frames.")
                    return self.sample_frames(img_idxs, sequence, rng, attempts=attempts+1)
                
                next_context = max(overlaps, key=lambda x: x[1])[0]
            
            context_frames.append(next_context)
        
        # Step 2: Select target frames
        # We want frames that overlap well with ANY of the context frames
        remaining_indices = [idx for idx in img_indices if idx not in context_frames]
        valid_target_candidates = []
        
        for idx in remaining_indices:
            # Check overlap with context frames
            best_overlap_with_context = max(coverage[cf][idx] for cf in context_frames)
            if best_overlap_with_context > self.target_overlap_threshold:
                valid_target_candidates.append(idx)
        
        if len(valid_target_candidates) >= num_target:
            # Randomly choose targets from the valid ones
            target_frames = rng.choice(valid_target_candidates, size=num_target, replace=False).tolist()
        else:
            # If not enough valid candidates, choose the best overlapping frames
            all_overlaps = []
            for idx in remaining_indices:
                best_overlap_with_context = max(coverage[cf][idx] for cf in context_frames)
                all_overlaps.append((idx, best_overlap_with_context))
            all_overlaps.sort(key=lambda x: x[1], reverse=True)
            
            # If there still aren't enough frames, retry
            if len(all_overlaps) < num_target:
                if attempts >= max_attempts:
                    raise RuntimeError("Unable to find enough target frames even after retries.")
                return self.sample_frames(img_idxs, sequence, rng, attempts=attempts+1)
            
            target_frames = [frame for frame, _ in all_overlaps[:num_target]]
        
        # Combine all selected frames: first half is context, second half is target
        selected_indices = context_frames + target_frames
        
        # Just a sanity check that we got the correct count
        if len(selected_indices) < self.num_frames:
            if attempts >= max_attempts:
                raise RuntimeError("Not enough frames selected to meet the requirement.")
            return self.sample_frames(img_idxs, sequence, rng, attempts=attempts+1)
        
        # Map indices back to image file names
        selected_img_ids = [img_idxs[i] for i in selected_indices]

        return selected_img_ids
