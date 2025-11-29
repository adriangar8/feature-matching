"""
Deep Learning Feature Matcher

Wraps learned descriptors (ResNet, etc.) into the same interface as traditional matchers.
Allows fair comparison on full image matching tasks.
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple, List, Optional
from dataclasses import dataclass

from src.baselines.traditional_matchers import BaseFeatureMatcher, MatchingResult


class DeepFeatureMatcher(BaseFeatureMatcher):
    """
    Deep learning based feature matcher.
    
    Uses a keypoint detector (SIFT/ORB) for detection,
    but a learned CNN for descriptor computation.
    
    Args:
        model: PyTorch model that takes (B, 3, H, W) patches and returns (B, D) descriptors
        device: torch device
        detector: Keypoint detector ("sift" or "orb")
        patch_size: Size of patches to extract around keypoints
    """

    def __init__(
        self,
        model: torch.nn.Module,
        device: str = "cuda",
        detector: str = "sift",
        patch_size: int = 32,
        name: str = "DeepMatcher",
    ):
        super().__init__(name)
        self.model = model
        self.device = device
        self.patch_size = patch_size
        self.model.eval()

        if detector == "sift":
            self.detector_fn = cv2.SIFT_create()
        elif detector == "orb":
            self.detector_fn = cv2.ORB_create(nfeatures=1000)
        else:
            raise ValueError(f"Unknown detector: {detector}")

        self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

    def _extract_patches(
        self, img: np.ndarray, keypoints: List[cv2.KeyPoint]
    ) -> Tuple[torch.Tensor, List[int]]:
        """Extract patches around keypoints."""
        half = self.patch_size // 2
        h, w = img.shape[:2]

        patches = []
        valid_indices = []

        # Convert to RGB if grayscale
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        for i, kp in enumerate(keypoints):
            x, y = int(kp.pt[0]), int(kp.pt[1])

            if x - half < 0 or x + half >= w or y - half < 0 or y + half >= h:
                continue

            patch = img[y - half : y + half, x - half : x + half]
            patches.append(patch)
            valid_indices.append(i)

        if not patches:
            return torch.empty(0), []

        # Convert to tensor (B, C, H, W)
        patches = np.stack(patches)
        patches = torch.from_numpy(patches).float().permute(0, 3, 1, 2) / 255.0

        return patches, valid_indices

    def detect_and_compute(
        self, img: np.ndarray
    ) -> Tuple[List[cv2.KeyPoint], Optional[np.ndarray]]:
        """Detect keypoints and compute learned descriptors."""
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        # Detect keypoints
        keypoints = self.detector_fn.detect(gray, None)

        if len(keypoints) == 0:
            return [], None

        # Extract patches
        patches, valid_indices = self._extract_patches(img, keypoints)

        if len(valid_indices) == 0:
            return [], None

        # Compute descriptors
        with torch.no_grad():
            patches = patches.to(self.device)
            descriptors = self.model(patches).cpu().numpy()

        # Filter keypoints to valid ones
        valid_keypoints = [keypoints[i] for i in valid_indices]

        return valid_keypoints, descriptors

    def match(
        self,
        desc1: np.ndarray,
        desc2: np.ndarray,
        ratio_thresh: float = 0.75,
    ) -> List[cv2.DMatch]:
        """Match descriptors using ratio test."""
        if desc1 is None or desc2 is None or len(desc1) < 2 or len(desc2) < 2:
            return []

        # Ensure float32
        desc1 = desc1.astype(np.float32)
        desc2 = desc2.astype(np.float32)

        matches = self.matcher.knnMatch(desc1, desc2, k=2)

        good = []
        for m_n in matches:
            if len(m_n) == 2:
                m, n = m_n
                if m.distance < ratio_thresh * n.distance:
                    good.append(m)

        return good


class DeepPatchMatcher:
    """
    Matcher for pre-extracted patches (used in triplet evaluation).
    
    Different from DeepFeatureMatcher which operates on full images.
    """

    def __init__(self, model: torch.nn.Module, device: str = "cuda"):
        self.model = model
        self.device = device
        self.model.eval()

    def compute_descriptor(self, patch: torch.Tensor) -> torch.Tensor:
        """Compute descriptor for a single patch."""
        with torch.no_grad():
            if patch.dim() == 3:
                patch = patch.unsqueeze(0)
            patch = patch.to(self.device)
            return self.model(patch)

    def compute_distance(
        self, desc1: torch.Tensor, desc2: torch.Tensor
    ) -> torch.Tensor:
        """Compute L2 distance between descriptors."""
        return torch.norm(desc1 - desc2, dim=-1)

    def match_batch(
        self,
        anchors: torch.Tensor,
        positives: torch.Tensor,
        negatives: torch.Tensor,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Match a batch of triplets.
        
        Returns:
            pos_dists: Distances to positive patches
            neg_dists: Distances to negative patches  
            accuracy: Fraction where pos_dist < neg_dist
        """
        with torch.no_grad():
            anchors = anchors.to(self.device)
            positives = positives.to(self.device)
            negatives = negatives.to(self.device)

            da = self.model(anchors)
            dp = self.model(positives)
            dn = self.model(negatives)

            pos_dists = torch.norm(da - dp, dim=1).cpu().numpy()
            neg_dists = torch.norm(da - dn, dim=1).cpu().numpy()

            accuracy = (pos_dists < neg_dists).mean()

        return pos_dists, neg_dists, accuracy
