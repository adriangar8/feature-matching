"""
Traditional Feature Matching Baselines

Implements classical keypoint detection and matching methods:
- SIFT + BFMatcher / FLANN
- ORB + BFMatcher / FLANN  
- BRISK + BFMatcher
- AKAZE + BFMatcher
- (SURF if available - patented, may not be in all OpenCV builds)

Each method provides:
- detect_and_compute: Detect keypoints and compute descriptors
- match: Match descriptors between two images
- evaluate_pair: Full pipeline for a single image pair
"""

import cv2
import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, List, Optional, Dict
from dataclasses import dataclass


@dataclass
class MatchingResult:
    """Result of matching two images."""
    keypoints1: List[cv2.KeyPoint]
    keypoints2: List[cv2.KeyPoint]
    matches: List[cv2.DMatch]
    inliers: Optional[np.ndarray] = None
    homography: Optional[np.ndarray] = None
    num_matches: int = 0
    num_inliers: int = 0
    inlier_ratio: float = 0.0
    match_time_ms: float = 0.0


class BaseFeatureMatcher(ABC):
    """Abstract base class for feature matchers."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def detect_and_compute(
        self, img: np.ndarray
    ) -> Tuple[List[cv2.KeyPoint], Optional[np.ndarray]]:
        """Detect keypoints and compute descriptors."""
        pass

    @abstractmethod
    def match(
        self,
        desc1: np.ndarray,
        desc2: np.ndarray,
        ratio_thresh: float = 0.75,
    ) -> List[cv2.DMatch]:
        """Match descriptors using ratio test."""
        pass

    def evaluate_pair(
        self,
        img1: np.ndarray,
        img2: np.ndarray,
        H_gt: Optional[np.ndarray] = None,
        ratio_thresh: float = 0.75,
        reproj_thresh: float = 3.0,
    ) -> MatchingResult:
        """
        Full matching pipeline for an image pair.
        
        Args:
            img1: First image (grayscale)
            img2: Second image (grayscale)
            H_gt: Ground truth homography (optional, for evaluation)
            ratio_thresh: Lowe's ratio test threshold
            reproj_thresh: RANSAC reprojection threshold
            
        Returns:
            MatchingResult with matches, inliers, and metrics
        """
        import time

        start = time.time()

        # Detect and compute
        kp1, desc1 = self.detect_and_compute(img1)
        kp2, desc2 = self.detect_and_compute(img2)

        if desc1 is None or desc2 is None or len(desc1) < 4 or len(desc2) < 4:
            return MatchingResult(
                keypoints1=kp1,
                keypoints2=kp2,
                matches=[],
                match_time_ms=(time.time() - start) * 1000,
            )

        # Match
        matches = self.match(desc1, desc2, ratio_thresh)

        elapsed = (time.time() - start) * 1000

        if len(matches) < 4:
            return MatchingResult(
                keypoints1=kp1,
                keypoints2=kp2,
                matches=matches,
                num_matches=len(matches),
                match_time_ms=elapsed,
            )

        # Extract matched points
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

        # Estimate homography with RANSAC
        H_est, inliers = cv2.findHomography(pts1, pts2, cv2.RANSAC, reproj_thresh)

        if inliers is None:
            inliers = np.zeros(len(matches), dtype=np.uint8)

        inliers = inliers.ravel()
        num_inliers = int(np.sum(inliers))

        return MatchingResult(
            keypoints1=kp1,
            keypoints2=kp2,
            matches=matches,
            inliers=inliers,
            homography=H_est,
            num_matches=len(matches),
            num_inliers=num_inliers,
            inlier_ratio=num_inliers / len(matches) if len(matches) > 0 else 0.0,
            match_time_ms=elapsed,
        )


class SIFTMatcher(BaseFeatureMatcher):
    """SIFT detector with BFMatcher or FLANN."""

    def __init__(self, use_flann: bool = False, nfeatures: int = 0):
        super().__init__("SIFT" + ("_FLANN" if use_flann else "_BF"))
        self.detector = cv2.SIFT_create(nfeatures=nfeatures)
        self.use_flann = use_flann

        if use_flann:
            index_params = dict(algorithm=1, trees=5)  # FLANN_INDEX_KDTREE
            search_params = dict(checks=50)
            self.matcher = cv2.FlannBasedMatcher(index_params, search_params)
        else:
            self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

    def detect_and_compute(
        self, img: np.ndarray
    ) -> Tuple[List[cv2.KeyPoint], Optional[np.ndarray]]:
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return self.detector.detectAndCompute(img, None)

    def match(
        self,
        desc1: np.ndarray,
        desc2: np.ndarray,
        ratio_thresh: float = 0.75,
    ) -> List[cv2.DMatch]:
        if desc1 is None or desc2 is None:
            return []

        matches = self.matcher.knnMatch(desc1, desc2, k=2)

        # Lowe's ratio test
        good = []
        for m_n in matches:
            if len(m_n) == 2:
                m, n = m_n
                if m.distance < ratio_thresh * n.distance:
                    good.append(m)

        return good


class ORBMatcher(BaseFeatureMatcher):
    """ORB detector with BFMatcher or FLANN."""

    def __init__(self, use_flann: bool = False, nfeatures: int = 500):
        super().__init__("ORB" + ("_FLANN" if use_flann else "_BF"))
        self.detector = cv2.ORB_create(nfeatures=nfeatures)
        self.use_flann = use_flann

        if use_flann:
            index_params = dict(
                algorithm=6,  # FLANN_INDEX_LSH
                table_number=6,
                key_size=12,
                multi_probe_level=1,
            )
            search_params = dict(checks=50)
            self.matcher = cv2.FlannBasedMatcher(index_params, search_params)
        else:
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    def detect_and_compute(
        self, img: np.ndarray
    ) -> Tuple[List[cv2.KeyPoint], Optional[np.ndarray]]:
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return self.detector.detectAndCompute(img, None)

    def match(
        self,
        desc1: np.ndarray,
        desc2: np.ndarray,
        ratio_thresh: float = 0.75,
    ) -> List[cv2.DMatch]:
        if desc1 is None or desc2 is None:
            return []

        try:
            matches = self.matcher.knnMatch(desc1, desc2, k=2)
        except cv2.error:
            return []

        good = []
        for m_n in matches:
            if len(m_n) == 2:
                m, n = m_n
                if m.distance < ratio_thresh * n.distance:
                    good.append(m)

        return good


class BRISKMatcher(BaseFeatureMatcher):
    """BRISK detector with BFMatcher."""

    def __init__(self, thresh: int = 30, octaves: int = 3):
        super().__init__("BRISK_BF")
        self.detector = cv2.BRISK_create(thresh=thresh, octaves=octaves)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    def detect_and_compute(
        self, img: np.ndarray
    ) -> Tuple[List[cv2.KeyPoint], Optional[np.ndarray]]:
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return self.detector.detectAndCompute(img, None)

    def match(
        self,
        desc1: np.ndarray,
        desc2: np.ndarray,
        ratio_thresh: float = 0.75,
    ) -> List[cv2.DMatch]:
        if desc1 is None or desc2 is None:
            return []

        try:
            matches = self.matcher.knnMatch(desc1, desc2, k=2)
        except cv2.error:
            return []

        good = []
        for m_n in matches:
            if len(m_n) == 2:
                m, n = m_n
                if m.distance < ratio_thresh * n.distance:
                    good.append(m)

        return good


class AKAZEMatcher(BaseFeatureMatcher):
    """AKAZE detector with BFMatcher."""

    def __init__(self, threshold: float = 0.001):
        super().__init__("AKAZE_BF")
        self.detector = cv2.AKAZE_create(threshold=threshold)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    def detect_and_compute(
        self, img: np.ndarray
    ) -> Tuple[List[cv2.KeyPoint], Optional[np.ndarray]]:
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return self.detector.detectAndCompute(img, None)

    def match(
        self,
        desc1: np.ndarray,
        desc2: np.ndarray,
        ratio_thresh: float = 0.75,
    ) -> List[cv2.DMatch]:
        if desc1 is None or desc2 is None:
            return []

        try:
            matches = self.matcher.knnMatch(desc1, desc2, k=2)
        except cv2.error:
            return []

        good = []
        for m_n in matches:
            if len(m_n) == 2:
                m, n = m_n
                if m.distance < ratio_thresh * n.distance:
                    good.append(m)

        return good


def get_all_matchers() -> Dict[str, BaseFeatureMatcher]:
    """Get all available matchers."""
    matchers = {
        "SIFT_BF": SIFTMatcher(use_flann=False),
        "SIFT_FLANN": SIFTMatcher(use_flann=True),
        "ORB_BF": ORBMatcher(use_flann=False, nfeatures=1000),
        "ORB_FLANN": ORBMatcher(use_flann=True, nfeatures=1000),
        "BRISK_BF": BRISKMatcher(),
        "AKAZE_BF": AKAZEMatcher(),
    }

    # Try to add SURF if available (patented, not always present)
    try:
        surf = cv2.xfeatures2d.SURF_create()
        matchers["SURF_BF"] = _SURFMatcher(use_flann=False)
        matchers["SURF_FLANN"] = _SURFMatcher(use_flann=True)
    except AttributeError:
        pass  # SURF not available

    return matchers


class _SURFMatcher(BaseFeatureMatcher):
    """SURF detector (if available)."""

    def __init__(self, use_flann: bool = False, hessian_threshold: float = 400):
        super().__init__("SURF" + ("_FLANN" if use_flann else "_BF"))
        self.detector = cv2.xfeatures2d.SURF_create(hessianThreshold=hessian_threshold)
        self.use_flann = use_flann

        if use_flann:
            index_params = dict(algorithm=1, trees=5)
            search_params = dict(checks=50)
            self.matcher = cv2.FlannBasedMatcher(index_params, search_params)
        else:
            self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

    def detect_and_compute(
        self, img: np.ndarray
    ) -> Tuple[List[cv2.KeyPoint], Optional[np.ndarray]]:
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return self.detector.detectAndCompute(img, None)

    def match(
        self,
        desc1: np.ndarray,
        desc2: np.ndarray,
        ratio_thresh: float = 0.75,
    ) -> List[cv2.DMatch]:
        if desc1 is None or desc2 is None:
            return []

        matches = self.matcher.knnMatch(desc1, desc2, k=2)

        good = []
        for m_n in matches:
            if len(m_n) == 2:
                m, n = m_n
                if m.distance < ratio_thresh * n.distance:
                    good.append(m)

        return good
