"""
Traditional descriptor extractors (SIFT, ORB, BRISK).
"""

from typing import Optional
import numpy as np
import cv2


class TraditionalExtractor:
    """Extract traditional descriptors (SIFT, ORB, BRISK)."""
    
    def __init__(self, method: str):
        self.method = method.lower()
        
        if self.method == "sift":
            self.extractor = cv2.SIFT_create() # type: ignore
            self.use_l2 = True
        elif self.method == "orb":
            self.extractor = cv2.ORB_create() # type: ignore
            self.use_l2 = False
        elif self.method == "brisk":
            self.extractor = cv2.BRISK_create() # type: ignore
            self.use_l2 = False
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def extract(self, patch: np.ndarray) -> Optional[np.ndarray]:
        if patch.max() <= 1.0:
            patch = (patch * 255).astype(np.uint8)
        else:
            patch = patch.astype(np.uint8)
        
        h, w = patch.shape[:2]
        if w < 64 or h < 64:
            patch = cv2.resize(patch, (64, 64), interpolation=cv2.INTER_LINEAR)
        
        h, w = patch.shape[:2]
        kp = cv2.KeyPoint(w / 2, h / 2, 16)
        
        try:
            _, desc = self.extractor.compute(patch, [kp])
            if desc is not None and len(desc) > 0:
                return desc[0]
        except Exception:
            pass
        return None
    
    def distance(self, d1: np.ndarray, d2: np.ndarray) -> float:
        if self.use_l2:
            d1 = d1.astype(np.float32)
            d2 = d2.astype(np.float32)
            d1 /= (np.linalg.norm(d1) + 1e-8)
            d2 /= (np.linalg.norm(d2) + 1e-8)
            return float(1 - np.dot(d1, d2))
        else:
            return float(cv2.norm(d1, d2, cv2.NORM_HAMMING))
