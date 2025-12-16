import random
import numpy as np
import cv2

from .constants import IMAGENET_MEAN, IMAGENET_STD


def normalize_patch(patch: np.ndarray) -> np.ndarray:
    if patch.max() > 1.0:
        patch = patch / 255.0
    rgb = np.stack([patch, patch, patch], axis=-1)
    rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
    return rgb.transpose(2, 0, 1).astype(np.float32)


def augment_patch(patch: np.ndarray) -> np.ndarray:
    if random.random() < 0.5:
        delta = random.uniform(-0.2, 0.2)
        patch = np.clip(patch + delta, 0, 1)
    
    if random.random() < 0.5:
        factor = random.uniform(0.7, 1.3)
        mean = patch.mean()
        patch = np.clip((patch - mean) * factor + mean, 0, 1)
    
    if random.random() < 0.3:
        noise = np.random.normal(0, 0.02, patch.shape).astype(np.float32)
        patch = np.clip(patch + noise, 0, 1)
    
    if random.random() < 0.3:
        angle = random.uniform(-15, 15)
        h, w = patch.shape
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
        patch = cv2.warpAffine(patch, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    
    if random.random() < 0.5:
        patch = np.fliplr(patch).copy()
    
    return patch
