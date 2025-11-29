"""
Comprehensive Evaluation Metrics

Includes:
- Triplet accuracy (descriptor learning)
- Mean Matching Accuracy (MMA) at various thresholds
- Homography estimation accuracy
- Forgetting metrics for continual learning
- Per-domain evaluation
"""

import torch
import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from torch.utils.data import DataLoader

from src.datasets.hpatches_loader import warp_points


@dataclass
class TripletMetrics:
    """Metrics from triplet evaluation."""
    accuracy: float
    pos_dists: np.ndarray
    neg_dists: np.ndarray
    margin_violation_rate: float = 0.0


@dataclass
class MatchingMetrics:
    """Metrics from image pair matching."""
    num_matches: float
    num_inliers: float
    inlier_ratio: float
    match_time_ms: float


@dataclass
class HomographyMetrics:
    """Metrics from homography estimation."""
    correctness_1px: float  # % pairs with corner error < 1px
    correctness_3px: float  # % pairs with corner error < 3px
    correctness_5px: float  # % pairs with corner error < 5px
    mean_corner_error: float
    median_corner_error: float


@dataclass
class MMAMetrics:
    """Mean Matching Accuracy at various thresholds."""
    mma_1px: float
    mma_3px: float
    mma_5px: float
    mma_10px: float


@dataclass
class ForgettingMetrics:
    """Metrics for measuring catastrophic forgetting."""
    initial_accuracy: float
    final_accuracy: float
    forgetting: float  # initial - final
    forgetting_rate: float  # forgetting / initial
    backward_transfer: float  # negative = forgetting


@dataclass
class FullEvaluationResult:
    """Complete evaluation results."""
    triplet: Optional[TripletMetrics] = None
    matching: Optional[MatchingMetrics] = None
    homography: Optional[HomographyMetrics] = None
    mma: Optional[MMAMetrics] = None
    forgetting: Optional[ForgettingMetrics] = None
    per_sequence: Dict[str, dict] = field(default_factory=dict)


# =============================================================================
# TRIPLET EVALUATION
# =============================================================================

def triplet_accuracy(
    model: torch.nn.Module,
    loader: DataLoader,
    device: str = "cuda",
    margin: float = 1.0,
) -> TripletMetrics:
    """
    Evaluate triplet accuracy.
    
    Measures what fraction of triplets satisfy d(a,p) < d(a,n).
    """
    model.eval()
    
    pos_dists = []
    neg_dists = []
    correct = 0
    total = 0
    margin_violations = 0

    with torch.no_grad():
        for a, p, n in loader:
            a, p, n = a.to(device), p.to(device), n.to(device)

            da = model(a)
            dp = model(p)
            dn = model(n)

            dp_dist = torch.norm(da - dp, dim=1).cpu().numpy()
            dn_dist = torch.norm(da - dn, dim=1).cpu().numpy()

            pos_dists.extend(dp_dist.tolist())
            neg_dists.extend(dn_dist.tolist())

            correct += (dp_dist < dn_dist).sum()
            margin_violations += ((dp_dist + margin) > dn_dist).sum()
            total += len(dp_dist)

    accuracy = correct / total if total > 0 else 0
    margin_violation_rate = margin_violations / total if total > 0 else 0

    return TripletMetrics(
        accuracy=accuracy,
        pos_dists=np.array(pos_dists),
        neg_dists=np.array(neg_dists),
        margin_violation_rate=margin_violation_rate,
    )


# =============================================================================
# HOMOGRAPHY ESTIMATION
# =============================================================================

def compute_corner_error(
    H_est: np.ndarray,
    H_gt: np.ndarray,
    img_shape: Tuple[int, int],
) -> float:
    """
    Compute mean corner error between estimated and ground truth homography.
    
    Projects the 4 corners of the image using both homographies
    and computes the mean distance.
    """
    h, w = img_shape
    corners = np.array([
        [0, 0],
        [w, 0],
        [w, h],
        [0, h],
    ], dtype=np.float32)

    corners_est = warp_points(H_est, corners)
    corners_gt = warp_points(H_gt, corners)

    errors = np.linalg.norm(corners_est - corners_gt, axis=1)
    return errors.mean()


def homography_accuracy(
    results: List[Tuple[np.ndarray, np.ndarray, Tuple[int, int]]],
    thresholds: List[float] = [1, 3, 5],
) -> HomographyMetrics:
    """
    Compute homography estimation accuracy.
    
    Args:
        results: List of (H_est, H_gt, img_shape) tuples
        thresholds: Corner error thresholds in pixels
        
    Returns:
        HomographyMetrics with accuracy at each threshold
    """
    errors = []

    for H_est, H_gt, img_shape in results:
        if H_est is None:
            errors.append(float('inf'))
        else:
            errors.append(compute_corner_error(H_est, H_gt, img_shape))

    errors = np.array(errors)
    valid_errors = errors[errors != float('inf')]

    if len(valid_errors) == 0:
        return HomographyMetrics(
            correctness_1px=0,
            correctness_3px=0,
            correctness_5px=0,
            mean_corner_error=float('inf'),
            median_corner_error=float('inf'),
        )

    return HomographyMetrics(
        correctness_1px=(errors < 1).mean(),
        correctness_3px=(errors < 3).mean(),
        correctness_5px=(errors < 5).mean(),
        mean_corner_error=valid_errors.mean(),
        median_corner_error=np.median(valid_errors),
    )


# =============================================================================
# MEAN MATCHING ACCURACY (MMA)
# =============================================================================

def compute_mma(
    keypoints1: List[cv2.KeyPoint],
    keypoints2: List[cv2.KeyPoint],
    matches: List[cv2.DMatch],
    H_gt: np.ndarray,
    thresholds: List[float] = [1, 3, 5, 10],
) -> Dict[str, float]:
    """
    Compute Mean Matching Accuracy at various thresholds.
    
    For each match, compute the reprojection error using ground truth H.
    MMA@t = fraction of matches with error < t pixels.
    """
    if len(matches) == 0:
        return {f"mma_{t}px": 0.0 for t in thresholds}

    pts1 = np.array([keypoints1[m.queryIdx].pt for m in matches])
    pts2 = np.array([keypoints2[m.trainIdx].pt for m in matches])

    # Warp pts1 using ground truth homography
    pts1_warped = warp_points(H_gt, pts1)

    # Compute reprojection errors
    errors = np.linalg.norm(pts1_warped - pts2, axis=1)

    results = {}
    for t in thresholds:
        results[f"mma_{t}px"] = (errors < t).mean()

    return results


# =============================================================================
# FORGETTING METRICS
# =============================================================================

def compute_forgetting(
    accuracy_before: float,
    accuracy_after: float,
) -> ForgettingMetrics:
    """
    Compute forgetting metrics.
    
    Args:
        accuracy_before: Accuracy on domain A before training on domain B
        accuracy_after: Accuracy on domain A after training on domain B
    """
    forgetting = accuracy_before - accuracy_after
    forgetting_rate = forgetting / accuracy_before if accuracy_before > 0 else 0

    return ForgettingMetrics(
        initial_accuracy=accuracy_before,
        final_accuracy=accuracy_after,
        forgetting=forgetting,
        forgetting_rate=forgetting_rate,
        backward_transfer=-forgetting,  # Negative = forgetting
    )


# =============================================================================
# TRADITIONAL MATCHER EVALUATION
# =============================================================================

def evaluate_traditional_matcher(
    matcher,  # BaseFeatureMatcher
    sequence_loader: DataLoader,
) -> Dict[str, any]:
    """
    Evaluate a traditional matcher on HPatches sequences.
    
    Returns comprehensive metrics including MMA and homography accuracy.
    """
    all_results = []
    homography_data = []
    matching_stats = []
    mma_scores = {f"mma_{t}px": [] for t in [1, 3, 5, 10]}

    for img1, img2, H_gt, seq in sequence_loader:
        # DataLoader returns batched tensors, extract single items
        img1 = img1[0].numpy() if torch.is_tensor(img1) else img1
        img2 = img2[0].numpy() if torch.is_tensor(img2) else img2
        H_gt = H_gt[0].numpy() if torch.is_tensor(H_gt) else H_gt

        result = matcher.evaluate_pair(img1, img2, H_gt)

        matching_stats.append({
            "num_matches": result.num_matches,
            "num_inliers": result.num_inliers,
            "inlier_ratio": result.inlier_ratio,
            "match_time_ms": result.match_time_ms,
        })

        if result.homography is not None:
            homography_data.append((result.homography, H_gt, img1.shape[:2]))

        # Compute MMA
        if result.num_matches > 0:
            mma = compute_mma(
                result.keypoints1,
                result.keypoints2,
                result.matches,
                H_gt,
            )
            for k, v in mma.items():
                mma_scores[k].append(v)

    # Aggregate metrics
    matching_metrics = MatchingMetrics(
        num_matches=np.mean([s["num_matches"] for s in matching_stats]),
        num_inliers=np.mean([s["num_inliers"] for s in matching_stats]),
        inlier_ratio=np.mean([s["inlier_ratio"] for s in matching_stats]),
        match_time_ms=np.mean([s["match_time_ms"] for s in matching_stats]),
    )

    homography_metrics = homography_accuracy(homography_data) if homography_data else None

    mma_metrics = MMAMetrics(
        mma_1px=np.mean(mma_scores["mma_1px"]) if mma_scores["mma_1px"] else 0,
        mma_3px=np.mean(mma_scores["mma_3px"]) if mma_scores["mma_3px"] else 0,
        mma_5px=np.mean(mma_scores["mma_5px"]) if mma_scores["mma_5px"] else 0,
        mma_10px=np.mean(mma_scores["mma_10px"]) if mma_scores["mma_10px"] else 0,
    )

    return {
        "matching": matching_metrics,
        "homography": homography_metrics,
        "mma": mma_metrics,
    }


# =============================================================================
# DEEP MATCHER EVALUATION
# =============================================================================

def evaluate_deep_matcher(
    matcher,  # DeepFeatureMatcher
    triplet_loader: DataLoader,
    sequence_loader: DataLoader,
    device: str = "cuda",
) -> FullEvaluationResult:
    """
    Full evaluation of a deep learning matcher.
    
    Includes triplet accuracy, MMA, and homography estimation.
    """
    # Triplet evaluation
    triplet_metrics = triplet_accuracy(matcher.model, triplet_loader, device)

    # Sequence-level evaluation
    seq_results = evaluate_traditional_matcher(matcher, sequence_loader)

    return FullEvaluationResult(
        triplet=triplet_metrics,
        matching=seq_results["matching"],
        homography=seq_results["homography"],
        mma=seq_results["mma"],
    )


# =============================================================================
# DOMAIN TRANSFER EVALUATION
# =============================================================================

def evaluate_domain_transfer(
    model: torch.nn.Module,
    source_loader: DataLoader,
    target_loader: DataLoader,
    device: str = "cuda",
) -> Dict[str, TripletMetrics]:
    """
    Evaluate model on both source and target domains.
    
    Useful for measuring how well a model transfers between
    illumination and viewpoint domains.
    """
    source_metrics = triplet_accuracy(model, source_loader, device)
    target_metrics = triplet_accuracy(model, target_loader, device)

    return {
        "source": source_metrics,
        "target": target_metrics,
        "transfer_gap": source_metrics.accuracy - target_metrics.accuracy,
    }
