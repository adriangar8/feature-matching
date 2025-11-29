"""
Visualization Utilities

Provides plotting functions for:
- Triplet embeddings (t-SNE)
- Distance distributions
- Matching visualizations
- Forgetting curves
- Comparison plots
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import torch
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple
import cv2

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def plot_triplet_embeddings(
    model: torch.nn.Module,
    loader: DataLoader,
    device: str = "cuda",
    max_samples: int = 500,
    save_path: Optional[str] = None,
    title: str = "t-SNE Embeddings",
) -> plt.Figure:
    """
    Visualize embeddings using t-SNE.
    
    Colors: 0=Anchor (blue), 1=Positive (green), 2=Negative (red)
    """
    model.eval()
    feats = []
    labels = []

    with torch.no_grad():
        for a, p, n in loader:
            a, p, n = a.to(device), p.to(device), n.to(device)
            
            fa = model(a).cpu().numpy()
            fp = model(p).cpu().numpy()
            fn = model(n).cpu().numpy()

            feats.extend([fa, fp, fn])
            labels.extend([0] * len(fa) + [1] * len(fp) + [2] * len(fn))

            if sum(len(f) for f in feats) >= max_samples * 3:
                break

    X = np.vstack(feats)
    L = np.array(labels)

    # Subsample if needed
    if len(X) > max_samples * 3:
        idx = np.random.choice(len(X), max_samples * 3, replace=False)
        X, L = X[idx], L[idx]

    # t-SNE
    perplexity = min(30, len(X) - 1)
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    X2 = tsne.fit_transform(X)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 8))
    
    colors = ['#1f77b4', '#2ca02c', '#d62728']
    labels_str = ['Anchor', 'Positive', 'Negative']
    
    for i, (color, label) in enumerate(zip(colors, labels_str)):
        mask = L == i
        ax.scatter(X2[mask, 0], X2[mask, 1], c=color, label=label, s=10, alpha=0.6)

    ax.legend()
    ax.set_title(title)
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_distance_distributions(
    pos_dists: np.ndarray,
    neg_dists: np.ndarray,
    save_path: Optional[str] = None,
    title: str = "Descriptor Distance Distribution",
) -> plt.Figure:
    """Plot histograms of positive and negative pair distances."""
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.hist(pos_dists, bins=50, alpha=0.6, label='Positive pairs', color='green', density=True)
    ax.hist(neg_dists, bins=50, alpha=0.6, label='Negative pairs', color='red', density=True)

    ax.axvline(np.median(pos_dists), color='darkgreen', linestyle='--', label=f'Pos median: {np.median(pos_dists):.2f}')
    ax.axvline(np.median(neg_dists), color='darkred', linestyle='--', label=f'Neg median: {np.median(neg_dists):.2f}')

    ax.set_xlabel('Distance')
    ax.set_ylabel('Density')
    ax.set_title(title)
    ax.legend()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_forgetting_curve(
    accuracies: Dict[str, List[float]],
    epochs: List[int],
    save_path: Optional[str] = None,
    title: str = "Forgetting Curve",
) -> plt.Figure:
    """
    Plot accuracy on source domain while training on target domain.
    
    Args:
        accuracies: Dict mapping method name to list of accuracies
        epochs: List of epoch numbers
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    for method, accs in accuracies.items():
        ax.plot(epochs, accs, marker='o', label=method, linewidth=2, markersize=6)

    ax.set_xlabel('Epochs on Target Domain')
    ax.set_ylabel('Accuracy on Source Domain')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_method_comparison(
    results: Dict[str, Dict[str, float]],
    metric: str = "accuracy",
    save_path: Optional[str] = None,
    title: str = "Method Comparison",
) -> plt.Figure:
    """
    Bar chart comparing different methods.
    
    Args:
        results: Dict mapping method name to dict of metrics
        metric: Which metric to plot
    """
    methods = list(results.keys())
    values = [results[m].get(metric, 0) for m in methods]

    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(methods, values, color=sns.color_palette("husl", len(methods)))

    # Add value labels on bars
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10)

    ax.set_xlabel('Method')
    ax.set_ylabel(metric.replace('_', ' ').title())
    ax.set_title(title)
    plt.xticks(rotation=45, ha='right')

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_mma_comparison(
    results: Dict[str, Dict[str, float]],
    save_path: Optional[str] = None,
    title: str = "Mean Matching Accuracy Comparison",
) -> plt.Figure:
    """
    Plot MMA at different thresholds for multiple methods.
    """
    thresholds = [1, 3, 5, 10]
    methods = list(results.keys())

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(thresholds))
    width = 0.8 / len(methods)

    for i, method in enumerate(methods):
        values = [results[method].get(f"mma_{t}px", 0) for t in thresholds]
        offset = (i - len(methods)/2 + 0.5) * width
        ax.bar(x + offset, values, width, label=method)

    ax.set_xlabel('Threshold (pixels)')
    ax.set_ylabel('MMA')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{t}px' for t in thresholds])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_matches(
    img1: np.ndarray,
    img2: np.ndarray,
    keypoints1: List[cv2.KeyPoint],
    keypoints2: List[cv2.KeyPoint],
    matches: List[cv2.DMatch],
    inliers: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
    title: str = "Feature Matches",
    max_matches: int = 100,
) -> plt.Figure:
    """
    Visualize feature matches between two images.
    
    Green = inliers, Red = outliers (if inliers provided)
    """
    # Subsample if too many matches
    if len(matches) > max_matches:
        idx = np.random.choice(len(matches), max_matches, replace=False)
        matches = [matches[i] for i in idx]
        if inliers is not None:
            inliers = inliers[idx]

    # Determine match colors
    if inliers is not None:
        match_color = [(0, 255, 0) if inliers[i] else (255, 0, 0) for i in range(len(matches))]
    else:
        match_color = (0, 255, 0)

    # Draw matches
    img_matches = cv2.drawMatches(
        img1, keypoints1,
        img2, keypoints2,
        matches, None,
        matchColor=match_color if isinstance(match_color, tuple) else None,
        matchesMask=None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )

    # If custom colors, draw manually
    if isinstance(match_color, list):
        img_matches = cv2.drawMatches(
            img1, keypoints1,
            img2, keypoints2,
            matches, None,
            matchesMask=None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )

    fig, ax = plt.subplots(figsize=(15, 8))
    ax.imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB) if len(img_matches.shape) == 3 else img_matches, cmap='gray')
    ax.set_title(title)
    ax.axis('off')

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_adaptation_curve(
    steps: List[int],
    accuracies: List[float],
    save_path: Optional[str] = None,
    title: str = "Meta-Adaptation Curve",
) -> plt.Figure:
    """
    Plot accuracy vs number of adaptation steps (for MAML evaluation).
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(steps, accuracies, marker='o', linewidth=2, markersize=8, color='#1f77b4')
    ax.fill_between(steps, accuracies, alpha=0.2)

    ax.set_xlabel('Adaptation Steps')
    ax.set_ylabel('Accuracy')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_domain_comparison(
    results: Dict[str, Dict[str, float]],
    domains: List[str] = ["illumination", "viewpoint"],
    save_path: Optional[str] = None,
    title: str = "Domain Performance Comparison",
) -> plt.Figure:
    """
    Plot performance across domains for different methods.
    """
    methods = list(results.keys())
    
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(methods))
    width = 0.35

    for i, domain in enumerate(domains):
        values = [results[m].get(f"{domain}_accuracy", 0) for m in methods]
        offset = (i - len(domains)/2 + 0.5) * width
        ax.bar(x + offset, values, width, label=domain.title())

    ax.set_xlabel('Method')
    ax.set_ylabel('Accuracy')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def create_summary_table(
    results: Dict[str, Dict[str, float]],
    metrics: List[str],
) -> str:
    """
    Create a formatted markdown table of results.
    """
    # Header
    header = "| Method | " + " | ".join(metrics) + " |"
    separator = "|" + "|".join(["---"] * (len(metrics) + 1)) + "|"

    rows = [header, separator]

    for method, method_results in results.items():
        values = [f"{method_results.get(m, 0):.4f}" for m in metrics]
        row = f"| {method} | " + " | ".join(values) + " |"
        rows.append(row)

    return "\n".join(rows)
