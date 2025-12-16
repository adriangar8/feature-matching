"""
Methodology figure creation for paper.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

def create_methodology_figure(output_dir: Path):
    """Create methodology diagram for paper."""
    
    fig = plt.figure(figsize=(14, 8))
    gs = GridSpec(2, 3, figure=fig, height_ratios=[1, 1.2], hspace=0.3, wspace=0.3)
    
    # Colors
    query_color = '#9B59B6'
    match_color = '#27AE60'
    distractor_color = '#E67E22'
    
    # Top row: Triplet sampling
    ax_triplet = fig.add_subplot(gs[0, :])
    ax_triplet.set_xlim(0, 10)
    ax_triplet.set_ylim(0, 3)
    ax_triplet.axis('off')
    
    # Draw anchor
    anchor_rect = mpatches.FancyBboxPatch((0.5, 1), 1.5, 1.5, 
                                           boxstyle="round,pad=0.05",
                                           facecolor=query_color, alpha=0.3,
                                           edgecolor=query_color, linewidth=2)
    ax_triplet.add_patch(anchor_rect)
    ax_triplet.text(1.25, 0.6, 'Anchor\n(Query)', ha='center', fontsize=11, fontweight='bold')
    
    # Draw positive
    pos_rect = mpatches.FancyBboxPatch((3.5, 1), 1.5, 1.5,
                                       boxstyle="round,pad=0.05",
                                       facecolor=match_color, alpha=0.3,
                                       edgecolor=match_color, linewidth=2)
    ax_triplet.add_patch(pos_rect)
    ax_triplet.text(4.25, 0.6, 'Positive\n(Match)', ha='center', fontsize=11, fontweight='bold')
    
    # Draw negative
    neg_rect = mpatches.FancyBboxPatch((6.5, 1), 1.5, 1.5,
                                       boxstyle="round,pad=0.05",
                                       facecolor=distractor_color, alpha=0.3,
                                       edgecolor=distractor_color, linewidth=2)
    ax_triplet.add_patch(neg_rect)
    ax_triplet.text(7.25, 0.6, 'Negative\n(Distractor)', ha='center', fontsize=11, fontweight='bold')
    
    # Arrows
    ax_triplet.annotate('', xy=(3.4, 1.75), xytext=(2.1, 1.75),
                       arrowprops=dict(arrowstyle='->', color=match_color, lw=2))
    ax_triplet.text(2.75, 2.0, 'Pull\nCloser', ha='center', fontsize=10, color=match_color)
    
    ax_triplet.annotate('', xy=(6.4, 1.75), xytext=(5.1, 1.75),
                       arrowprops=dict(arrowstyle='<->', color=distractor_color, lw=2))
    ax_triplet.text(5.75, 2.0, 'Push\nApart', ha='center', fontsize=10, color=distractor_color)
    
    ax_triplet.set_title('Triplet Contrastive Learning: Learning Invariant Descriptors', 
                        fontsize=14, fontweight='bold', pad=10)
    
    # Bottom row: Embedding space before/after
    ax_before = fig.add_subplot(gs[1, 0])
    ax_after = fig.add_subplot(gs[1, 1])
    ax_loss = fig.add_subplot(gs[1, 2])
    
    # Before training
    np.random.seed(42)
    n_points = 15
    before_query = np.array([0, 0])
    before_pos = np.random.randn(2) * 0.8
    before_neg = np.random.randn(n_points, 2) * 0.8
    
    ax_before.scatter(before_neg[:, 0], before_neg[:, 1], c=distractor_color, 
                     s=80, alpha=0.6, label='Distractors')
    ax_before.scatter(before_pos[0], before_pos[1], c=match_color, 
                     s=150, marker='s', edgecolors='black', linewidths=2, label='Match')
    ax_before.scatter(before_query[0], before_query[1], c=query_color, 
                     s=150, marker='^', edgecolors='black', linewidths=2, label='Query')
    
    ax_before.set_xlim(-2, 2)
    ax_before.set_ylim(-2, 2)
    ax_before.set_title('Before Training', fontweight='bold')
    ax_before.set_xlabel('Embedding Dim 1')
    ax_before.set_ylabel('Embedding Dim 2')
    ax_before.grid(True, alpha=0.3)
    ax_before.legend(loc='upper right', fontsize=9)
    
    # After training
    after_query = np.array([0, 0])
    after_pos = np.array([0.2, 0.15])
    angles = np.linspace(0, 2*np.pi, n_points, endpoint=False)
    radii = 1.2 + np.random.rand(n_points) * 0.3
    after_neg = np.stack([radii * np.cos(angles), radii * np.sin(angles)], axis=1)
    
    ax_after.scatter(after_neg[:, 0], after_neg[:, 1], c=distractor_color,
                    s=80, alpha=0.6, label='Distractors')
    ax_after.scatter(after_pos[0], after_pos[1], c=match_color,
                    s=150, marker='s', edgecolors='black', linewidths=2, label='Match')
    ax_after.scatter(after_query[0], after_query[1], c=query_color,
                    s=150, marker='^', edgecolors='black', linewidths=2, label='Query')
    
    circle = plt.Circle((0, 0), 0.3, fill=False, color='gray', linestyle='--', linewidth=1.5) # type: ignore
    ax_after.add_patch(circle)
    
    ax_after.set_xlim(-2, 2)
    ax_after.set_ylim(-2, 2)
    ax_after.set_title('After Training', fontweight='bold')
    ax_after.set_xlabel('Embedding Dim 1')
    ax_after.set_ylabel('Embedding Dim 2')
    ax_after.grid(True, alpha=0.3)
    ax_after.legend(loc='upper right', fontsize=9)
    
    # Loss function
    ax_loss.text(0.5, 0.85, 'Triplet Loss:', ha='center', fontsize=12, fontweight='bold',
                transform=ax_loss.transAxes)
    ax_loss.text(0.5, 0.65, r'$\mathcal{L} = \max(0, d_+ - d_- + m)$', ha='center', fontsize=14,
                transform=ax_loss.transAxes)
    ax_loss.text(0.5, 0.45, 'where:', ha='center', fontsize=10, transform=ax_loss.transAxes)
    ax_loss.text(0.5, 0.30, r'$d_+ = \|f(a) - f(p)\|^2$', ha='center', fontsize=10,
                transform=ax_loss.transAxes, color=match_color)
    ax_loss.text(0.5, 0.15, r'$d_- = \|f(a) - f(n)\|^2$', ha='center', fontsize=10,
                transform=ax_loss.transAxes, color=distractor_color)
    ax_loss.text(0.5, 0.02, r'$m = 0.3$ (margin)', ha='center', fontsize=10,
                transform=ax_loss.transAxes, color='gray')
    ax_loss.axis('off')
    ax_loss.set_title('Objective Function', fontweight='bold')
    
    plt.tight_layout()
    
    save_path = output_dir / 'methodology_contrastive_learning.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Saved methodology figure: {save_path}")
    return save_path
