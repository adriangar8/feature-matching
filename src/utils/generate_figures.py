#!/usr/bin/env python3
"""
Generate Paper Figures

This script generates:
1. methodology_contrastive_learning.png - Contrastive learning diagram
2. Domain example images (illum_ref.png, illum_target.png, view_ref.png, view_target.png)

Usage:
    python generate_paper_figures.py --hpatches /path/to/hpatches --output figures/
"""

import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from pathlib import Path


def create_methodology_figure(output_dir: Path):
    """Create contrastive learning methodology diagram."""
    
    fig = plt.figure(figsize=(10, 6))
    gs = GridSpec(2, 3, figure=fig, height_ratios=[1, 1.2], hspace=0.35, wspace=0.3)
    
    # Colors
    query_color = '#9B59B6'      # Purple
    match_color = '#27AE60'       # Green
    distractor_color = '#E67E22'  # Orange
    
    # Top row: Triplet sampling
    ax_triplet = fig.add_subplot(gs[0, :])
    ax_triplet.set_xlim(0, 10)
    ax_triplet.set_ylim(0, 3)
    ax_triplet.axis('off')
    
    # Draw anchor
    anchor_rect = mpatches.FancyBboxPatch((0.5, 0.8), 1.8, 1.8, 
                                           boxstyle="round,pad=0.05",
                                           facecolor=query_color, alpha=0.3,
                                           edgecolor=query_color, linewidth=2)
    ax_triplet.add_patch(anchor_rect)
    ax_triplet.text(1.4, 0.4, 'Anchor\n(Query)', ha='center', fontsize=10, fontweight='bold')
    
    # Draw positive
    pos_rect = mpatches.FancyBboxPatch((3.6, 0.8), 1.8, 1.8,
                                       boxstyle="round,pad=0.05",
                                       facecolor=match_color, alpha=0.3,
                                       edgecolor=match_color, linewidth=2)
    ax_triplet.add_patch(pos_rect)
    ax_triplet.text(4.5, 0.4, 'Positive\n(Match)', ha='center', fontsize=10, fontweight='bold')
    
    # Draw negative
    neg_rect = mpatches.FancyBboxPatch((6.7, 0.8), 1.8, 1.8,
                                       boxstyle="round,pad=0.05",
                                       facecolor=distractor_color, alpha=0.3,
                                       edgecolor=distractor_color, linewidth=2)
    ax_triplet.add_patch(neg_rect)
    ax_triplet.text(7.6, 0.4, 'Negative\n(Distractor)', ha='center', fontsize=10, fontweight='bold')
    
    # Arrows
    ax_triplet.annotate('', xy=(3.5, 1.7), xytext=(2.4, 1.7),
                       arrowprops=dict(arrowstyle='->', color=match_color, lw=2.5))
    ax_triplet.text(2.95, 2.1, 'Pull\nCloser', ha='center', fontsize=9, color=match_color, fontweight='bold')
    
    ax_triplet.annotate('', xy=(6.6, 1.7), xytext=(5.5, 1.7),
                       arrowprops=dict(arrowstyle='<->', color=distractor_color, lw=2.5))
    ax_triplet.text(6.05, 2.1, 'Push\nApart', ha='center', fontsize=9, color=distractor_color, fontweight='bold')
    
    ax_triplet.set_title('Triplet Contrastive Learning', fontsize=12, fontweight='bold', pad=10)
    
    # Bottom row: Embedding space before/after
    ax_before = fig.add_subplot(gs[1, 0])
    ax_after = fig.add_subplot(gs[1, 1])
    ax_loss = fig.add_subplot(gs[1, 2])
    
    # Before training - random embeddings
    np.random.seed(42)
    n_points = 12
    before_query = np.array([0, 0])
    before_pos = np.random.randn(2) * 0.7
    before_neg = np.random.randn(n_points, 2) * 0.7
    
    ax_before.scatter(before_neg[:, 0], before_neg[:, 1], c=distractor_color, 
                     s=60, alpha=0.6, label='Distractors')
    ax_before.scatter(before_pos[0], before_pos[1], c=match_color, 
                     s=120, marker='s', edgecolors='black', linewidths=1.5, label='Match')
    ax_before.scatter(before_query[0], before_query[1], c=query_color, 
                     s=120, marker='^', edgecolors='black', linewidths=1.5, label='Query')
    
    ax_before.set_xlim(-1.8, 1.8)
    ax_before.set_ylim(-1.8, 1.8)
    ax_before.set_title('Before Training', fontsize=11, fontweight='bold')
    ax_before.set_xlabel('Dim 1', fontsize=9)
    ax_before.set_ylabel('Dim 2', fontsize=9)
    ax_before.grid(True, alpha=0.3)
    ax_before.set_aspect('equal')
    
    # After training - structured embeddings
    after_query = np.array([0, 0])
    after_pos = np.array([0.15, 0.12])
    angles = np.linspace(0, 2*np.pi, n_points, endpoint=False)
    radii = 1.1 + np.random.rand(n_points) * 0.25
    after_neg = np.stack([radii * np.cos(angles), radii * np.sin(angles)], axis=1)
    
    ax_after.scatter(after_neg[:, 0], after_neg[:, 1], c=distractor_color,
                    s=60, alpha=0.6, label='Distractors')
    ax_after.scatter(after_pos[0], after_pos[1], c=match_color,
                    s=120, marker='s', edgecolors='black', linewidths=1.5, label='Match')
    ax_after.scatter(after_query[0], after_query[1], c=query_color,
                    s=120, marker='^', edgecolors='black', linewidths=1.5, label='Query')
    
    # Draw margin circle
    circle = plt.Circle((0, 0), 0.3, fill=False, color='gray', linestyle='--', linewidth=1.5) # type: ignore
    ax_after.add_patch(circle)
    ax_after.text(0.35, -0.1, 'm', fontsize=9, color='gray', style='italic')
    
    ax_after.set_xlim(-1.8, 1.8)
    ax_after.set_ylim(-1.8, 1.8)
    ax_after.set_title('After Training', fontsize=11, fontweight='bold')
    ax_after.set_xlabel('Dim 1', fontsize=9)
    ax_after.set_ylabel('Dim 2', fontsize=9)
    ax_after.grid(True, alpha=0.3)
    ax_after.legend(loc='upper right', fontsize=7)
    ax_after.set_aspect('equal')
    
    # Loss function panel
    ax_loss.text(0.5, 0.82, 'Triplet Loss:', ha='center', fontsize=11, fontweight='bold',
                transform=ax_loss.transAxes)
    ax_loss.text(0.5, 0.62, r'$\mathcal{L} = \max(0, d_+ - d_- + m)$', ha='center', fontsize=12,
                transform=ax_loss.transAxes)
    ax_loss.text(0.5, 0.42, 'where:', ha='center', fontsize=9, transform=ax_loss.transAxes)
    ax_loss.text(0.5, 0.28, r'$d_+ = \|f(a) - f(p)\|^2$', ha='center', fontsize=9,
                transform=ax_loss.transAxes, color=match_color)
    ax_loss.text(0.5, 0.14, r'$d_- = \|f(a) - f(n)\|^2$', ha='center', fontsize=9,
                transform=ax_loss.transAxes, color=distractor_color)
    ax_loss.text(0.5, 0.02, r'$m = 0.3$ (margin)', ha='center', fontsize=9,
                transform=ax_loss.transAxes, color='gray')
    ax_loss.axis('off')
    ax_loss.set_title('Objective', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    
    save_path = output_dir / 'methodology_contrastive_learning.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✓ Saved: {save_path}")
    return save_path


def extract_domain_examples(hpatches_root: Path, output_dir: Path):
    """Extract example images from HPatches for illumination and viewpoint domains."""
    
    # Find illumination and viewpoint sequences
    illum_seqs = sorted([d for d in hpatches_root.iterdir() if d.name.startswith('i_') and d.is_dir()])
    view_seqs = sorted([d for d in hpatches_root.iterdir() if d.name.startswith('v_') and d.is_dir()])
    
    if not illum_seqs or not view_seqs:
        print("Warning: Could not find HPatches sequences")
        return
    
    # Select good examples (first few sequences are usually good)
    illum_seq = illum_seqs[0]
    view_seq = view_seqs[0]
    
    print(f"Using illumination sequence: {illum_seq.name}")
    print(f"Using viewpoint sequence: {view_seq.name}")
    
    # Load and save illumination images
    illum_ref = cv2.imread(str(illum_seq / "1.ppm"))
    illum_target = cv2.imread(str(illum_seq / "4.ppm"))  # Use image 4 for visible change
    
    if illum_ref is not None and illum_target is not None:
        # Resize for paper (max 400px width)
        scale = min(400 / illum_ref.shape[1], 300 / illum_ref.shape[0])
        new_size = (int(illum_ref.shape[1] * scale), int(illum_ref.shape[0] * scale))
        
        illum_ref_resized = cv2.resize(illum_ref, new_size)
        illum_target_resized = cv2.resize(illum_target, new_size)
        
        cv2.imwrite(str(output_dir / "illum_ref.png"), illum_ref_resized)
        cv2.imwrite(str(output_dir / "illum_target.png"), illum_target_resized)
        print(f"✓ Saved: illum_ref.png, illum_target.png")
    
    # Load and save viewpoint images
    view_ref = cv2.imread(str(view_seq / "1.ppm"))
    view_target = cv2.imread(str(view_seq / "4.ppm"))
    
    if view_ref is not None and view_target is not None:
        scale = min(400 / view_ref.shape[1], 300 / view_ref.shape[0])
        new_size = (int(view_ref.shape[1] * scale), int(view_ref.shape[0] * scale))
        
        view_ref_resized = cv2.resize(view_ref, new_size)
        view_target_resized = cv2.resize(view_target, new_size)
        
        cv2.imwrite(str(output_dir / "view_ref.png"), view_ref_resized)
        cv2.imwrite(str(output_dir / "view_target.png"), view_target_resized)
        print(f"✓ Saved: view_ref.png, view_target.png")


def create_placeholder_domain_images(output_dir: Path):
    """Create placeholder images if HPatches is not available."""
    
    # Create simple placeholder images with text
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.text(0.5, 0.5, 'Illumination\nReference', ha='center', va='center', fontsize=14)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_facecolor('#f0f0f0')
    plt.savefig(output_dir / 'illum_ref.png', dpi=150, bbox_inches='tight', facecolor='#f0f0f0')
    plt.close()
    
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.text(0.5, 0.5, 'Illumination\nTarget', ha='center', va='center', fontsize=14)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_facecolor('#d0d0d0')
    plt.savefig(output_dir / 'illum_target.png', dpi=150, bbox_inches='tight', facecolor='#d0d0d0')
    plt.close()
    
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.text(0.5, 0.5, 'Viewpoint\nReference', ha='center', va='center', fontsize=14)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_facecolor('#e8e8f0')
    plt.savefig(output_dir / 'view_ref.png', dpi=150, bbox_inches='tight', facecolor='#e8e8f0')
    plt.close()
    
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.text(0.5, 0.5, 'Viewpoint\nTarget', ha='center', va='center', fontsize=14)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_facecolor('#d8d8e0')
    plt.savefig(output_dir / 'view_target.png', dpi=150, bbox_inches='tight', facecolor='#d8d8e0')
    plt.close()
    
    print("✓ Created placeholder domain images")


def main():
    parser = argparse.ArgumentParser(description="Generate paper figures")
    parser.add_argument("--hpatches", type=str, default=None, 
                        help="Path to HPatches dataset (optional)")
    parser.add_argument("--output", type=str, default="figures",
                        help="Output directory for figures")
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*50)
    print("Generating Paper Figures")
    print("="*50)
    
    # 1. Create methodology figure
    create_methodology_figure(output_dir)
    
    # 2. Create domain example images
    if args.hpatches and Path(args.hpatches).exists():
        extract_domain_examples(Path(args.hpatches), output_dir)
    else:
        print("\nNote: HPatches path not provided or doesn't exist.")
        print("Creating placeholder images. Replace with actual HPatches images.")
        create_placeholder_domain_images(output_dir)
    
    print("\n" + "="*50)
    print(f"All figures saved to: {output_dir}/")
    print("="*50)
    print("\nFigures generated:")
    print("  - methodology_contrastive_learning.png")
    print("  - illum_ref.png")
    print("  - illum_target.png")
    print("  - view_ref.png")
    print("  - view_target.png")


if __name__ == "__main__":
    main()