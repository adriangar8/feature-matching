"""
Matching comparison visualizer for SIFT vs Learned descriptors.
"""

from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from ..utils.preprocessing import normalize_patch

class MatchingComparisonVisualizer:
    """
    Create side-by-side qualitative comparisons of SIFT vs Learned descriptors.
    Shows the same query on both methods and their top matches.
    """
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Colors
        self.query_color = (155, 89, 182)      # Purple (BGR)
        self.correct_color = (96, 174, 39)     # Green (BGR)
        self.wrong_color = (34, 126, 230)      # Orange (BGR)
        self.distractor_color = (180, 180, 180)  # Gray (BGR)
    
    def extract_sift_descriptor(self, patch: np.ndarray) -> Optional[np.ndarray]:
        """Extract SIFT descriptor from a patch."""
        sift = cv2.SIFT_create() # type: ignore
        
        if patch.max() <= 1.0:
            patch = (patch * 255).astype(np.uint8)
        else:
            patch = patch.astype(np.uint8)
        
        if patch.shape[0] < 64 or patch.shape[1] < 64:
            patch = cv2.resize(patch, (64, 64))
        
        h, w = patch.shape[:2]
        kp = cv2.KeyPoint(w/2, h/2, 16)
        
        _, desc = sift.compute(patch, [kp])
        if desc is not None and len(desc) > 0:
            desc = desc[0] / (np.linalg.norm(desc[0]) + 1e-8)
            return desc
        return None
    
    def extract_deep_descriptor(
        self, 
        model: nn.Module, 
        patch: np.ndarray, 
        device: str
    ) -> np.ndarray:
        """Extract learned descriptor from a patch."""
        model.eval()
        normalized = normalize_patch(patch)
        tensor = torch.from_numpy(normalized).unsqueeze(0).to(device)
        
        with torch.no_grad():
            desc = model(tensor)
            desc = F.normalize(desc, p=2, dim=1)
        
        return desc.cpu().numpy()[0]
    
    def compute_rankings(
        self,
        query_desc: np.ndarray,
        correct_desc: np.ndarray,
        distractor_descs: List[np.ndarray],
        use_cosine: bool = True
    ) -> Tuple[int, List[int], List[float]]:
        """
        Compute ranking of correct match among distractors.
        Returns: (rank_of_correct, sorted_indices, distances)
        """
        all_descs = [correct_desc] + distractor_descs
        
        if use_cosine:
            # Cosine distance (1 - similarity)
            distances = []
            for desc in all_descs:
                sim = np.dot(query_desc, desc) / (np.linalg.norm(query_desc) * np.linalg.norm(desc) + 1e-8)
                distances.append(1 - sim)
        else:
            # L2 distance
            distances = [np.linalg.norm(query_desc - desc) for desc in all_descs]
        
        sorted_indices = np.argsort(distances)
        rank = int(np.where(sorted_indices == 0)[0][0]) + 1
        
        return rank, sorted_indices.tolist(), distances # type: ignore
    
    def create_comparison_figure(
        self,
        query_img: np.ndarray,
        target_img: np.ndarray,
        query_pos: Tuple[float, float],
        correct_pos: Tuple[float, float],
        distractor_positions: List[Tuple[float, float]],
        sift_rank: int,
        sift_top5_positions: List[Tuple[float, float]],
        learned_rank: int,
        learned_top5_positions: List[Tuple[float, float]],
        seq_name: str,
        domain: str,
        save_path: Path,
        model_name: str = "ResNet50"
    ):
        """
        Create a figure comparing SIFT vs Learned matching on the same image pair.
        
        Layout:
        - Row 1: Query image (shared)
        - Row 2: SIFT results on target image
        - Row 3: Learned results on target image
        """
        
        fig = plt.figure(figsize=(16, 14))
        gs = GridSpec(3, 2, figure=fig, height_ratios=[1, 1.2, 1.2], hspace=0.25, wspace=0.1)
        
        # =====================================================================
        # Row 1: Query Image
        # =====================================================================
        ax_query = fig.add_subplot(gs[0, :])
        
        query_display = cv2.cvtColor(query_img, cv2.COLOR_GRAY2RGB) if len(query_img.shape) == 2 else query_img.copy()
        qx, qy = int(query_pos[0]), int(query_pos[1])
        
        # Draw query point
        cv2.circle(query_display, (qx, qy), 20, (155, 89, 182), 3)
        cv2.circle(query_display, (qx, qy), 8, (155, 89, 182), -1)
        
        # Extract and show query patch
        half = 16
        if qy-half >= 0 and qy+half <= query_img.shape[0] and qx-half >= 0 and qx+half <= query_img.shape[1]:
            query_patch = query_img[qy-half:qy+half, qx-half:qx+half]
            # Add patch inset
            ax_inset = ax_query.inset_axes([0.02, 0.65, 0.15, 0.3]) # type: ignore
            ax_inset.imshow(query_patch, cmap='gray')
            ax_inset.set_title('Query Patch', fontsize=10)
            ax_inset.axis('off')
            for spine in ax_inset.spines.values():
                spine.set_edgecolor('purple')
                spine.set_linewidth(3)
        
        ax_query.imshow(cv2.cvtColor(query_display, cv2.COLOR_BGR2RGB))
        ax_query.set_title(f'Query Image - {domain.capitalize()} Domain\\nSequence: {seq_name}', 
                          fontsize=14, fontweight='bold')
        ax_query.axis('off')
        
        # =====================================================================
        # Row 2: SIFT Results
        # =====================================================================
        ax_sift = fig.add_subplot(gs[1, :])
        
        sift_display = cv2.cvtColor(target_img, cv2.COLOR_GRAY2RGB) if len(target_img.shape) == 2 else target_img.copy()
        mx, my = int(correct_pos[0]), int(correct_pos[1])
        
        # Draw all distractors (faint)
        for dx, dy in distractor_positions[:50]:
            dxi, dyi = int(dx), int(dy)
            cv2.circle(sift_display, (dxi, dyi), 5, (200, 200, 200), 1)
        
        # Draw SIFT top-5 predictions
        for i, (px, py) in enumerate(sift_top5_positions[:5]):
            pxi, pyi = int(px), int(py)
            if i == 0:  # Top prediction
                color = (96, 174, 39) if sift_rank == 1 else (34, 126, 230)  # Green if correct, orange if wrong
                cv2.circle(sift_display, (pxi, pyi), 18, color, 3)
                cv2.putText(sift_display, "1", (pxi-6, pyi+6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            else:
                cv2.circle(sift_display, (pxi, pyi), 12, (100, 100, 255), 2)
                cv2.putText(sift_display, str(i+1), (pxi-6, pyi+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 255), 2)
        
        # Draw ground truth
        cv2.circle(sift_display, (mx, my), 22, (96, 174, 39), 3)
        cv2.circle(sift_display, (mx, my), 6, (96, 174, 39), -1)
        
        ax_sift.imshow(cv2.cvtColor(sift_display, cv2.COLOR_BGR2RGB))
        
        sift_status = "✓ CORRECT" if sift_rank == 1 else f"✗ Rank {sift_rank}"
        sift_color = 'green' if sift_rank == 1 else 'red'
        ax_sift.set_title(f'SIFT Matching Result: {sift_status}', fontsize=14, fontweight='bold', color=sift_color)
        ax_sift.axis('off')
        
        # Legend for SIFT
        ax_sift.text(0.02, 0.98, '● Ground Truth (green)\\n○ Top-5 predictions (numbered)', 
                    transform=ax_sift.transAxes, fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # =====================================================================
        # Row 3: Learned Results
        # =====================================================================
        ax_learned = fig.add_subplot(gs[2, :])
        
        learned_display = cv2.cvtColor(target_img, cv2.COLOR_GRAY2RGB) if len(target_img.shape) == 2 else target_img.copy()
        
        # Draw all distractors (faint)
        for dx, dy in distractor_positions[:50]:
            dxi, dyi = int(dx), int(dy)
            cv2.circle(learned_display, (dxi, dyi), 5, (200, 200, 200), 1)
        
        # Draw Learned top-5 predictions
        for i, (px, py) in enumerate(learned_top5_positions[:5]):
            pxi, pyi = int(px), int(py)
            if i == 0:
                color = (96, 174, 39) if learned_rank == 1 else (34, 126, 230)
                cv2.circle(learned_display, (pxi, pyi), 18, color, 3)
                cv2.putText(learned_display, "1", (pxi-6, pyi+6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            else:
                cv2.circle(learned_display, (pxi, pyi), 12, (100, 100, 255), 2)
                cv2.putText(learned_display, str(i+1), (pxi-6, pyi+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 255), 2)
        
        # Draw ground truth
        cv2.circle(learned_display, (mx, my), 22, (96, 174, 39), 3)
        cv2.circle(learned_display, (mx, my), 6, (96, 174, 39), -1)
        
        ax_learned.imshow(cv2.cvtColor(learned_display, cv2.COLOR_BGR2RGB))
        
        learned_status = "✓ CORRECT" if learned_rank == 1 else f"✗ Rank {learned_rank}"
        learned_color = 'green' if learned_rank == 1 else 'red'
        ax_learned.set_title(f'{model_name} Matching Result: {learned_status}', fontsize=14, fontweight='bold', color=learned_color)
        ax_learned.axis('off')
        
        ax_learned.text(0.02, 0.98, '● Ground Truth (green)\\n○ Top-5 predictions (numbered)', 
                       transform=ax_learned.transAxes, fontsize=10, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"  Saved matching comparison: {save_path}")
    
    def generate_comparisons(
        self,
        data_mgr,  # HPatchesManager
        model: nn.Module,
        device: str,
        n_examples: int = 5,
        max_distractors: int = 100,
        model_name: str = "ResNet50"
    ):
        """
        Generate comparison figures for multiple examples.
        """
        
        print("\\nGenerating SIFT vs Learned matching comparisons...")
        
        for domain in ["illumination", "viewpoint"]:
            test_seqs = data_mgr.get_sequences(domain, "test")[:3]
            
            examples_saved = 0
            
            for seq_name in test_seqs:
                if examples_saved >= n_examples:
                    break
                
                try:
                    seq = data_mgr.load_sequence(seq_name)
                except Exception:
                    continue
                
                ref_img = seq["ref"]
                
                # Use target image 3 (moderate transformation)
                target = None
                for t in seq["targets"]:
                    if t["idx"] == 3:
                        target = t
                        break
                if target is None and seq["targets"]:
                    target = seq["targets"][0]
                if target is None:
                    continue
                
                target_img = target["image"]
                H = target["H"]
                
                # Detect keypoints
                detector = cv2.SIFT_create(nfeatures=500) # type: ignore
                kps_ref = detector.detect(ref_img)
                kps_target = detector.detect(target_img)
                
                if len(kps_ref) < 10 or len(kps_target) < 20:
                    continue
                
                half = 16
                
                # Find a good query keypoint
                for kp in kps_ref[:20]:
                    qx, qy = int(kp.pt[0]), int(kp.pt[1])
                    
                    if qy-half < 0 or qy+half > ref_img.shape[0] or qx-half < 0 or qx+half > ref_img.shape[1]:
                        continue
                    
                    query_patch = ref_img[qy-half:qy+half, qx-half:qx+half].astype(np.float32) / 255.0
                    
                    # Get ground truth position
                    pt_ref = np.array([[kp.pt[0], kp.pt[1]]], dtype=np.float32).reshape(-1, 1, 2)
                    pt_target = cv2.perspectiveTransform(pt_ref, H)[0, 0]
                    mx, my = int(pt_target[0]), int(pt_target[1])
                    
                    if my-half < 0 or my+half > target_img.shape[0] or mx-half < 0 or mx+half > target_img.shape[1]:
                        continue
                    
                    correct_patch = target_img[my-half:my+half, mx-half:mx+half].astype(np.float32) / 255.0
                    
                    # Get distractor patches and positions
                    distractor_patches = []
                    distractor_positions = []
                    
                    for kp_t in kps_target:
                        dx, dy = int(kp_t.pt[0]), int(kp_t.pt[1])
                        
                        dist_to_match = np.sqrt((dx - mx)**2 + (dy - my)**2)
                        if dist_to_match < 15:
                            continue
                        
                        if dy-half < 0 or dy+half > target_img.shape[0] or dx-half < 0 or dx+half > target_img.shape[1]:
                            continue
                        
                        d_patch = target_img[dy-half:dy+half, dx-half:dx+half].astype(np.float32) / 255.0
                        distractor_patches.append(d_patch)
                        distractor_positions.append((kp_t.pt[0], kp_t.pt[1]))
                        
                        if len(distractor_patches) >= max_distractors:
                            break
                    
                    if len(distractor_patches) < 20:
                        continue
                    
                    # =========================================================
                    # SIFT matching
                    # =========================================================
                    sift_query = self.extract_sift_descriptor(query_patch)
                    sift_correct = self.extract_sift_descriptor(correct_patch)
                    sift_distractors = [self.extract_sift_descriptor(p) for p in distractor_patches]
                    sift_distractors = [d for d in sift_distractors if d is not None]
                    
                    if sift_query is None or sift_correct is None or len(sift_distractors) < 10:
                        continue
                    
                    sift_rank, sift_sorted_idx, _ = self.compute_rankings(
                        sift_query, sift_correct, sift_distractors, use_cosine=True
                    )
                    
                    # Get positions of SIFT top-5
                    sift_top5_positions = []
                    for idx in sift_sorted_idx[:5]:
                        if idx == 0:
                            sift_top5_positions.append((mx, my))
                        else:
                            sift_top5_positions.append(distractor_positions[idx - 1])
                    
                    # =========================================================
                    # Learned matching
                    # =========================================================
                    learned_query = self.extract_deep_descriptor(model, query_patch, device)
                    learned_correct = self.extract_deep_descriptor(model, correct_patch, device)
                    learned_distractors = [self.extract_deep_descriptor(model, p, device) for p in distractor_patches]
                    
                    learned_rank, learned_sorted_idx, _ = self.compute_rankings(
                        learned_query, learned_correct, learned_distractors, use_cosine=True
                    )
                    
                    # Get positions of Learned top-5
                    learned_top5_positions = []
                    for idx in learned_sorted_idx[:5]:
                        if idx == 0:
                            learned_top5_positions.append((mx, my))
                        else:
                            learned_top5_positions.append(distractor_positions[idx - 1])
                    
                    # =========================================================
                    # Create comparison figure
                    # =========================================================
                    save_path = self.output_dir / f"matching_comparison_{domain}_{seq_name}_{examples_saved:02d}.png"
                    
                    self.create_comparison_figure(
                        query_img=ref_img,
                        target_img=target_img,
                        query_pos=(qx, qy),
                        correct_pos=(mx, my),
                        distractor_positions=distractor_positions,
                        sift_rank=sift_rank,
                        sift_top5_positions=sift_top5_positions,
                        learned_rank=learned_rank,
                        learned_top5_positions=learned_top5_positions,
                        seq_name=seq_name,
                        domain=domain,
                        save_path=save_path,
                        model_name=model_name
                    )
                    
                    examples_saved += 1
                    break  # One example per sequence
        
        print(f"  Generated {examples_saved} comparison figures per domain")
