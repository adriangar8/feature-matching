from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from sklearn.manifold import TSNE

from ..data.structures import TSNESample
from ..utils.preprocessing import normalize_patch

class TSNEVisualizer:    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.query_color = '#9B59B6'      
        self.match_color = '#27AE60'      
        self.distractor_color = '#E67E22' 
    
    def extract_embeddings_deep(
        self,
        model: nn.Module,
        sample: TSNESample,
        device: str
    ) -> np.ndarray:
        model.eval()
        
        all_patches = [sample.query_patch, sample.correct_patch] + sample.distractor_patches
        
        tensors = []
        for p in all_patches:
            normalized = normalize_patch(p)
            tensors.append(torch.from_numpy(normalized))
        
        batch = torch.stack(tensors).to(device)
        
        with torch.no_grad():
            embeddings = model(batch)
            embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings.cpu().numpy()
    
    def extract_embeddings_sift(self, sample: TSNESample) -> np.ndarray:
        sift = cv2.SIFT_create() # type: ignore
        embeddings = []
        
        all_patches = [sample.query_patch, sample.correct_patch] + sample.distractor_patches
        
        for patch in all_patches:
            if patch.max() <= 1.0:
                patch_uint8 = (patch * 255).astype(np.uint8)
            else:
                patch_uint8 = patch.astype(np.uint8)
            
            patch_resized = cv2.resize(patch_uint8, (64, 64))
            h, w = patch_resized.shape[:2]
            kp = cv2.KeyPoint(w/2, h/2, 16)
            
            _, desc = sift.compute(patch_resized, [kp])
            if desc is not None:
                desc = desc[0] / (np.linalg.norm(desc[0]) + 1e-8)
                embeddings.append(desc)
            else:
                embeddings.append(np.zeros(128))
        
        return np.array(embeddings)
    
    def compute_tsne_2d(self, embeddings: np.ndarray, perplexity: int = 5) -> np.ndarray:
        n_samples = embeddings.shape[0]
        perplexity = min(perplexity, n_samples - 1)
        
        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            max_iter=1000,
            random_state=42,
            init='pca'
        )
        return tsne.fit_transform(embeddings)
    
    def create_tsne_figure(
        self,
        coords: np.ndarray,
        sample: TSNESample,
        model_name: str,
        save_path: Path
    ):
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        all_patches = [sample.query_patch, sample.correct_patch] + sample.distractor_patches
        
        for i in range(2, len(coords)):
            ax.scatter(
                coords[i, 0], coords[i, 1],
                c=self.distractor_color, s=150, alpha=0.6,
                marker='o', zorder=1
            )
        
        ax.scatter(
            coords[1, 0], coords[1, 1],
            c=self.match_color, s=300, alpha=1.0,
            marker='s', edgecolors='black', linewidths=2,
            label='Correct Match', zorder=2
        )
        
        ax.scatter(
            coords[0, 0], coords[0, 1],
            c=self.query_color, s=300, alpha=1.0,
            marker='^', edgecolors='black', linewidths=2,
            label='Query', zorder=3
        )
        
        ax.plot(
            [coords[0, 0], coords[1, 0]],
            [coords[0, 1], coords[1, 1]],
            'g--', linewidth=2, alpha=0.7, zorder=0
        )
        
        def add_patch_thumbnail(patch, x, y, zoom=0.6, border_color='black'):
            if patch.max() <= 1.0:
                patch_display = (patch * 255).astype(np.uint8)
            else:
                patch_display = patch.astype(np.uint8)
            
            patch_display = cv2.copyMakeBorder(
                patch_display, 2, 2, 2, 2,
                cv2.BORDER_CONSTANT,
                value=128
            )
            
            imagebox = OffsetImage(patch_display, zoom=zoom, cmap='gray')
            ab = AnnotationBbox(imagebox, (x, y), frameon=False, zorder=10)
            ax.add_artist(ab)
        
        add_patch_thumbnail(all_patches[0], coords[0, 0], coords[0, 1], zoom=0.8)
        add_patch_thumbnail(all_patches[1], coords[1, 0], coords[1, 1], zoom=0.8)
        
        if len(coords) > 4:
            distractor_coords = coords[2:]
            distractor_dists = [np.linalg.norm(coords[0] - c) for c in distractor_coords]
            closest_idx = np.argmin(distractor_dists)
            add_patch_thumbnail(
                all_patches[2 + closest_idx],
                distractor_coords[closest_idx, 0],
                distractor_coords[closest_idx, 1],
                zoom=0.5
            )
        
        query_to_match = np.linalg.norm(coords[0] - coords[1])
        query_to_distractors = [np.linalg.norm(coords[0] - coords[i]) for i in range(2, len(coords))]
        min_distractor_dist = min(query_to_distractors) if query_to_distractors else float('inf')
        mean_distractor_dist = np.mean(query_to_distractors) if query_to_distractors else 0
        
        ax.set_xlabel('T-SNE Dimension 1', fontsize=12)
        ax.set_ylabel('T-SNE Dimension 2', fontsize=12)
        
        title = f'{model_name} Embedding Space - {sample.domain.capitalize()}\n'
        title += f'Sequence: {sample.seq_name}'
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        separation_ratio = min_distractor_dist / query_to_match if query_to_match > 0 else 0
        metrics_text = f'Query-Match dist: {query_to_match:.2f}\n'
        metrics_text += f'Query-Nearest Distractor: {min_distractor_dist:.2f}\n'
        metrics_text += f'Separation Ratio: {separation_ratio:.2f}x'
        
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes, fontsize=11,
                verticalalignment='top', bbox=props)
        
        ax.legend(loc='upper right', fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved T-SNE: {save_path}")
    
    def create_global_context_figure(
        self,
        sample: TSNESample,
        model_name: str,
        save_path: Path
    ):
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        query_img = cv2.cvtColor(sample.query_global_img, cv2.COLOR_GRAY2RGB)
        qx, qy = int(sample.query_pos[0]), int(sample.query_pos[1])
        cv2.circle(query_img, (qx, qy), 15, (155, 89, 182), 3)
        cv2.circle(query_img, (qx, qy), 5, (155, 89, 182), -1)
        
        axes[0].imshow(query_img)
        axes[0].set_title('Query Image', fontsize=12, fontweight='bold')
        axes[0].axis('off')
        
        match_img = cv2.cvtColor(sample.match_global_img, cv2.COLOR_GRAY2RGB)
        mx, my = int(sample.match_pos[0]), int(sample.match_pos[1])
        cv2.circle(match_img, (mx, my), 15, (39, 174, 96), 3)
        cv2.circle(match_img, (mx, my), 5, (39, 174, 96), -1)
        
        for dx, dy in sample.distractor_positions[:10]:
            dxi, dyi = int(dx), int(dy)
            cv2.circle(match_img, (dxi, dyi), 8, (230, 126, 34), 2)
        
        axes[1].imshow(match_img)
        axes[1].set_title('Target Image (with distractors)', fontsize=12, fontweight='bold')
        axes[1].axis('off')
        
        plt.suptitle(
            f'{model_name} - {sample.domain.capitalize()} Domain\nSequence: {sample.seq_name}',
            fontsize=14, fontweight='bold', y=1.02
        )
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved global context: {save_path}")
    
    def create_combined_tsne_figure(
        self,
        samples: List[TSNESample],
        models: Dict[str, Tuple[nn.Module, str]], 
        device: str,
        save_path: Path
    ):
        
        n_models = len(models)
        n_domains = 2
        
        fig, axes = plt.subplots(n_domains, n_models, figsize=(5*n_models, 10))
        
        illum_samples = [s for s in samples if s.domain == "illumination"]
        view_samples = [s for s in samples if s.domain == "viewpoint"]
        
        for col, (model_name, (model, model_type)) in enumerate(models.items()):
            for row, (domain_samples, domain_name) in enumerate([(illum_samples, "Illumination"), (view_samples, "Viewpoint")]):
                ax = axes[row, col] if n_models > 1 else axes[row]
                
                if not domain_samples:
                    ax.text(0.5, 0.5, "No samples", ha='center', va='center')
                    ax.set_title(f'{model_name}\n{domain_name}')
                    continue
                
                sample = domain_samples[0]
                
                if model_type == "sift":
                    embeddings = self.extract_embeddings_sift(sample)
                else:
                    embeddings = self.extract_embeddings_deep(model, sample, device)
                
                coords = self.compute_tsne_2d(embeddings)
                
                for i in range(2, len(coords)):
                    ax.scatter(coords[i, 0], coords[i, 1], c=self.distractor_color, 
                              s=80, alpha=0.6, marker='o')
                
                ax.scatter(coords[1, 0], coords[1, 1], c=self.match_color,
                          s=200, marker='s', edgecolors='black', linewidths=2)
                ax.scatter(coords[0, 0], coords[0, 1], c=self.query_color,
                          s=200, marker='^', edgecolors='black', linewidths=2)
                
                ax.plot([coords[0, 0], coords[1, 0]], [coords[0, 1], coords[1, 1]],
                       'g--', linewidth=2, alpha=0.7)
                
                query_to_match = np.linalg.norm(coords[0] - coords[1])
                query_to_distractors = [np.linalg.norm(coords[0] - coords[i]) for i in range(2, len(coords))]
                min_dist = min(query_to_distractors) if query_to_distractors else 0
                sep_ratio = min_dist / query_to_match if query_to_match > 0 else 0
                
                ax.set_title(f'{model_name}\n{domain_name} (sep={sep_ratio:.1f}x)', fontsize=11)
                ax.grid(True, alpha=0.3)
        
        legend_elements = [
            mpatches.Patch(facecolor=self.query_color, label='Query'),
            mpatches.Patch(facecolor=self.match_color, label='Correct Match'),
            mpatches.Patch(facecolor=self.distractor_color, label='Distractors'),
        ]
        fig.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=11)
        
        plt.suptitle('T-SNE Embedding Space Comparison', fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0, 0.05, 1, 0.95]) # type: ignore
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved combined T-SNE: {save_path}")
