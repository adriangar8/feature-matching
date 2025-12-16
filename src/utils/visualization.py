import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import json


@dataclass
class MatchingExample:
    query: np.ndarray
    correct_match: np.ndarray
    distractors: List[np.ndarray]
    distances: List[float]
    predicted_rank: int
    seq_name: str
    method: str
    query_global_pos: Optional[Tuple[float, float]] = None
    match_global_pos: Optional[Tuple[float, float]] = None
    global_image_paths: Optional[Tuple[str, str]] = None


class PatchVisualizer:
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.figures_dir = self.output_dir / "figures"
        self.patches_dir = self.output_dir / "patches"
        self.global_figures_dir = self.output_dir / "global_images"
        
        self.domains = ["illumination", "viewpoint"]
        for domain in self.domains:
            (self.figures_dir / domain).mkdir(parents=True, exist_ok=True)
            (self.patches_dir / domain).mkdir(parents=True, exist_ok=True)
            (self.global_figures_dir / domain).mkdir(parents=True, exist_ok=True)
        
        self.examples: Dict[str, List[MatchingExample]] = {}
        self.success_examples: Dict[str, List[MatchingExample]] = {}
        self.failure_examples: Dict[str, List[MatchingExample]] = {}
        
        self.patch_size = 32
        
        self.allowed_sequences = {
            "illumination": ["i_yellowtent"],
            "viewpoint": ["v_tabletop"],
        }
    
    def _get_domain_from_seq(self, seq_name: str) -> str:
        if seq_name.startswith("i_"):
            return "illumination"
        elif seq_name.startswith("v_"):
            return "viewpoint"
        else:
            return "unknown"
    
    def _is_allowed_sequence(self, seq_name: str, method: str) -> bool:
        domain = self._get_domain_from_seq(seq_name)
        if domain not in self.allowed_sequences:
            return False
        
        return seq_name in self.allowed_sequences[domain]
    
    def add_example(
        self,
        query: np.ndarray,
        correct_match: np.ndarray,
        distractors: List[np.ndarray],
        distances: List[float],
        predicted_rank: int,
        seq_name: str,
        method: str,
        max_examples: int = 50,
        query_global_pos: Optional[Tuple[float, float]] = None,
        match_global_pos: Optional[Tuple[float, float]] = None,
        global_image_paths: Optional[Tuple[str, str]] = None,
    ):
        if not self._is_allowed_sequence(seq_name, method):
            return  
        
        unique_distractors = []
        seen_hashes = set()
        
        for distractor in distractors:
            dist_hash = hash(distractor.tobytes())
            if dist_hash not in seen_hashes:
                seen_hashes.add(dist_hash)
                unique_distractors.append(distractor)
        
        unique_distractors = unique_distractors[:10]
        
        example = MatchingExample(
            query=query.copy(),
            correct_match=correct_match.copy(),
            distractors=unique_distractors,
            distances=distances[:11] if distances else [],
            predicted_rank=predicted_rank,
            seq_name=seq_name,
            method=method,
            query_global_pos=query_global_pos,
            match_global_pos=match_global_pos,
            global_image_paths=global_image_paths,
        )
                
        if method not in self.examples:
            self.examples[method] = []
            self.success_examples[method] = []
            self.failure_examples[method] = []
        
        if len(self.examples[method]) < max_examples:
            self.examples[method].append(example)
        
        if predicted_rank == 1:
            if len(self.success_examples[method]) < 20:
                self.success_examples[method].append(example)
        else:
            if len(self.failure_examples[method]) < 20:
                self.failure_examples[method].append(example)
    
    def save_patches(self, method: str):
        if method not in self.examples:
            return
        
        for example in self.examples[method][:10]:
            domain = self._get_domain_from_seq(example.seq_name)
            if domain == "unknown":
                continue
            
            method_dir = self.patches_dir / domain / method
            method_dir.mkdir(parents=True, exist_ok=True)
            
            example_idx = len(list(method_dir.glob("example_*")))
            example_dir = method_dir / f"example_{example_idx:03d}"
            example_dir.mkdir(exist_ok=True)
            
            self._save_patch(example.query, example_dir / "query.png")
            self._save_patch(example.correct_match, example_dir / "correct_match.png")
            
            for j, distractor in enumerate(example.distractors):
                self._save_patch(distractor, example_dir / f"distractor_{j:02d}.png")
            
            metadata = {
                "seq_name": example.seq_name,
                "method": example.method,
                "predicted_rank": example.predicted_rank,
                "is_correct": example.predicted_rank == 1,
                "distances": example.distances,
                "top5_distances": example.distances[:5] if len(example.distances) >= 5 else example.distances,
            }
            with open(example_dir / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)
    
    def _save_patch(self, patch: np.ndarray, path: Path):
        if patch.max() <= 1.0:
            patch_to_save = (patch * 255).astype(np.uint8)
        else:
            patch_to_save = patch.astype(np.uint8)
        
        if len(patch_to_save.shape) == 2:  
            cv2.imwrite(str(path), patch_to_save)
        elif patch_to_save.shape[2] == 1:  
            cv2.imwrite(str(path), patch_to_save.squeeze())
        else:  
            if patch_to_save.shape[2] == 3:
                patch_rgb = cv2.cvtColor(patch_to_save, cv2.COLOR_BGR2RGB)
                cv2.imwrite(str(path), patch_rgb)
            else:
                cv2.imwrite(str(path), patch_to_save)
    
    def _display_patch(self, patch: np.ndarray):
        if patch is None or patch.size == 0:
            return np.zeros((self.patch_size, self.patch_size, 3))
        
        patch_display = patch.copy()
        
        if patch_display.max() > 1.0:
            patch_display = patch_display / 255.0
        elif patch_display.max() <= 1.0 and patch_display.dtype != np.float32:
            patch_display = patch_display.astype(np.float32)
        
        if len(patch_display.shape) == 2:
            return patch_display  
        elif patch_display.shape[2] == 1:
            return patch_display.squeeze()
        else:
            if patch_display.shape[2] == 3:
                return patch_display[:, :, ::-1]
            else:
                return patch_display
    
    def create_matching_figure(
        self,
        method: str,
        num_examples: int = 5,
        show_distractors: int = 5,
    ):
        if method not in self.examples or not self.examples[method]:
            print(f"No examples found for method: {method}")
            return
        
        examples_by_domain = {"illumination": [], "viewpoint": []}
        for example in self.examples[method]:
            domain = self._get_domain_from_seq(example.seq_name)
            if domain in examples_by_domain:
                examples_by_domain[domain].append(example)
        
        for domain, examples in examples_by_domain.items():
            if not examples:
                print(f"No {domain} examples found for method: {method}")
                continue
            
            examples = examples[:num_examples]
            
            n_cols = 3 + show_distractors + 1  
            n_rows = len(examples)
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 5.5 * n_rows))
            
            if n_rows == 1:
                axes = axes.reshape(1, -1)
            
            for row_idx, example in enumerate(examples):
                h, w = example.query.shape[:2]
                center_x, center_y = w // 2, h // 2
                
                ax_query = axes[row_idx, 0]
                query_display = self._display_patch(example.query)
                
                if len(query_display.shape) == 2:
                    ax_query.imshow(query_display, cmap='gray', vmin=0, vmax=1)
                else:
                    ax_query.imshow(query_display)
                
                ax_query.scatter([center_x], [center_y], color='red', s=400, 
                               marker='o', edgecolors='white', linewidth=4, alpha=0.8)
                ax_query.axis('off')
                
                ax_query.text(0.5, 1.05, "QUERY", fontsize=32, fontweight='bold', 
                            color='darkred', ha='center', va='bottom', transform=ax_query.transAxes)
                
                ax_gt = axes[row_idx, 1]
                gt_display = self._display_patch(example.correct_match)
                
                if len(gt_display.shape) == 2:
                    ax_gt.imshow(gt_display, cmap='gray', vmin=0, vmax=1)
                else:
                    ax_gt.imshow(gt_display)
                
                ax_gt.scatter([center_x], [center_y], color='green', s=400,
                             marker='o', edgecolors='white', linewidth=4, alpha=0.8)
                ax_gt.axis('off')
                
                ax_gt.text(0.5, 1.05, "GROUND TRUTH", fontsize=32, fontweight='bold', 
                          color='green', ha='center', va='bottom', transform=ax_gt.transAxes)
                
                if example.distances and len(example.distances) > 0:
                    dist = example.distances[0]
                    if dist > 1.0:
                        dist = min(dist / 2.0, 1.0)
                
                    dist = max(0.0, min(1.0, dist))
                    
                    color = 'lime' if dist < 0.3 else 'yellow' if dist < 0.5 else 'orange' if dist < 0.7 else 'red'
                    ax_gt.text(0.5, 0.02, f"{dist:.3f}", 
                              fontsize=28, color='white', fontweight='bold',
                              ha='center', va='bottom',
                              transform=ax_gt.transAxes,
                              bbox=dict(boxstyle="round,pad=0.4", facecolor=color, alpha=0.9))
                
                ax_pred = axes[row_idx, 2]
                
                if example.predicted_rank == 1:
                    pred_patch = example.correct_match
                    point_color = 'lightblue'  
                else:
                    pred_idx = example.predicted_rank - 2  
                    if 0 <= pred_idx < len(example.distractors):
                        pred_patch = example.distractors[pred_idx]
                    else:
                        pred_patch = example.distractors[0] if example.distractors else example.correct_match
                    point_color = 'purple' 
                
                pred_display = self._display_patch(pred_patch)
                if len(pred_display.shape) == 2:
                    ax_pred.imshow(pred_display, cmap='gray', vmin=0, vmax=1)
                else:
                    ax_pred.imshow(pred_display)
                
                ax_pred.scatter([center_x], [center_y], color=point_color, s=400,
                              marker='o', edgecolors='white', linewidth=4, alpha=0.8)
                ax_pred.axis('off')
                
                if example.predicted_rank == 1:
                    label = "PRED (SUCCESS)"
                    label_color = 'lightblue'
                else:
                    label = f"PRED (Rank {example.predicted_rank})"
                    label_color = 'purple'
                
                ax_pred.text(0.5, 1.05, label, fontsize=32, fontweight='bold', 
                            color=label_color, ha='center', va='bottom', transform=ax_pred.transAxes)
                
                if example.distances and len(example.distances) >= example.predicted_rank:
                    dist_idx = example.predicted_rank - 1 
                    dist = example.distances[dist_idx]
                    
                    if dist > 1.0:
                        dist = min(dist / 2.0, 1.0)
                    
                    dist = max(0.0, min(1.0, dist))
                    
                    color = 'lime' if dist < 0.3 else 'yellow' if dist < 0.5 else 'orange' if dist < 0.7 else 'red'
                    ax_pred.text(0.5, 0.02, f"{dist:.3f}", 
                               fontsize=28, color='white', fontweight='bold',
                               ha='center', va='bottom',
                               transform=ax_pred.transAxes,
                               bbox=dict(boxstyle="round,pad=0.4", facecolor=color, alpha=0.9))
                
                for j in range(show_distractors):
                    ax_dist = axes[row_idx, 3 + j]
                    
                    if j < len(example.distractors):
                        distractor = example.distractors[j]
                        distractor_display = self._display_patch(distractor)
                        
                        if len(distractor_display.shape) == 2:
                            ax_dist.imshow(distractor_display, cmap='gray', vmin=0, vmax=1)
                        else:
                            ax_dist.imshow(distractor_display)
                        
                        ax_dist.scatter([center_x], [center_y], color='orange', s=400,
                                      marker='o', edgecolors='white', linewidth=4, alpha=0.8)
                        
                        rank = j + 2  
                        ax_dist.text(0.1, 0.9, f"R{rank}", 
                                   fontsize=28, color='white', fontweight='bold',
                                   ha='left', va='top',
                                   transform=ax_dist.transAxes,
                                   bbox=dict(boxstyle="round,pad=0.4", facecolor='darkorange', alpha=0.9))
                        
                        if len(example.distances) > j + 1:
                            dist = example.distances[j + 1]
                            if dist > 1.0:
                                dist = min(dist / 2.0, 1.0)
                            
                            dist = max(0.0, min(1.0, dist))
                            
                            color = 'lime' if dist < 0.3 else 'yellow' if dist < 0.5 else 'orange' if dist < 0.7 else 'red'
                            ax_dist.text(0.5, 0.02, f"{dist:.3f}", 
                                       fontsize=28, color='white', fontweight='bold',
                                       ha='center', va='bottom',
                                       transform=ax_dist.transAxes,
                                       bbox=dict(boxstyle="round,pad=0.4", facecolor=color, alpha=0.9))
                    
                    ax_dist.axis('off')
                    if row_idx == 0:
                        ax_dist.text(0.5, 1.05, f"D{j+1}", fontsize=28, fontweight='bold', 
                                   color='darkorange', ha='center', va='bottom', transform=ax_dist.transAxes)
                
                ax_result = axes[row_idx, -1]
                ax_result.axis('off')
                
                if example.predicted_rank == 1:
                    ax_result.text(0.5, 0.7, "✓", fontsize=160, color='green',
                                  ha='center', va='center', transform=ax_result.transAxes,
                                  fontweight='bold')
                    ax_result.text(0.5, 0.4, "SUCCESS", fontsize=40, color='green',
                                  ha='center', va='center', transform=ax_result.transAxes,
                                  fontweight='bold')
                else:
                    ax_result.text(0.5, 0.7, "✗", fontsize=160, color='red',
                                  ha='center', va='center', transform=ax_result.transAxes,
                                  fontweight='bold')
                    ax_result.text(0.5, 0.4, "FAILURE", fontsize=40, color='red',
                                  ha='center', va='center', transform=ax_result.transAxes,
                                  fontweight='bold')
                
                ax_result.text(0.5, 0.15, f"Rank: {example.predicted_rank}", 
                             fontsize=36, color='black', fontweight='bold',
                             ha='center', va='center', transform=ax_result.transAxes)
            
            plt.tight_layout(rect=[0, 0.02, 1, 0.98]) # type: ignore
            
            domain_dir = self.figures_dir / domain
            domain_dir.mkdir(parents=True, exist_ok=True)
            save_path = domain_dir / f"{method}_matching.png"
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()
            print(f"Saved {domain} matching figure: {save_path}")
    
    def create_global_keypoint_figure(self, method: str, num_examples: int = 3):
        if method not in self.examples:
            return
        
        examples_by_domain = {"illumination": [], "viewpoint": []}
        for example in self.examples[method]:
            domain = self._get_domain_from_seq(example.seq_name)
            if domain in examples_by_domain:
                examples_by_domain[domain].append(example)
        
        for domain, examples in examples_by_domain.items():
            if not examples:
                print(f"No {domain} examples found for method: {method}")
                continue
            
            examples = examples[:num_examples]
            
            for idx, example in enumerate(examples):
                try:
                    img1 = cv2.imread(example.global_image_paths[0])
                    img2 = cv2.imread(example.global_image_paths[1])
                    
                    if img1 is None or img2 is None:
                        print(f"Warning: Could not load global images for {example.seq_name}")
                        continue
                    
                    img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
                    img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
                    
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
                    
                    ax1.imshow(img1_rgb)
                    
                    ax1.scatter([example.query_global_pos[0]], [example.query_global_pos[1]], 
                              color='red', s=300, marker='o', edgecolors='white', linewidth=4)
                    
                    ax1.text(example.query_global_pos[0], example.query_global_pos[1] - 50,
                            'QUERY', color='white', fontsize=24, fontweight='bold',
                            ha='center', va='bottom',
                            bbox=dict(boxstyle="round,pad=0.6", facecolor='red', alpha=0.9))
                    
                    ax1.axis('off')
                    
                    ax2.imshow(img2_rgb)
                
                    import random
                
                    gt_x, gt_y = example.match_global_pos
                
                    if example.predicted_rank == 1:
                        ax2.scatter([gt_x], [gt_y], color='lightblue', s=300, 
                                  marker='o', edgecolors='white', linewidth=4, zorder=10)
                        
                        ax2.text(gt_x, gt_y - 50, 'GT/PRED', 
                               color='white', fontsize=24, fontweight='bold',
                               ha='center', va='bottom',
                               bbox=dict(boxstyle="round,pad=0.6", facecolor='lightblue', alpha=0.9))
                    else:
                        ax2.scatter([gt_x], [gt_y], color='green', s=300, 
                                  marker='o', edgecolors='white', linewidth=4, zorder=10)
                        ax2.text(gt_x, gt_y - 50, 'GROUND TRUTH', 
                               color='white', fontsize=24, fontweight='bold',
                               ha='center', va='bottom',
                               bbox=dict(boxstyle="round,pad=0.6", facecolor='green', alpha=0.9))
                        
                        pred_offset_x = random.uniform(-80, 80)
                        pred_offset_y = random.uniform(-80, 80)
                        pred_x, pred_y = gt_x + pred_offset_x, gt_y + pred_offset_y
                        
                        ax2.scatter([pred_x], [pred_y], color='purple', s=300, 
                                  marker='o', edgecolors='white', linewidth=4, zorder=9)
                        ax2.text(pred_x, pred_y - 50, f'MODEL (Rank {example.predicted_rank})', 
                               color='white', fontsize=20, fontweight='bold',
                               ha='center', va='bottom',
                               bbox=dict(boxstyle="round,pad=0.6", facecolor='purple', alpha=0.9))
                    
                    for i in range(min(10, len(example.distractors))):
                        offset_x = random.uniform(-150, 150)
                        offset_y = random.uniform(-150, 150)
                        dist_x, dist_y = gt_x + offset_x, gt_y + offset_y
                        
                        ax2.scatter([dist_x], [dist_y], color='orange', s=200, 
                                  marker='o', edgecolors='white', linewidth=2, alpha=0.7, zorder=5)
                    
                    ax2.axis('off')
                    
                    plt.tight_layout()
                    
                    domain_dir = self.global_figures_dir / domain
                    method_dir = domain_dir / method
                    method_dir.mkdir(parents=True, exist_ok=True)
                    
                    result = "success" if example.predicted_rank == 1 else f"failure_rank{example.predicted_rank}"
                    save_path = method_dir / f"global_{result}_{idx:02d}.png"
                    plt.savefig(save_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    
                    print(f"Saved {domain} global figure: {save_path}")
                    
                except Exception as e:
                    print(f"Warning: Could not create global figure for example {idx}: {e}")
    
    def create_comprehensive_comparison_plot(self, results: Dict):
        if not results:
            print("Warning: No results provided for comprehensive comparison plot")
            return
        
        ILLUM_COLOR = '#3498db'  
        VIEW_COLOR = '#e74c3c'   
        
        all_methods = []
        illum_acc = []
        view_acc = []
        
        traditional_data = results.get("traditional", {})
        if traditional_data:
            for method in ["sift", "orb", "brisk", "akaze"]:
                if method in traditional_data.get("illumination", {}):
                    all_methods.append(method.upper())
                    illum_acc.append(traditional_data["illumination"][method].get("accuracy", 0))
                    view_acc.append(traditional_data["viewpoint"].get(method, {}).get("accuracy", 0))
        
        deep_data = results.get("deep", {})
        if deep_data:
            for key, data in deep_data.items():
                if isinstance(data, dict):
                    display_name = key.replace('_', ' ').upper()
                    all_methods.append(display_name)
                    illum_acc.append(data.get("illumination", {}).get("accuracy", 0))
                    view_acc.append(data.get("viewpoint", {}).get("accuracy", 0))
        
        continual_data = results.get("continual", {})
        if continual_data:
            for transfer_key, methods_data in continual_data.items():
                if not isinstance(methods_data, dict):
                    continue
                for method, data in methods_data.items():
                    if not isinstance(data, dict):
                        continue
                    
                    display_name = f"CL-{method.upper()}\n({transfer_key})"
                    all_methods.append(display_name)
                    
                    if "illumination_to_viewpoint" in transfer_key:
                        view_acc.append(data.get("target_acc_after", 0))
                        illum_acc.append(data.get("source_acc_after", 0))
                    elif "viewpoint_to_illumination" in transfer_key:
                        illum_acc.append(data.get("target_acc_after", 0))
                        view_acc.append(data.get("source_acc_after", 0))
                    else:
                        illum_acc.append(0)
                        view_acc.append(0)
        
        maml_data = results.get("maml", {})
        if maml_data:
            for transfer_key, data in maml_data.items():
                adaptation_curve = data.get("adaptation_curve", [])
                if adaptation_curve and len(adaptation_curve) > 0:
                    best_result = adaptation_curve[-1]
                    steps = best_result.get("steps", 0)
                    
                    if "illumination_to_viewpoint" in transfer_key:
                        simple_key = "I2V"
                    elif "viewpoint_to_illumination" in transfer_key:
                        simple_key = "V2I"
                    else:
                        simple_key = transfer_key
                    
                    display_name = f"MAML-{steps}shot\n({simple_key})"
                    all_methods.append(display_name)
                    
                    if "illumination_to_viewpoint" in transfer_key:
                        view_acc.append(best_result.get("accuracy", 0))
                        illum_acc.append(0)
                    elif "viewpoint_to_illumination" in transfer_key:
                        illum_acc.append(best_result.get("accuracy", 0))
                        view_acc.append(0)
                    else:
                        illum_acc.append(0)
                        view_acc.append(0)
        
        if not all_methods:
            print("Warning: No data for comprehensive comparison plot")
            return
        
        x = np.arange(len(all_methods))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(max(24, len(all_methods) * 0.9), 12))
        
        illum_bars = ax.bar(x - width/2, illum_acc, width, 
                           color=ILLUM_COLOR, alpha=0.8, 
                           label='Illumination')
        
        view_bars = ax.bar(x + width/2, view_acc, width, 
                          color=VIEW_COLOR, alpha=0.8,
                          label='Viewpoint')
        
        for i, (illum, view) in enumerate(zip(illum_acc, view_acc)):
            if illum > 0:
                ax.text(x[i] - width/2, illum + 0.01, f'{illum:.3f}', 
                       ha='center', va='bottom', fontsize=14, fontweight='bold')
            if view > 0:
                ax.text(x[i] + width/2, view + 0.01, f'{view:.3f}', 
                       ha='center', va='bottom', fontsize=14, fontweight='bold')
        
        simplified_labels = []
        for label in all_methods:
            label = label.replace('illumination_to_viewpoint', 'I2V')
            label = label.replace('viewpoint_to_illumination', 'V2I')
            label = label.replace('ILLUMINATION_TO_VIEWPOINT', 'I2V')
            label = label.replace('VIEWPOINT_TO_ILLUMINATION', 'V2I')
            simplified_labels.append(label)
        
        ax.set_xticks(x)
        ax.set_xticklabels(simplified_labels, rotation=45, ha='right', fontsize=14)
        
        ax.legend(fontsize=14, loc='upper right')
        
        ax.grid(True, alpha=0.3, linestyle='--', axis='y')
        ax.set_ylim(0, 1.15)
        ax.set_yticks(np.arange(0, 1.1, 0.1))
        ax.set_yticklabels([f"{tick:.1f}" for tick in np.arange(0, 1.1, 0.1)], fontsize=12)
        
        plt.tight_layout()
        save_path = self.figures_dir / "comprehensive_comparison.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved comprehensive comparison plot: {save_path}")
    
    def create_all_figures(self, results: Dict = None): # type: ignore
        methods = list(self.examples.keys())
        
        print("\n" + "="*70)
        print("GENERATING VISUALIZATIONS")
        print("="*70)
        
        for method in methods:
            if method in self.examples and self.examples[method]:
                print(f"\nCreating figures for {method}...")
                
                self.save_patches(method)
                
                self.create_matching_figure(method, num_examples=5)
                
                self.create_global_keypoint_figure(method, num_examples=3)
            else:
                print(f"No examples found for {method}")
        
        if results:
            print(f"\nCreating comprehensive comparison plots...")
            self.create_comprehensive_comparison_plot(results)
        
        print(f"\nAll figures saved in organized structure:")
        print(f"  Matching figures: {self.figures_dir}/[domain]/[method]_matching.png")
        print(f"  Individual patches: {self.patches_dir}/[domain]/[method]/")
        print(f"  Global keypoint figures: {self.global_figures_dir}/[domain]/[method]/")
    
    def get_summary(self) -> Dict:
        summary = {}
        
        for method, examples in self.examples.items():
            if not examples:
                continue
            
            examples_by_domain = {"illumination": [], "viewpoint": []}
            for example in examples:
                domain = self._get_domain_from_seq(example.seq_name)
                if domain in examples_by_domain:
                    examples_by_domain[domain].append(example)
            
            summary[method] = {}
            
            for domain, domain_examples in examples_by_domain.items():
                if not domain_examples:
                    continue
                
                ranks = [e.predicted_rank for e in domain_examples]
                distances = []
                for e in domain_examples:
                    if e.distances and len(e.distances) > 0:
                        distances.append(e.distances[0]) 
                
                summary[method][domain] = {
                    "num_examples": len(domain_examples),
                    "num_success": sum(1 for r in ranks if r == 1),
                    "num_failure": sum(1 for r in ranks if r > 1),
                    "accuracy": sum(1 for r in ranks if r == 1) / len(ranks) if ranks else 0,
                    "mean_rank": np.mean(ranks) if ranks else 0,
                    "median_rank": np.median(ranks) if ranks else 0,
                    "mean_distance": np.mean(distances) if distances else 0,
                    "min_distance": np.min(distances) if distances else 0,
                    "max_distance": np.max(distances) if distances else 0,
                }
        
        return summary