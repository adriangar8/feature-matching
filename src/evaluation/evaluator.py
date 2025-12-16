"""
Evaluation functions for traditional and deep learning models.
"""

from typing import List, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from ..data.structures import EvalPair, EvalResult
from ..descriptors.traditional import TraditionalExtractor
from ..utils.preprocessing import normalize_patch
from ..utils.visualization import PatchVisualizer


def evaluate_traditional(
    method: str,
    eval_pairs: List[EvalPair],
    max_distractors: int = 100,
    visualizer: Optional[PatchVisualizer] = None,
    save_every: int = 20,
    hpatches_root: str = None, # type: ignore
) -> EvalResult:
    extractor = TraditionalExtractor(method)
    ranks = []
    
    for idx, pair in enumerate(tqdm(eval_pairs, desc=f"Eval {method}")):
        query_desc = extractor.extract(pair.query)
        if query_desc is None:
            continue
        
        correct_desc = extractor.extract(pair.correct_match)
        if correct_desc is None:
            continue
        
        distractors = pair.distractors[:max_distractors]
        distractor_descs = []
        valid_distractors = []
        for d in distractors:
            desc = extractor.extract(d)
            if desc is not None:
                distractor_descs.append(desc)
                valid_distractors.append(d)
        
        if len(distractor_descs) < 5:
            continue
        
        distances = [extractor.distance(query_desc, correct_desc)]
        for d_desc in distractor_descs:
            distances.append(extractor.distance(query_desc, d_desc))
        
        sorted_idx = np.argsort(distances)
        rank = int(np.where(sorted_idx == 0)[0][0]) + 1
        ranks.append(rank)
        
        if visualizer is not None and idx % save_every == 0:
            distractor_order = np.argsort(distances[1:])
            sorted_distractors = [valid_distractors[i] for i in distractor_order[:10] if i < len(valid_distractors)]
            
            global_image_paths = None
            query_pos = None
            match_pos = None
            
            if pair.query_image_path and pair.match_image_path:
                global_image_paths = (pair.query_image_path, pair.match_image_path)
                query_pos = pair.query_keypoint_pos
                match_pos = pair.match_keypoint_pos
            
            visualizer.add_example(
                query=pair.query,
                correct_match=pair.correct_match,
                distractors=sorted_distractors,
                distances=distances,
                predicted_rank=rank,
                seq_name=pair.seq_name,
                method=method,
                query_global_pos=query_pos,
                match_global_pos=match_pos,
                global_image_paths=global_image_paths,
            )
    
    if not ranks:
        return EvalResult(0, 0, 0, float('inf'), float('inf'), 0, 0)
    
    ranks = np.array(ranks)
    return EvalResult(
        accuracy=float(np.mean(ranks == 1)),
        accuracy_top5=float(np.mean(ranks <= 5)),
        accuracy_top10=float(np.mean(ranks <= 10)),
        mean_rank=float(np.mean(ranks)),
        median_rank=float(np.median(ranks)),
        num_queries=len(ranks),
        num_candidates=max_distractors + 1,
    )


def evaluate_deep(
    model: nn.Module,
    eval_pairs: List[EvalPair],
    device: str = "cuda",
    max_distractors: int = 100,
    visualizer: Optional[PatchVisualizer] = None,
    method_name: str = "deep",
    save_every: int = 20,
) -> EvalResult:
    """Evaluate deep learning model with proper normalization."""
    model.eval()
    ranks = []
    
    for idx, pair in enumerate(tqdm(eval_pairs, desc=f"Eval {method_name}")):
        distractors = pair.distractors[:max_distractors]
        all_patches = [pair.query, pair.correct_match] + distractors
        
        tensors = []
        for p in all_patches:
            normalized = normalize_patch(p)
            tensors.append(torch.from_numpy(normalized))
        
        batch = torch.stack(tensors).to(device)
        
        with torch.no_grad():
            descs = model(batch)
            descs = F.normalize(descs, p=2, dim=1)
        
        query_desc = descs[0:1]
        candidate_descs = descs[1:]
        
        similarities = (query_desc @ candidate_descs.T).squeeze(0)
        distances = (1 - similarities).cpu().numpy()
        
        sorted_idx = np.argsort(distances)
        rank = int(np.where(sorted_idx == 0)[0][0]) + 1
        ranks.append(rank)
        
        if visualizer is not None and idx % save_every == 0:
            distractor_distances = distances[1:]
            distractor_order = np.argsort(distractor_distances)
            sorted_distractors = [distractors[i] for i in distractor_order[:10] if i < len(distractors)]
            
            global_image_paths = None
            query_pos = None
            match_pos = None
            
            if pair.query_image_path and pair.match_image_path:
                global_image_paths = (pair.query_image_path, pair.match_image_path)
                query_pos = pair.query_keypoint_pos
                match_pos = pair.match_keypoint_pos
            
            visualizer.add_example(
                query=pair.query,
                correct_match=pair.correct_match,
                distractors=sorted_distractors,
                distances=distances.tolist(),
                predicted_rank=rank,
                seq_name=pair.seq_name,
                method=method_name,
                query_global_pos=query_pos,
                match_global_pos=match_pos,
                global_image_paths=global_image_paths,
            )
    
    if not ranks:
        return EvalResult(0, 0, 0, float('inf'), float('inf'), 0, 0)
    
    ranks = np.array(ranks)
    return EvalResult(
        accuracy=float(np.mean(ranks == 1)),
        accuracy_top5=float(np.mean(ranks <= 5)),
        accuracy_top10=float(np.mean(ranks <= 10)),
        mean_rank=float(np.mean(ranks)),
        median_rank=float(np.median(ranks)),
        num_queries=len(ranks),
        num_candidates=max_distractors + 1,
    )
