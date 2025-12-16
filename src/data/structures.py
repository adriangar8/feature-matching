from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np

@dataclass
class EvalPair:
    query: np.ndarray
    correct_match: np.ndarray
    distractors: List[np.ndarray]
    seq_name: str
    query_image_path: Optional[str] = None
    match_image_path: Optional[str] = None
    query_keypoint_pos: Optional[Tuple[float, float]] = None
    match_keypoint_pos: Optional[Tuple[float, float]] = None


@dataclass
class EvalResult:
    accuracy: float
    accuracy_top5: float
    accuracy_top10: float
    mean_rank: float
    median_rank: float
    num_queries: int
    num_candidates: int


@dataclass 
class TSNESample:
    query_patch: np.ndarray
    correct_patch: np.ndarray
    distractor_patches: List[np.ndarray]
    query_global_img: np.ndarray
    match_global_img: np.ndarray
    query_pos: Tuple[float, float]
    match_pos: Tuple[float, float]
    distractor_positions: List[Tuple[float, float]]
    seq_name: str
    domain: str
