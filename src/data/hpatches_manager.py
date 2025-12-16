import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import cv2
from tqdm import tqdm

from .structures import EvalPair, TSNESample


class HPatchesManager:    
    def __init__(self, root: str, test_ratio: float = 0.2, seed: int = 42):
        self.root = Path(root)
        self.patch_size = 32
        self.test_ratio = test_ratio
        
        if not self.root.exists():
            raise ValueError(f"HPatches root not found: {root}")
        
        all_seqs = sorted([d.name for d in self.root.iterdir() if d.is_dir()])
        self.illum_seqs = [s for s in all_seqs if s.startswith('i_')]
        self.view_seqs = [s for s in all_seqs if s.startswith('v_')]
        
        if not self.illum_seqs and not self.view_seqs:
            raise ValueError(f"No sequences found in {root}")
        
        rng = np.random.RandomState(seed)
        
        illum_shuffled = self.illum_seqs.copy()
        view_shuffled = self.view_seqs.copy()
        rng.shuffle(illum_shuffled)
        rng.shuffle(view_shuffled)
        
        n_illum_test = max(1, int(len(self.illum_seqs) * test_ratio))
        n_view_test = max(1, int(len(self.view_seqs) * test_ratio))
        
        self.splits = {
            "illumination": {
                "train": illum_shuffled[n_illum_test:],
                "test": illum_shuffled[:n_illum_test],
            },
            "viewpoint": {
                "train": view_shuffled[n_view_test:],
                "test": view_shuffled[:n_view_test],
            },
        }
        
        self.detector = cv2.SIFT_create(nfeatures=500) # type: ignore
        
        print(f"HPatchesManager initialized from: {root}")
        print(f"  Illumination: {len(self.splits['illumination']['train'])} train, "
              f"{len(self.splits['illumination']['test'])} test")
        print(f"  Viewpoint: {len(self.splits['viewpoint']['train'])} train, "
              f"{len(self.splits['viewpoint']['test'])} test")
    
    def get_sequences(self, domain: str, split: str) -> List[str]:
        if domain == "both":
            return (self.splits["illumination"][split] + 
                    self.splits["viewpoint"][split])
        return self.splits[domain][split]
    
    def load_sequence(self, seq_name: str) -> Dict:
        seq_dir = self.root / seq_name
        
        ref_img = cv2.imread(str(seq_dir / "1.ppm"), cv2.IMREAD_GRAYSCALE)
        if ref_img is None:
            raise ValueError(f"Could not load reference image from {seq_dir}")
        
        targets = []
        for i in range(2, 7):
            img_path = seq_dir / f"{i}.ppm"
            h_path = seq_dir / f"H_1_{i}"
            
            if img_path.exists() and h_path.exists():
                img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                H = np.loadtxt(str(h_path))
                targets.append({"image": img, "H": H, "idx": i})
        
        return {"ref": ref_img, "targets": targets, "name": seq_name}
    
    def extract_patch(self, img: np.ndarray, x: float, y: float) -> Optional[np.ndarray]:
        h, w = img.shape[:2]
        half = self.patch_size // 2
        
        xi, yi = int(round(x)), int(round(y))
        
        if xi - half < 0 or xi + half > w or yi - half < 0 or yi + half > h:
            return None
        
        patch = img[yi - half:yi + half, xi - half:xi + half]
        
        if patch.shape != (self.patch_size, self.patch_size):
            return None
        
        return patch.astype(np.float32) / 255.0
    
    def create_eval_pairs(
        self,
        sequences: List[str],
        max_pairs_per_seq: int = 100,
        min_distractors: int = 20,
    ) -> List[EvalPair]:
        eval_pairs = []
        
        for seq_name in tqdm(sequences, desc="Creating eval pairs"):
            try:
                seq = self.load_sequence(seq_name)
            except Exception as e:
                print(f"Warning: Could not load {seq_name}: {e}")
                continue
                
            ref_img = seq["ref"]
            kps_ref = self.detector.detect(ref_img)
            if len(kps_ref) < 10:
                continue
            
            for target in seq["targets"]:
                target_img = target["image"]
                H = target["H"]
                target_idx = target["idx"]
                
                kps_target = self.detector.detect(target_img)
                
                all_target_patches = []
                all_target_pts = []
                for kp in kps_target:
                    patch = self.extract_patch(target_img, kp.pt[0], kp.pt[1])
                    if patch is not None:
                        all_target_patches.append(patch)
                        all_target_pts.append(kp.pt)
                
                if len(all_target_patches) < min_distractors:
                    continue
                
                pairs_added = 0
                for kp in kps_ref:
                    if pairs_added >= max_pairs_per_seq:
                        break
                    
                    query = self.extract_patch(ref_img, kp.pt[0], kp.pt[1])
                    if query is None:
                        continue
                    
                    pt_ref = np.array([[kp.pt[0], kp.pt[1]]], dtype=np.float32).reshape(-1, 1, 2)
                    pt_target = cv2.perspectiveTransform(pt_ref, H)[0, 0]
                    
                    correct = self.extract_patch(target_img, pt_target[0], pt_target[1])
                    if correct is None:
                        continue
                    
                    distractors = []
                    for patch, pt in zip(all_target_patches, all_target_pts):
                        dist = np.sqrt((pt[0] - pt_target[0])**2 + (pt[1] - pt_target[1])**2)
                        if dist > 10:
                            distractors.append(patch)
                    
                    if len(distractors) < min_distractors:
                        continue
                    
                    eval_pairs.append(EvalPair(
                        query=query,
                        correct_match=correct,
                        distractors=distractors,
                        seq_name=seq_name,
                        query_image_path=str(self.root / seq_name / "1.ppm"),
                        match_image_path=str(self.root / seq_name / f"{target_idx}.ppm"),
                        query_keypoint_pos=(kp.pt[0], kp.pt[1]),
                        match_keypoint_pos=(pt_target[0], pt_target[1]),
                    ))
                    pairs_added += 1
        
        print(f"Created {len(eval_pairs)} evaluation pairs")
        return eval_pairs
    
    def create_training_triplets(
        self,
        sequences: List[str],
        max_triplets: int = 10000,
        min_negative_distance: float = 50.0,
        use_hardest_negative: bool = True,
    ) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        triplets = []
        
        for seq_name in tqdm(sequences, desc="Creating triplets"):
            if len(triplets) >= max_triplets:
                break
            
            try:
                seq = self.load_sequence(seq_name)
            except Exception:
                continue
                
            ref_img = seq["ref"]
            kps_ref = self.detector.detect(ref_img)
            
            if len(kps_ref) < 20:
                continue
            
            for target in seq["targets"]:
                if len(triplets) >= max_triplets:
                    break
                
                target_img = target["image"]
                H = target["H"]
                
                valid_kps = []
                for kp in kps_ref:
                    anchor = self.extract_patch(ref_img, kp.pt[0], kp.pt[1])
                    if anchor is None:
                        continue
                    
                    pt_target = cv2.perspectiveTransform(
                        np.array([[kp.pt]], dtype=np.float32), H
                    )[0, 0]
                    
                    positive = self.extract_patch(target_img, pt_target[0], pt_target[1])
                    if positive is None:
                        continue
                    
                    valid_kps.append({
                        "anchor": anchor,
                        "positive": positive,
                        "pt_target": pt_target,
                    })
                
                if len(valid_kps) < 10:
                    continue
                
                for i, kp_data in enumerate(valid_kps):
                    if len(triplets) >= max_triplets:
                        break
                    
                    valid_negatives = []
                    for j in range(len(valid_kps)):
                        if i == j:
                            continue
                        
                        dist = np.sqrt(
                            (valid_kps[i]["pt_target"][0] - valid_kps[j]["pt_target"][0])**2 +
                            (valid_kps[i]["pt_target"][1] - valid_kps[j]["pt_target"][1])**2
                        )
                        
                        if dist > min_negative_distance:
                            valid_negatives.append((j, dist))
                    
                    if not valid_negatives:
                        continue
                    
                    if use_hardest_negative:
                        valid_negatives.sort(key=lambda x: x[1])
                        neg_idx = valid_negatives[0][0]
                    else:
                        neg_idx = random.choice(valid_negatives)[0]
                    
                    triplets.append((
                        kp_data["anchor"],
                        kp_data["positive"],
                        valid_kps[neg_idx]["positive"],
                    ))
        
        print(f"Created {len(triplets)} training triplets")
        return triplets
    
    def create_tsne_sample(
        self,
        seq_name: str,
        n_distractors: int = 10,
        target_idx: int = 3
    ) -> Optional[TSNESample]:
        try:
            seq = self.load_sequence(seq_name)
        except Exception:
            return None
        
        ref_img = seq["ref"]
        
        target = None
        for t in seq["targets"]:
            if t["idx"] == target_idx:
                target = t
                break
        
        if target is None:
            target = seq["targets"][0] if seq["targets"] else None
        
        if target is None:
            return None
        
        target_img = target["image"]
        H = target["H"]
        
        kps_ref = self.detector.detect(ref_img)
        kps_target = self.detector.detect(target_img)
        
        if len(kps_ref) < 5 or len(kps_target) < n_distractors + 5:
            return None
        
        half = self.patch_size // 2
        
        for kp in kps_ref:
            qx, qy = int(kp.pt[0]), int(kp.pt[1])
            
            if qx - half < 0 or qx + half > ref_img.shape[1] or qy - half < 0 or qy + half > ref_img.shape[0]:
                continue
            
            query_patch = ref_img[qy-half:qy+half, qx-half:qx+half].astype(np.float32) / 255.0
            
            pt_ref = np.array([[kp.pt[0], kp.pt[1]]], dtype=np.float32).reshape(-1, 1, 2)
            pt_target = cv2.perspectiveTransform(pt_ref, H)[0, 0]
            mx, my = int(pt_target[0]), int(pt_target[1])
            
            if mx - half < 0 or mx + half > target_img.shape[1] or my - half < 0 or my + half > target_img.shape[0]:
                continue
            
            correct_patch = target_img[my-half:my+half, mx-half:mx+half].astype(np.float32) / 255.0
            
            distractor_patches = []
            distractor_positions = []
            
            for kp_t in kps_target:
                dx, dy = int(kp_t.pt[0]), int(kp_t.pt[1])
                
                dist_to_match = np.sqrt((dx - mx)**2 + (dy - my)**2)
                if dist_to_match < 30:
                    continue
                
                if dx - half < 0 or dx + half > target_img.shape[1] or dy - half < 0 or dy + half > target_img.shape[0]:
                    continue
                
                d_patch = target_img[dy-half:dy+half, dx-half:dx+half].astype(np.float32) / 255.0
                distractor_patches.append(d_patch)
                distractor_positions.append((kp_t.pt[0], kp_t.pt[1]))
                
                if len(distractor_patches) >= n_distractors:
                    break
            
            if len(distractor_patches) < n_distractors:
                continue
            
            domain = "illumination" if seq_name.startswith("i_") else "viewpoint"
            
            return TSNESample(
                query_patch=query_patch,
                correct_patch=correct_patch,
                distractor_patches=distractor_patches,
                query_global_img=ref_img,
                match_global_img=target_img,
                query_pos=(qx, qy),
                match_pos=(mx, my),
                distractor_positions=distractor_positions,
                seq_name=seq_name,
                domain=domain
            )
        
        return None
