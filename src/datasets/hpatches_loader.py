import torch
from torch.utils.data import Dataset, DataLoader
import os, cv2, random
import numpy as np

def warp_point(H, x, y):
    
    pt = np.array([x, y, 1.0])
    x2, y2, z = H @ pt
    
    return x2 / z, y2 / z

def extract_patch(img, x, y, size=32):
    
    half = size // 2
    
    x, y = int(x), int(y)
    h, w = img.shape[:2]
    
    if x-half < 0 or x+half >= w or y-half < 0 or y+half >= h:
        return None
    
    return img[y-half:y+half, x-half:x+half]

class HPatchesTriplets(Dataset):
    
    def __init__(self, root, domain="illumination", size=32):
        
        self.root = root
        self.size = size
        self.domain = domain

        prefix = "i_" if domain == "illumination" else "v_"
        self.sequences = sorted([s for s in os.listdir(root) if s.startswith(prefix)])

        self.items = []

        for seq in self.sequences:
            
            seq_path = os.path.join(root, seq)

            # -- load images --
            
            imgs = [cv2.cvtColor(cv2.imread(os.path.join(seq_path, f)), cv2.COLOR_BGR2RGB)
                    for f in sorted(os.listdir(seq_path)) if f.endswith(".ppm")]

            if len(imgs) < 2:
                continue

            # -- load homographies H_1_k (from img1 to img_k) --
            
            Hs = []
            
            for k in range(2, len(imgs)+1):
                
                Hfile = os.path.join(seq_path, f"H_1_{k}")
                
                if os.path.exists(Hfile):
                    Hs.append(np.loadtxt(Hfile))
                    
                else:
                    Hs.append(np.eye(3)) # illumination sequences -> identity

            # -- detect keypoints in reference image --
            
            sift = cv2.SIFT_create()
            kps, _ = sift.detectAndCompute(imgs[0], None)

            # -- for each keypoint, create pairs with all other images --
            
            for kp in kps:
            
                x1, y1 = kp.pt

                for k, H in enumerate(Hs):
            
                    x2, y2 = warp_point(H, x1, y1)

                    # -- extract patches --
                    
                    a_patch = extract_patch(imgs[0], x1, y1, size=self.size)
                    p_patch = extract_patch(imgs[k+1], x2, y2, size=self.size)

                    if a_patch is None or p_patch is None:
                        continue

                    self.items.append((a_patch, p_patch, seq)) # store seq for negatives

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        
        a_patch, p_patch, seq = self.items[idx]

        # -- negative from different (x, y) --
        # -- sample random patch from same sequence but not correspondence --
        neg_seq = seq
        
        # (you can improve this if needed)
        
        # -- convert to tensor --
        
        to_tensor = lambda x: torch.tensor(x).float().permute(2,0,1) / 255.

        # -- simple negative: choose random item from whole dataset --
        
        neg_idx = random.randint(0, len(self.items)-1)
        n_patch = self.items[neg_idx][0]

        return to_tensor(a_patch), to_tensor(p_patch), to_tensor(n_patch)

def get_dataloader(domain, batch_size=32):
    
    ds = HPatchesTriplets("data/hpatches/hpatches", domain=domain)
    
    return DataLoader(ds, batch_size=batch_size, shuffle=True)