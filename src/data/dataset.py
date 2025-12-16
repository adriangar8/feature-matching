import random
from typing import List, Tuple
import torch
import numpy as np

from ..utils.preprocessing import normalize_patch, augment_patch

class TripletDataset(torch.utils.data.Dataset):    
    def __init__(self, triplets: List[Tuple], augment: bool = True):
        self.triplets = triplets
        self.augment = augment
    
    def __len__(self):
        return len(self.triplets)
    
    def __getitem__(self, idx):
        anchor, positive, negative = self.triplets[idx]
        
        if self.augment:
            if random.random() < 0.3:
                anchor = augment_patch(anchor.copy())
            positive = augment_patch(positive.copy())
            negative = augment_patch(negative.copy())
        
        anchor_t = torch.from_numpy(normalize_patch(anchor))
        positive_t = torch.from_numpy(normalize_patch(positive))
        negative_t = torch.from_numpy(normalize_patch(negative))
        
        return anchor_t, positive_t, negative_t
