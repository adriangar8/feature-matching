import torch
import numpy as np
import cv2

# -- 1. DESCRIPTOR ACCURACY --

def correspondence_accuracy(model, loader, device, threshold=None):
    
    model.eval()
    
    correct = 0
    total = 0
    
    pos_dists = []
    neg_dists = []

    with torch.no_grad():
        
        for a, p, n in loader:
            
            a, p, n = a.to(device), p.to(device), n.to(device)

            da = model(a)
            dp = model(p)
            dn = model(n)

            dp_dist = torch.norm(da - dp, dim=1).cpu().numpy()
            dn_dist = torch.norm(da - dn, dim=1).cpu().numpy()

            pos_dists.extend(dp_dist.tolist())
            neg_dists.extend(dn_dist.tolist())

            correct += (dp_dist < dn_dist).sum()
            total += len(dp_dist)

    accuracy = correct / total
    
    return accuracy, np.array(pos_dists), np.array(neg_dists)

# -- 2. SIFT BASELINE --

def sift_descriptor(patch):
    
    # patch: numpy (H,W,3)
    
    gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
    sift = cv2.SIFT_create()
    
    kp = cv2.KeyPoint(16, 16, 16)
    _, desc = sift.compute(gray, [kp])
    
    if desc is None:
        return None
    
    d = desc[0]
    
    return d / np.linalg.norm(d)

def sift_accuracy(loader):
    
    correct = 0
    total = 0

    for a, p, n in loader:
        
        B = a.shape[0]
        
        for i in range(B):
            
            A = a[i].permute(1,2,0).numpy() * 255
            P = p[i].permute(1,2,0).numpy() * 255
            N = n[i].permute(1,2,0).numpy() * 255

            da = sift_descriptor(A.astype(np.uint8))
            dp = sift_descriptor(P.astype(np.uint8))
            dn = sift_descriptor(N.astype(np.uint8))
            
            if da is None or dp is None or dn is None:
                continue

            dp_dist = np.linalg.norm(da - dp)
            dn_dist = np.linalg.norm(da - dn)

            if dp_dist < dn_dist:
                correct += 1
                
            total += 1

    return correct / total if total > 0 else 0