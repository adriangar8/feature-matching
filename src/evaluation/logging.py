import wandb
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np

# -- 1. VISUALIZE TRIPLET PATCHES --

def log_patch_triplets(a, p, n, step):
    
    imgs = []
    
    for i in range(min(4, a.shape[0])):
        
        trip = torch.cat([a[i], p[i], n[i]], dim=2) # horizontal concat
        
        imgs.append(wandb.Image(trip, caption=f"Triplet {i}"))
        
    wandb.log({"triplets": imgs}, step=step)

# -- 2. TSNE VISUALIZATION --

def log_tsne(model, loader, device, step):
    
    model.eval()
    feats = []
    labels = []

    with torch.no_grad():
        for a, p, n in loader:
            a, p, n = a.to(device), p.to(device), n.to(device)
            fa = model(a).cpu().numpy()
            fp = model(p).cpu().numpy()
            fn = model(n).cpu().numpy()

            feats.append(fa); labels += [0]*len(fa)
            feats.append(fp); labels += [1]*len(fp)
            feats.append(fn); labels += [2]*len(fn)

    X = np.concatenate(feats)
    L = np.array(labels)

    tsne = TSNE(n_components=2, perplexity=30)
    X2 = tsne.fit_transform(X)

    plt.figure(figsize=(6, 6))
    plt.scatter(X2[:,0], X2[:,1], c=L, cmap="viridis", s=3)
    plt.title("t-SNE Embeddings")
    wandb.log({"tsne_embeddings": wandb.Image(plt)}, step=step)
    plt.close()

# -- 3. POS/NEG DISTANCE DISTRIBUTIONS --

def log_distance_hist(pos_dists, neg_dists, step):
    
    plt.figure(figsize=(6, 4))
    plt.hist(pos_dists, bins=40, alpha=0.6, label="Positive")
    plt.hist(neg_dists, bins=40, alpha=0.6, label="Negative")
    plt.legend()
    plt.title("Descriptor Distance Distribution")
    wandb.log({"distance_hist": wandb.Image(plt)}, step=step)
    plt.close()