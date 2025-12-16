from typing import List, Optional, Tuple
import torch
import torch.nn as nn
from tqdm import tqdm
from copy import deepcopy
import wandb

from ..data.dataset import TripletDataset
from ..data.structures import EvalPair
from ..evaluation.evaluator import evaluate_deep


def train_model(
    model: nn.Module,
    triplets: List[Tuple],
    epochs: int,
    batch_size: int,
    lr: float,
    device: str,
    log_prefix: str = "",
    val_pairs: Optional[List[EvalPair]] = None,
) -> nn.Module:
    
    dataset = TripletDataset(triplets, augment=True)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True,
    )
    
    model = model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    num_training_steps = epochs * len(loader)
    num_warmup_steps = min(500, num_training_steps // 10)
    
    def lr_lambda(step):
        if step < num_warmup_steps:
            return float(step) / float(max(1, num_warmup_steps))
        return max(0.1, 1.0 - (step - num_warmup_steps) / (num_training_steps - num_warmup_steps))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    criterion = nn.TripletMarginLoss(margin=0.3, p=2)
    
    best_val_acc = 0
    best_model_state = None
    global_step = 0
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        num_batches = 0
        num_nonzero_loss = 0
        
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        for anchor, positive, negative in pbar:
            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)
            
            optimizer.zero_grad()
            
            d_a = model(anchor)
            d_p = model(positive)
            d_n = model(negative)
            
            loss = criterion(d_a, d_p, d_n)
            
            if loss.item() > 1e-6:
                num_nonzero_loss += 1
            
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            num_batches += 1
            global_step += 1
            
            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                lr=f"{scheduler.get_last_lr()[0]:.2e}",
                nonzero=f"{num_nonzero_loss}/{num_batches}"
            )
            
            if wandb.run is not None:
                wandb.log({f"{log_prefix}_batch_loss": loss.item()})
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        nonzero_ratio = num_nonzero_loss / num_batches if num_batches > 0 else 0
        
        print(f"  Epoch {epoch+1}: loss = {avg_loss:.4f}, nonzero = {nonzero_ratio:.1%}, lr = {scheduler.get_last_lr()[0]:.2e}")
        
        if wandb.run is not None:
            wandb.log({
                f"{log_prefix}_epoch": epoch + 1,
                f"{log_prefix}_epoch_loss": avg_loss,
                f"{log_prefix}_nonzero_ratio": nonzero_ratio,
            })
        
        if val_pairs is not None and (epoch + 1) % 5 == 0:
            val_result = evaluate_deep(model, val_pairs[:500], device, max_distractors=50)
            print(f"    Val: Acc = {val_result.accuracy:.4f}, Top-5 = {val_result.accuracy_top5:.4f}")
            
            if val_result.accuracy > best_val_acc:
                best_val_acc = val_result.accuracy
                best_model_state = deepcopy(model.state_dict())
        
        if epoch >= 2 and nonzero_ratio < 0.01:
            print(f"  WARNING: Model may be collapsing!")
    
    if best_model_state is not None:
        print(f"  Loading best model (val_acc = {best_val_acc:.4f})")
        model.load_state_dict(best_model_state)
    
    return model
