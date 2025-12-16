from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from copy import deepcopy
import wandb

from ..data.dataset import TripletDataset


class NaiveFinetuner:
    def __init__(self, model: nn.Module, lr: float = 1e-4, device: str = "cuda"):
        self.model = model
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        self.device = device
    
    def train_step(self, batch: Tuple[torch.Tensor, ...], criterion: nn.Module) -> float:
        anchor, positive, negative = batch
        anchor = anchor.to(self.device)
        positive = positive.to(self.device)
        negative = negative.to(self.device)
        
        self.optimizer.zero_grad()
        
        d_a = self.model(anchor)
        d_p = self.model(positive)
        d_n = self.model(negative)
        
        loss = criterion(d_a, d_p, d_n)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return loss.item()


class EWC:
    def __init__(self, model: nn.Module, lr: float = 1e-4, ewc_lambda: float = 400, device: str = "cuda"):
        self.model = model
        self.ewc_lambda = ewc_lambda
        self.device = device
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        
        self.fisher = {}
        self.optimal_params = {}
    
    def compute_fisher(self, triplets: List[Tuple], criterion: nn.Module, num_samples: int = 500):
        dataset = TripletDataset(triplets[:num_samples], augment=False)
        loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
        
        self.fisher = {
            n: torch.zeros_like(p) 
            for n, p in self.model.named_parameters() 
            if p.requires_grad
        }
        
        self.optimal_params = {
            n: p.clone().detach() 
            for n, p in self.model.named_parameters() 
            if p.requires_grad
        }
        
        self.model.eval()
        
        for anchor, positive, negative in loader:
            anchor = anchor.to(self.device)
            positive = positive.to(self.device)
            negative = negative.to(self.device)
            
            self.model.zero_grad()
            
            d_a = self.model(anchor)
            d_p = self.model(positive)
            d_n = self.model(negative)
            
            loss = criterion(d_a, d_p, d_n)
            loss.backward()
            
            for n, p in self.model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    self.fisher[n] += p.grad.data.clone().pow(2)
        
        for n in self.fisher:
            self.fisher[n] /= len(loader)
    
    def penalty(self) -> torch.Tensor:
        loss = 0
        for n, p in self.model.named_parameters():
            if n in self.fisher:
                loss += (self.fisher[n] * (p - self.optimal_params[n]).pow(2)).sum()
        return self.ewc_lambda * loss # type: ignore
    
    def train_step(self, batch: Tuple[torch.Tensor, ...], criterion: nn.Module) -> float:
        anchor, positive, negative = batch
        anchor = anchor.to(self.device)
        positive = positive.to(self.device)
        negative = negative.to(self.device)
        
        self.optimizer.zero_grad()
        
        d_a = self.model(anchor)
        d_p = self.model(positive)
        d_n = self.model(negative)
        
        task_loss = criterion(d_a, d_p, d_n)
        ewc_loss = self.penalty()
        loss = task_loss + ewc_loss
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return loss.item()


class LwF:
    def __init__(self, model: nn.Module, lr: float = 1e-4, lwf_lambda: float = 1.0, device: str = "cuda"):
        self.model = model
        self.lwf_lambda = lwf_lambda
        self.device = device
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        
        self.old_model = None
    
    def consolidate(self):
        self.old_model = deepcopy(self.model)
        self.old_model.eval()
        for p in self.old_model.parameters():
            p.requires_grad = False
    
    def distillation_loss(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.old_model is None:
            return torch.tensor(0.0, device=self.device)
        
        with torch.no_grad():
            old_outputs = self.old_model(inputs)
        new_outputs = self.model(inputs)
        
        old_norm = F.normalize(old_outputs, p=2, dim=1)
        new_norm = F.normalize(new_outputs, p=2, dim=1)
        
        loss = 1 - (old_norm * new_norm).sum(dim=1).mean()
        return self.lwf_lambda * loss
    
    def train_step(self, batch: Tuple[torch.Tensor, ...], criterion: nn.Module) -> float:
        anchor, positive, negative = batch
        anchor = anchor.to(self.device)
        positive = positive.to(self.device)
        negative = negative.to(self.device)
        
        self.optimizer.zero_grad()
        
        d_a = self.model(anchor)
        d_p = self.model(positive)
        d_n = self.model(negative)
        
        task_loss = criterion(d_a, d_p, d_n)
        distill_loss = self.distillation_loss(anchor)
        loss = task_loss + distill_loss
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return loss.item()


def train_continual(
    model: nn.Module,
    source_triplets: List[Tuple],
    target_triplets: List[Tuple],
    method: str,
    epochs_target: int,
    batch_size: int,
    lr: float,
    device: str,
    log_prefix: str = "",
    ewc_lambda: float = 400,
    lwf_lambda: float = 1.0,
) -> nn.Module:
    criterion = nn.TripletMarginLoss(margin=0.3)
    
    if method == "naive":
        trainer = NaiveFinetuner(model, lr=lr, device=device)
    elif method == "ewc":
        trainer = EWC(model, lr=lr, ewc_lambda=ewc_lambda, device=device)
        trainer.compute_fisher(source_triplets, criterion)
    elif method == "lwf":
        trainer = LwF(model, lr=lr, lwf_lambda=lwf_lambda, device=device)
        trainer.consolidate()
    else:
        trainer = NaiveFinetuner(model, lr=lr, device=device)
    
    dataset = TripletDataset(target_triplets, augment=True)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True,
    )
    
    for epoch in range(epochs_target):
        model.train()
        total_loss = 0
        num_batches = 0
        
        for batch in tqdm(loader, desc=f"Target {epoch+1}/{epochs_target}", leave=False):
            loss = trainer.train_step(batch, criterion)
            total_loss += loss
            num_batches += 1
            
            if wandb.run is not None:
                wandb.log({f"{log_prefix}_{method}_batch_loss": loss})
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        print(f"    Epoch {epoch+1}: loss = {avg_loss:.4f}")
        
        if wandb.run is not None:
            wandb.log({
                f"{log_prefix}_{method}_epoch": epoch + 1,
                f"{log_prefix}_{method}_epoch_loss": avg_loss
            })
    
    return model
