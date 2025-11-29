"""
Continual Learning Baselines

Implements methods to prevent catastrophic forgetting:
- EWC (Elastic Weight Consolidation)
- LwF (Learning without Forgetting)
- SI (Synaptic Intelligence)
- Naive fine-tuning (baseline)

These are compared against MAML for domain adaptation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple
from copy import deepcopy
from tqdm import tqdm


class NaiveFineTuner:
    """
    Naive fine-tuning baseline (no forgetting prevention).
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-4,
        device: str = "cuda",
    ):
        self.model = model
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    def train_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        loss_fn: nn.Module,
    ) -> float:
        """Single training step."""
        self.model.train()
        a, p, n = [x.to(self.device) for x in batch]

        da, dp, dn = self.model(a), self.model(p), self.model(n)
        loss = loss_fn(da, dp, dn)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def train_epoch(
        self,
        loader: DataLoader,
        loss_fn: nn.Module,
    ) -> float:
        """Train for one epoch."""
        total_loss = 0
        for batch in loader:
            total_loss += self.train_step(batch, loss_fn)
        return total_loss / len(loader)


class EWC:
    """
    Elastic Weight Consolidation (Kirkpatrick et al., 2017)
    
    Prevents forgetting by adding a quadratic penalty that discourages
    changes to parameters important for previous tasks.
    
    Args:
        model: Neural network model
        lr: Learning rate
        ewc_lambda: Importance of the EWC penalty
        device: torch device
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-4,
        ewc_lambda: float = 1000.0,
        device: str = "cuda",
    ):
        self.model = model
        self.device = device
        self.ewc_lambda = ewc_lambda
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # Storage for Fisher information and optimal parameters
        self.fisher: Dict[str, torch.Tensor] = {}
        self.optimal_params: Dict[str, torch.Tensor] = {}
        self.task_count = 0

    def compute_fisher(
        self,
        loader: DataLoader,
        loss_fn: nn.Module,
        num_samples: int = 200,
    ) -> None:
        """
        Compute Fisher information matrix (diagonal approximation).
        
        Should be called after training on a task, before moving to next task.
        """
        self.model.eval()
        fisher = {n: torch.zeros_like(p) for n, p in self.model.named_parameters() if p.requires_grad}

        count = 0
        for a, p, n in loader:
            if count >= num_samples:
                break

            a, p, n = a.to(self.device), p.to(self.device), n.to(self.device)

            self.model.zero_grad()
            da, dp, dn = self.model(a), self.model(p), self.model(n)
            loss = loss_fn(da, dp, dn)
            loss.backward()

            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    fisher[name] += param.grad.data.pow(2)

            count += a.size(0)

        # Normalize
        for name in fisher:
            fisher[name] /= count

        # Accumulate Fisher (for multiple tasks)
        if self.task_count == 0:
            self.fisher = fisher
        else:
            for name in fisher:
                self.fisher[name] = (self.fisher[name] * self.task_count + fisher[name]) / (self.task_count + 1)

        # Store optimal parameters
        self.optimal_params = {n: p.data.clone() for n, p in self.model.named_parameters()}
        self.task_count += 1

    def ewc_loss(self) -> torch.Tensor:
        """Compute EWC penalty."""
        if self.task_count == 0:
            return torch.tensor(0.0, device=self.device)

        loss = 0
        for name, param in self.model.named_parameters():
            if name in self.fisher:
                loss += (self.fisher[name] * (param - self.optimal_params[name]).pow(2)).sum()

        return self.ewc_lambda * loss

    def train_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        loss_fn: nn.Module,
    ) -> Tuple[float, float]:
        """Single training step with EWC penalty."""
        self.model.train()
        a, p, n = [x.to(self.device) for x in batch]

        da, dp, dn = self.model(a), self.model(p), self.model(n)
        task_loss = loss_fn(da, dp, dn)
        ewc_penalty = self.ewc_loss()
        total_loss = task_loss + ewc_penalty

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return task_loss.item(), ewc_penalty.item()

    def train_epoch(
        self,
        loader: DataLoader,
        loss_fn: nn.Module,
    ) -> Tuple[float, float]:
        """Train for one epoch."""
        total_task_loss = 0
        total_ewc_loss = 0

        for batch in loader:
            task_loss, ewc_loss = self.train_step(batch, loss_fn)
            total_task_loss += task_loss
            total_ewc_loss += ewc_loss

        n = len(loader)
        return total_task_loss / n, total_ewc_loss / n


class LwF:
    """
    Learning without Forgetting (Li & Hoiem, 2017)
    
    Prevents forgetting by distilling knowledge from the old model
    into the new model during training on new tasks.
    
    Args:
        model: Neural network model
        lr: Learning rate
        lwf_lambda: Weight for distillation loss
        temperature: Softmax temperature for distillation
        device: torch device
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-4,
        lwf_lambda: float = 1.0,
        temperature: float = 2.0,
        device: str = "cuda",
    ):
        self.model = model
        self.device = device
        self.lwf_lambda = lwf_lambda
        self.temperature = temperature
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        self.old_model: Optional[nn.Module] = None

    def consolidate(self) -> None:
        """
        Store current model as the "old" model for distillation.
        
        Should be called after training on a task, before moving to next task.
        """
        self.old_model = deepcopy(self.model)
        self.old_model.eval()
        for param in self.old_model.parameters():
            param.requires_grad = False

    def distillation_loss(
        self,
        new_outputs: torch.Tensor,
        old_outputs: torch.Tensor,
    ) -> torch.Tensor:
        """Compute knowledge distillation loss."""
        # Use cosine similarity loss for descriptor distillation
        return 1 - F.cosine_similarity(new_outputs, old_outputs, dim=1).mean()

    def train_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        loss_fn: nn.Module,
    ) -> Tuple[float, float]:
        """Single training step with distillation."""
        self.model.train()
        a, p, n = [x.to(self.device) for x in batch]

        da, dp, dn = self.model(a), self.model(p), self.model(n)
        task_loss = loss_fn(da, dp, dn)

        distill_loss = torch.tensor(0.0, device=self.device)
        if self.old_model is not None:
            with torch.no_grad():
                old_da = self.old_model(a)
                old_dp = self.old_model(p)

            distill_loss = (
                self.distillation_loss(da, old_da) +
                self.distillation_loss(dp, old_dp)
            ) / 2

        total_loss = task_loss + self.lwf_lambda * distill_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return task_loss.item(), distill_loss.item()

    def train_epoch(
        self,
        loader: DataLoader,
        loss_fn: nn.Module,
    ) -> Tuple[float, float]:
        """Train for one epoch."""
        total_task_loss = 0
        total_distill_loss = 0

        for batch in loader:
            task_loss, distill_loss = self.train_step(batch, loss_fn)
            total_task_loss += task_loss
            total_distill_loss += distill_loss

        n = len(loader)
        return total_task_loss / n, total_distill_loss / n


class SynapticIntelligence:
    """
    Synaptic Intelligence (Zenke et al., 2017)
    
    Similar to EWC but computes importance online during training
    rather than from Fisher information after training.
    
    Args:
        model: Neural network model
        lr: Learning rate
        si_lambda: Importance of the SI penalty
        damping: Small constant for numerical stability
        device: torch device
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-4,
        si_lambda: float = 1.0,
        damping: float = 0.1,
        device: str = "cuda",
    ):
        self.model = model
        self.device = device
        self.si_lambda = si_lambda
        self.damping = damping
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # Initialize tracking variables
        self.W: Dict[str, torch.Tensor] = {}  # Path integral
        self.omega: Dict[str, torch.Tensor] = {}  # Importance
        self.prev_params: Dict[str, torch.Tensor] = {}
        self.init_params: Dict[str, torch.Tensor] = {}

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.W[name] = torch.zeros_like(param)
                self.omega[name] = torch.zeros_like(param)
                self.prev_params[name] = param.data.clone()
                self.init_params[name] = param.data.clone()

    def update_omega(self) -> None:
        """Update importance weights after task."""
        for name, param in self.model.named_parameters():
            if name in self.W:
                delta = param.data - self.init_params[name]
                self.omega[name] += self.W[name] / (delta.pow(2) + self.damping)
                self.W[name].zero_()
                self.init_params[name] = param.data.clone()

    def si_loss(self) -> torch.Tensor:
        """Compute SI penalty."""
        loss = 0
        for name, param in self.model.named_parameters():
            if name in self.omega:
                loss += (self.omega[name] * (param - self.init_params[name]).pow(2)).sum()

        return self.si_lambda * loss

    def train_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        loss_fn: nn.Module,
    ) -> Tuple[float, float]:
        """Single training step with SI."""
        self.model.train()
        a, p, n = [x.to(self.device) for x in batch]

        da, dp, dn = self.model(a), self.model(p), self.model(n)
        task_loss = loss_fn(da, dp, dn)
        si_penalty = self.si_loss()
        total_loss = task_loss + si_penalty

        self.optimizer.zero_grad()
        total_loss.backward()

        # Update path integral before optimizer step
        for name, param in self.model.named_parameters():
            if param.grad is not None and name in self.W:
                self.W[name] -= param.grad * (param.data - self.prev_params[name])
                self.prev_params[name] = param.data.clone()

        self.optimizer.step()

        return task_loss.item(), si_penalty.item()

    def train_epoch(
        self,
        loader: DataLoader,
        loss_fn: nn.Module,
    ) -> Tuple[float, float]:
        """Train for one epoch."""
        total_task_loss = 0
        total_si_loss = 0

        for batch in loader:
            task_loss, si_loss = self.train_step(batch, loss_fn)
            total_task_loss += task_loss
            total_si_loss += si_loss

        n = len(loader)
        return total_task_loss / n, total_si_loss / n


def get_continual_learner(
    method: str,
    model: nn.Module,
    lr: float = 1e-4,
    device: str = "cuda",
    **kwargs,
):
    """Factory function for continual learning methods."""
    methods = {
        "naive": NaiveFineTuner,
        "ewc": EWC,
        "lwf": LwF,
        "si": SynapticIntelligence,
    }

    if method not in methods:
        raise ValueError(f"Unknown method: {method}. Choose from {list(methods.keys())}")

    return methods[method](model, lr=lr, device=device, **kwargs)
