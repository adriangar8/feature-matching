"""
Meta-learning script for restoring performance on forgotten domains.

Usage:
    python -m src.training.train_meta_restore --config configs/{}.yaml
"""

import torch
import yaml
import argparse
import os

from src.meta.maml import MAMLTrainer
from src.models.descriptor_wrapper import DescriptorWrapper
from src.utils.logger import init_wandb, log_metrics
from src.datasets.hpatches_loader import get_dataloader
from src.evaluation.metrics import correspondence_accuracy
from src.evaluation.visualization import (
    log_patch_triplets,
    log_tsne,
    log_distance_hist,
)
from src.evaluation.meta_curves import meta_adaptation_curve


def train_meta(config_path):
    
    """Main meta-training function."""
    
    with open(config_path) as f:
        config = yaml.safe_load(f)

    os.makedirs(os.path.dirname(config["save_path"]), exist_ok=True)

    wandb = init_wandb(run_name="meta_restore", config=config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Using device: {device}")

    model = DescriptorWrapper(pretrained=True).to(device)
    
    if os.path.exists(config["pretrained_path"]):
        
        model.load_state_dict(torch.load(config["pretrained_path"], map_location=device))
        
        print(f"Loaded pretrained weights from {config['pretrained_path']}")
    
    else:
        print(f"Warning: Pretrained path {config['pretrained_path']} not found, using ImageNet weights")

    maml = MAMLTrainer(
        model,
        lr_inner=config["meta_lr_inner"],
        lr_outer=config["meta_lr_outer"],
        device=device
    )

    loss_fn = torch.nn.TripletMarginLoss(margin=1.0)

    for epoch in range(config["meta_epochs"]):

        tasks = []

        for domain in config["tasks"]:
            
            loader = get_dataloader(domain, batch_size=config.get("meta_batch_size", 4))
            it = iter(loader)


            try:
                
                support = [next(it) for _ in range(config.get("num_inner_steps", 2))]
                query = [next(it) for _ in range(2)]
                tasks.append((support, query))
                
            except StopIteration:
                
                print(f"Warning: Not enough batches for domain {domain}")
                continue

        if not tasks:
            
            print("No valid tasks, skipping epoch")
            
            continue

        outer_loss = maml.meta_train_step(tasks, loss_fn)
        
        log_metrics("meta", {"outer_loss": outer_loss}, step=epoch)
        
        print(f"Meta Epoch {epoch+1}/{config['meta_epochs']} - Loss: {outer_loss:.4f}")

        eval_loader = get_dataloader(config["tasks"][0], batch_size=8)
        acc, pos_d, neg_d = correspondence_accuracy(model, eval_loader, device)
        
        wandb.log({"meta/correspondence_accuracy": acc}, step=epoch)

        log_distance_hist(pos_d, neg_d, step=epoch)

        a, p, n = next(iter(eval_loader))
        log_patch_triplets(a, p, n, step=epoch)

        if epoch % 2 == 0:
            log_tsne(model, eval_loader, device, step=epoch)

        meta_adaptation_curve(
            model, eval_loader, loss_fn, config["meta_lr_inner"], device
        )

    torch.save(model.state_dict(), config["save_path"])
    print(f"Saved meta-learned model to {config['save_path']}")


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Meta-train descriptor model")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    args = parser.parse_args()
    
    train_meta(args.config)