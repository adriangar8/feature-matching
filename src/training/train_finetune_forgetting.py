import torch
import torch.nn as nn
from tqdm import tqdm
import yaml
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.utils.logger import init_wandb, log_metrics
from src.models.descriptor_wrapper import DescriptorWrapper
from src.datasets.hpatches_loader import get_dataloader

from src.evaluation.metrics import correspondence_accuracy
from src.evaluation.logging import (
    log_patch_triplets,
    log_tsne,
    log_distance_hist,
)

def train_finetune(config_path):
    
    with open(config_path) as f:
        config = yaml.safe_load(f)

    wandb = init_wandb(run_name="finetune_forgetting", config=config)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = DescriptorWrapper(pretrained=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["lr"]))
    criterion = nn.TripletMarginLoss(margin=1.0)

    loader = get_dataloader(config["train_domain"], batch_size=config["batch_size"])

    for epoch in range(config["epochs"]):
        
        model.train()
        total_loss = 0

        for a, p, n in tqdm(loader, desc=f"Epoch {epoch+1}"):
            
            a, p, n = a.to(device), p.to(device), n.to(device)

            da = model(a)
            dp = model(p)
            dn = model(n)

            loss = criterion(da, dp, dn)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        
        log_metrics("train", {"loss": avg_loss}, step=epoch)
        
        print(f"Epoch {epoch+1}/{config['epochs']} - Loss: {avg_loss:.4f}")

        # -- evaluation --
        
        acc, pos_d, neg_d = correspondence_accuracy(model, loader, device)
        wandb.log({"train/correspondence_accuracy": acc}, step=epoch)

        log_distance_hist(pos_d, neg_d, step=epoch)

        # -- log a batch of triplets --

        log_patch_triplets(a.cpu(), p.cpu(), n.cpu(), step=epoch)

        # -- t-SNE every 2 epochs --
        
        if epoch % 2 == 0:
            log_tsne(model, loader, device, step=epoch)

    # -- save checkpoint --
    
    torch.save(model.state_dict(), config["save_path"])
    
    print("Saved fine-tuned model to", config["save_path"])

if __name__ == "__main__":
    
    path = sys.argv[sys.argv.index("--config") + 1]
    train_finetune(path)