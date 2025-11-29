import torch
import yaml
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.meta.maml import MAMLTrainer
from src.models.descriptor_wrapper import DescriptorWrapper
from src.utils.logger import init_wandb, log_metrics
from src.datasets.hpatches_loader import get_dataloader

from src.evaluation.metrics import correspondence_accuracy
from src.evaluation.logging import (
    log_patch_triplets,
    log_tsne,
    log_distance_hist,
)
from src.evaluation.meta_curves import meta_adaptation_curve

def train_meta(config_path):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    wandb = init_wandb(run_name="meta_restore", config=config)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = DescriptorWrapper(pretrained=True).to(device)
    model.load_state_dict(torch.load(config["pretrained_path"], map_location="cpu"))

    maml = MAMLTrainer(
        model,
        lr_inner=config["meta_lr_inner"],
        lr_outer=config["meta_lr_outer"]
    )
    
    loss_fn = torch.nn.TripletMarginLoss(margin=1.0)

    for epoch in range(config["meta_epochs"]):

        # -- build meta-batch --
        
        tasks = []
        
        for domain in config["tasks"]:
        
            loader = get_dataloader(domain, batch_size=4)
            it = iter(loader)

            support = [next(it) for _ in range(2)]
            query = [next(it) for _ in range(2)]
            tasks.append((support, query))

        # -- MAML step --
        
        outer_loss = maml.meta_train_step(tasks, loss_fn)
        log_metrics("meta", {"outer_loss": outer_loss}, step=epoch)
        print(f"Meta Epoch {epoch+1}/{config['meta_epochs']} - Loss: {outer_loss:.4f}")

        # -- evaluation using the first task loader --
        
        eval_loader = get_dataloader(config["tasks"][0], batch_size=8)
        acc, pos_d, neg_d = correspondence_accuracy(model, eval_loader, device)
        wandb.log({"meta/correspondence_accuracy": acc}, step=epoch)

        log_distance_hist(pos_d, neg_d, step=epoch)

        # -- for patch visualisation, use one batch --
        
        a, p, n = next(iter(eval_loader))
        log_patch_triplets(a, p, n, step=epoch)

        # -- TSNE every 2 epochs --
        
        if epoch % 2 == 0:
            log_tsne(model, eval_loader, device, step=epoch)

        # -- meta-adaptation curve --
        
        meta_adaptation_curve(model, eval_loader, loss_fn, config["meta_lr_inner"], device)

    # -- save checkpoint --
    
    torch.save(model.state_dict(), config["save_path"])
    print("Saved meta-learned model to", config["save_path"])

if __name__ == "__main__":
    
    path = sys.argv[sys.argv.index("--config") + 1]
    train_meta(path)