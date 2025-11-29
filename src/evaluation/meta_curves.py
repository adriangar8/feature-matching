import torch
from src.evaluation.metrics import correspondence_accuracy
import wandb

def meta_adaptation_curve(model, loader, loss_fn, lr_inner, device, steps=[0,1,2,5,10]):
    
    results = []

    # -- save initial weights --
    init_state = {k: v.clone() for k, v in model.state_dict().items()}

    for s in steps:
        
        # -- restore model state --
        
        model.load_state_dict(init_state)

        # -- take s inner-loop steps --
        
        if s > 0:
            
            for _ in range(s):
                
                for a, p, n in loader:
                    
                    a, p, n = a.to(device), p.to(device), n.to(device)
                    da, dp, dn = model(a), model(p), model(n)
                    loss = loss_fn(da, dp, dn)
                    grad = torch.autograd.grad(loss, model.parameters(), create_graph=False)
                    
                    for param, g in zip(model.parameters(), grad):
                        param.data -= lr_inner * g.data
                    
                    break # only 1 batch per step

        acc, _, _ = correspondence_accuracy(model, loader, device)
        results.append(acc)

    # -- log to wandb --
    
    table = wandb.Table(data=[[s, r] for s, r in zip(steps, results)], columns=["steps", "accuracy"])
    wandb.log({"meta_adaptation_curve": wandb.plot.line(table, "steps", "accuracy", title="Meta-Adaptation Curve")})

    return results