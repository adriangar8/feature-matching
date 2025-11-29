import torch, higher

class MAMLTrainer:
    
    def __init__(self, model, lr_inner=1e-3, lr_outer=1e-4):
        
        self.model = model
        self.lr_inner = lr_inner
        self.outer_optimizer = torch.optim.Adam(model.parameters(), lr=lr_outer)

    def meta_train_step(self, tasks, loss_fn):
        
        meta_loss = 0.0
        
        for task in tasks:
            
            support, query = task
            
            with higher.innerloop_ctx(self.model, torch.optim.SGD(self.model.parameters(), lr=self.lr_inner)) as (fmodel, diffopt):
                
                for a, p, n in support:
                    
                    loss = loss_fn(fmodel(a), fmodel(p), fmodel(n))
                    diffopt.step(loss)
                    
                query_loss = 0
                
                for a, p, n in query:
                    
                    query_loss += loss_fn(fmodel(a), fmodel(p), fmodel(n))
                    
                meta_loss += query_loss
                
        meta_loss /= len(tasks)
        self.outer_optimizer.zero_grad()
        meta_loss.backward()
        self.outer_optimizer.step()
        
        return meta_loss.item()