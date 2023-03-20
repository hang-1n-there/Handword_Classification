from copy import deepcopy
import numpy as np

import torch

class Trainer():
    
    def __init__(self, model, optimizer, crit):
        self.model = model
        self.optimizer = optimizer
        self.crit = crit
    
    def _batchify(self, x, y, batch_size, random_split = True):
        if random_split:
            indices = torch.randperm(x.size(0), device=x.device)
            x = torch.index_select(x, dim=0, index=indices)
            y = torch.index_select(y, dim=0, index=indices)
        
        x = x.split(batch_size, dim=0)
        y = y.split(batch_size, dim=0)
        
        return x,y
    
    def _train(self, x, y, config):
        self.model.train()
        
        x, y = self._batchify(x, y, config.batch_size)
        total_loss = 0
        
        for i, (x_i, y_i) in enumerate(zip(x,y)):
            y_hat_i = self.model(x_i)
            loss_i = self.crit(y_hat_i, y_i).squeeze()
          
            self.optimizer.zero_grad()
            loss_i.backward()
           
            self.optimizer.step()
            # verbose = 로그를 얼마나 세세하게 보여줄지에 대한 정도
            if config.verbose >= 2:
                print('Train Iteration {:d}/{:d}, loss : {:.f}'.format(i+1, len(x), float(loss_i)))

            total_loss += float(loss_i)
        
        return total_loss / len(x)
    