import torch
import numpy as np

class Accuracy():
    def __init__(self) -> None:
        self.value=0
        self.num=0
        return 
    def update(self,targets:torch.Tensor,prediciton:torch.Tensor):
        self.num+=1
        targets=targets.detach().cpu()
        prediciton=prediciton.detach().cpu()
        batch_size=targets.shape[0]
        self.value+=(prediciton.argmax(dim=1)==targets).float().mean().item()
        
        return 
    def get_value(self):
        return self.value/self.num
    def reset(self):
        self.value=0
        self.num=0
        
class RunningLoss():
    def __init__(self) -> None:
        self.value=0
        self.num=0
    def update(self,step_loss:torch.Tensor):
        self.num+=1
        self.value+=step_loss.detach().cpu().item()
        return 
    def get_value(self):
        return self.value/self.num
    def reset(self):
        self.value=0
        self.num=0
        return 