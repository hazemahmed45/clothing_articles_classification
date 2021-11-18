from abc import abstractmethod
import torch
import numpy as np
from sklearn.metrics import recall_score,precision_score,f1_score
class Metric():
    def __init__(self) -> None:
        self.name='metric'
    @abstractmethod
    def update(self,**kwargs):
        return 
    @abstractmethod
    def get_value(self):
        return 
    @abstractmethod
    def reset(self):
        return 
class Accuracy(Metric):
    def __init__(self):
        self.name='Accuracy'
        self.value=0
        self.num=0
        return 
    def update(self,**kwargs):
        self.num+=1
        targets=kwargs['y_true']
        prediciton=kwargs['y_pred']
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
        
class RunningLoss(Metric):
    def __init__(self):
        self.name='Loss'
        self.value=0
        self.num=0
    def update(self,**kwargs):
        self.num+=1
        step_loss=kwargs['batch_loss']
        self.value+=step_loss.detach().cpu().item()
        return 
    def get_value(self):
        return self.value/self.num
    def reset(self):
        self.value=0
        self.num=0
        return 
   
    
class Precision(Metric):
    def __init__(self):
        self.name='Precision'
        # self.tp=0
        # self.fp=0
        self.value=0
        self.num=0
        return 
    def update(self,**kwargs):
        self.num+=1
        y_true=kwargs['y_true']
        y_pred=kwargs['y_pred']
        y_true=y_true.detach().cpu().numpy()
        y_pred=y_pred.detach().cpu().numpy().argmax(axis=1)
        
        # print(y_true)
        # print(y_pred)
        self.value+=precision_score(y_true,y_pred,average='macro',zero_division=0)
        # print(precision_score(y_true,y_pred,average='macro'))
        # exit()
        return 
    def reset(self):
        self.num=0
        self.value=0
        return 
    def get_value(self):
        return self.value/self.num
class Recall(Metric):
    def __init__(self):
        self.name='Recall'
        self.value=0
        self.num=0
        return 
    def update(self, **kwargs):
        self.num+=1
        y_true=kwargs['y_true']
        y_pred=kwargs['y_pred']
        y_true=y_true.detach().cpu().numpy()
        y_pred=y_pred.detach().cpu().numpy().argmax(axis=1)
        self.value+=recall_score(y_true,y_pred,average='macro',zero_division=0)
        return 
    def reset(self):
        self.num=0
        self.value=0
        return 
    def get_value(self):
        return self.value/self.num
class F1Score(Metric):
    def __init__(self):
        self.name='F1Score'
        self.recall=Recall()
        self.precision=Precision()
        return 
    def update(self, **kwargs):
        y_true=kwargs['y_true']
        y_pred=kwargs['y_pred']
        self.recall.update(y_true=y_true,y_pred=y_pred)
        self.precision.update(y_true=y_true,y_pred=y_pred)
        return
    def reset(self):
        self.precision.reset()
        self.recall.reset()
        return
    def get_value(self):
        return 2*((self.precision.get_value()*self.recall.get_value())/(self.precision.get_value()+self.recall.get_value()))
