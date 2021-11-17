from torchvision.models import resnet18
import torch
from torch import nn

class ClothClassifier(nn.Module):
    def __init__(self,num_classes=5) -> None:
        super(ClothClassifier,self).__init__()
        self.backbone=resnet18(pretrained=True,progress=True)
        self.num_classes=num_classes
        self.backbone.fc=nn.Sequential(
            nn.Linear(self.backbone.fc.in_features,512),
            nn.BatchNorm1d(512),
            nn.Dropout(0.4),
            nn.Linear(512,512),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            nn.Linear(512,num_classes)
        )

    def forward(self,x):
        out = self.backbone(x)
        return out
    
    
# model=ClothClassifier(5).cuda()
# input = torch.randn(1, 3, 224, 224).cuda()
# print(summary(model,(3,224,224)))
# macs, params = profile(model, inputs=(input, ))
# macs_params=clever_format([macs, params], "%.3f")
# print(macs,params)