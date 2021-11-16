from torchvision.models import resnet18
import torch
from torch import nn
# from torchsummary import summary
# from thop import profile,clever_format

class ClothClassifier(nn.Module):
    def __init__(self,num_classes=5) -> None:
        super(ClothClassifier,self).__init__()
        self.backbone=resnet18(pretrained=True,progress=True)
        # print(self.backbone.fc)
        self.num_classes=num_classes
        self.backbone.fc=nn.Sequential(
            nn.Linear(self.backbone.fc.in_features,512),
            nn.Linear(512,512),
            nn.Linear(512,num_classes)
            
        )
        # self.pooling=nn.AvgPool2d()
    def forward(self,x):
        out = self.backbone(x)
        return out
    
    
# model=ClothClassifier(5).cuda()
# input = torch.randn(1, 3, 224, 224).cuda()
# print(summary(model,(3,224,224)))
# macs, params = profile(model, inputs=(input, ))
# macs_params=clever_format([macs, params], "%.3f")
# print(macs,params)