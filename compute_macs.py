from model import FashionMnistClassifier,ClothClassifier
import torch
from torchsummary import summary
from thop import profile,clever_format




model=ClothClassifier(5).cuda()
input = torch.randn(1, 3, 224, 224).cuda()
print(summary(model,(3,224,224)))
macs, params = profile(model, inputs=(input, ))
macs_params=clever_format([macs, params], "%.3f")
# print(macs,params)
model=FashionMnistClassifier(10)
print(model(torch.randn((1,1,28,28))).shape)