from model import FashionMnistClassifier,ClothClassifier
import torch
from torchsummary import summary
from thop import profile,clever_format




model=ClothClassifier(20).cuda()
input = torch.randn(1, 3, 224, 224).cuda()
# print(summary(model,(3,224,224)))
macs, params = profile(model, inputs=(input, ))
macs_params=clever_format([macs, params], "%.3f")
print(str(macs/1e9),"G",str(params/1e6),"M")

input=torch.randn((1,3,28,28)).cuda()
model=FashionMnistClassifier(10).cuda()
# print(summary(model,(3,28,28)))
macs,params=profile(model,inputs=(input,))
print(str(macs/1e9),"G",str(params/1e6),"M")
