from pandas.core.algorithms import mode
from torchvision.models import resnet18
import torch
from torch import nn

class ClothClassifier(nn.Module):
    def __init__(self,num_classes=5) -> None:
        super(ClothClassifier,self).__init__()
        self.in_layer=nn.Conv2d(1,3,3,padding=1)
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
        self.feature_maps=[]
        for module in self.backbone.modules():
            module.register_forward_hook(self.hook_fn_forward)
    def forward(self,x):

        out = self.backbone(x)
        self.feature_maps=self.feature_maps[:10]
        return out
    def hook_fn_forward(self,module,input,output):
        if(isinstance(module,nn.Conv2d)):
            self.feature_maps.append(output)
            return






def conv_block(in_filters,out_filters,padding=1,stride=1,kernel=3,dropout_rate=0.2,with_batch_norm=True,name='conv_block'):
    block=nn.Sequential()
    block.add_module(name=name+'_conv',module=nn.Conv2d(in_filters,out_filters,kernel,stride,padding))
    block.add_module(name=name+'_act',module=nn.ReLU())
    if(with_batch_norm):
        block.add_module(name=name+'_batchnorm',module=nn.BatchNorm2d(out_filters))
    if(dropout_rate!=0):
        block.add_module(name=name+'_dropout',module=nn.Dropout2d(dropout_rate))
    return block
class FashionMnistClassifier(nn.Module):
    def __init__(self,n_classes) -> None:
        super(FashionMnistClassifier,self).__init__()
        self.n_classes=n_classes
        self.conv_1=conv_block(3,32,name='conv_block_1')
        self.conv_2=conv_block(32,32,name='conv_block_2')
        self.pooling=nn.AvgPool2d(kernel_size=2)
        self.conv_3=conv_block(32,64,name='conv_block_3')
        self.conv_4=conv_block(64,64,name='conv_block_4')
        self.conv_5=conv_block(64,64,name='conv_block_5')
        self.conv_6=conv_block(64,64,name='conv_block_6')
        
        self.classifier=nn.Sequential(
            nn.AvgPool2d(7),
            nn.Flatten(),
            nn.Linear(64,32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32,self.n_classes)
        )
        return 
    def forward(self,x):
        self.feature_maps=[]
        # print(x.shape)
        x=self.conv_1(x)
        self.feature_maps.append(x)
        x=self.conv_2(x)
        self.feature_maps.append(x)
        x=self.pooling(x)
        # print(x.shape)
        x=self.conv_3(x)
        self.feature_maps.append(x)
        x=self.conv_4(x)
        self.feature_maps.append(x)
        x=self.pooling(x)
        # print(x.shape)
        x=self.conv_5(x)
        self.feature_maps.append(x)
        x=self.conv_6(x)
        self.feature_maps.append(x)
        # print(x.shape)
        x=self.classifier(x)
        return x
