import torch.nn as nn
from receptivefield.pytorch import PytorchReceptiveField
from receptivefield.image import get_default_image
from model import FashionMnistClassifier,ClothClassifier
import numpy as np
import torch 
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.init as init
import torch.nn.functional as F

from dataloader import FashionMnist
from matplotlib import pyplot as plt
import torch
import numpy as np
import os

def compute_RF_numerical(net,img_np):
    '''
    @param net: Pytorch network
    @param img_np: numpy array to use as input to the networks, it must be full of ones and with the correct
    shape.
    '''
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.fill_(1)
            # m.bias.data.fill_(0)
    net.apply(weights_init)
    img_ = Variable(torch.from_numpy(img_np).float(),requires_grad=True)
    out_cnn=net(img_)
    out_shape=out_cnn.size()
    ndims=len(out_cnn.size())
    grad=torch.zeros(out_cnn.size())
    l_tmp=[]
    for i in np.arange(ndims):
        if i==0 or i ==1:#batch or channel
            l_tmp.append(0)
        else:
            l_tmp.append(out_shape[i]/2)
            
    grad[tuple(l_tmp)]=1
    out_cnn.backward(gradient=grad)
    grad_np=img_.grad[0,0].data.numpy()
    idx_nonzeros=np.where(grad_np!=0)
    RF=[np.max(idx)-np.min(idx)+1 for idx in idx_nonzeros]
    
    return RF

def compute_N(out,f,s):
    return s*(out-1)+f if s>0.5 else ((out+(f-2))/2)+1#

def compute_RF(layers):
    out=1
    for f,s in reversed(layers):
        out=compute_N(out,f,s)
    return out
# # dataset=FashionMnist('Dataset/FashionMnist/fashion-mnist_train.csv')
# # define model functions
# def model_fn() -> nn.Module:
#     # model = SimpleVGG(disable_activations=True)
#     # model=FashionMnistClassifier(10)
#     # model.load_state_dict(torch.load('checkpoints/fashion_mnist_0.pt'))
#     model=ClothClassifier(19)
#     model.load_state_dict(torch.load('checkpoints/cloth_classifier_resnet18_5.pt'))
#     model.eval()
#     return modeltate_dict(torch.load('checkpoints/cloth_classifier_resnet18_5.pt'))

# input_shape = [224, 224, 3]
# rf = PytorchReceptiveField(model_fn)
# rf_params = rf.compute(input_shape = input_shape)
# print(rf_params)
# # plot receptive fields
# for i in range(3):
    
#     rf.plot_rf_grids(
#         custom_image=get_default_image(input_shape, name='cat'), 
#         figsize=(20, 12), 
#         layout=(1,6))
#     # plt.show()
#     plt.savefig(os.path.join('media','cd_rf_ex_'+str(i)+'.jpg'))
    
# model=ClothClassifier(19)
# model.load_state_dict(torch.load('checkpoints/cloth_classifier_resnet18_5.pt'))
# model.eval()
model=FashionMnistClassifier(10)
model.eval()

print(compute_RF_numerical(model,np.ones((1,3,28,28))))
model=ClothClassifier(19)
model.eval()

print(compute_RF_numerical(model,np.ones((1,3,224,224))))
