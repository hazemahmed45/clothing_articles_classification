import torch.nn as nn
from receptivefield.pytorch import PytorchReceptiveField
from receptivefield.image import get_default_image
from model import FashionMnistClassifier,ClothClassifier
from dataloader import FashionMnist
from matplotlib import pyplot as plt
import torch
import numpy as np
import os

# dataset=FashionMnist('Dataset/FashionMnist/fashion-mnist_train.csv')
# define model functions
def model_fn() -> nn.Module:
    # model = SimpleVGG(disable_activations=True)
    # model=FashionMnistClassifier(10)
    # model.load_state_dict(torch.load('checkpoints/fashion_mnist_0.pt'))
    model=ClothClassifier(19)
    model.load_state_dict(torch.load('checkpoints/cloth_classifier_resnet18_5.pt'))
    model.eval()
    return model

input_shape = [224, 224, 3]
rf = PytorchReceptiveField(model_fn)
rf_params = rf.compute(input_shape = input_shape)
print(rf_params)
# plot receptive fields
for i in range(3):
    
    rf.plot_rf_grids(
        custom_image=get_default_image(input_shape, name='cat'), 
        figsize=(20, 12), 
        layout=(1,6))
    # plt.show()
    plt.savefig(os.path.join('media','cd_rf_ex_'+str(i)+'.jpg'))