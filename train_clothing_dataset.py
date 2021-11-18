import torch
from torch.optim.lr_scheduler import StepLR
from torch.utils import data
from model import ClothClassifier,FashionMnistClassifier
from dataloader import ClothDatasetSplitter,FashionMnist
from tqdm import tqdm
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from augmentation import get_transform_pipeline
from metric import Accuracy,RunningLoss
from callbacks import CheckpointCallback
from torchsummary import summary
import os
import wandb
from torchvision.transforms import transforms
# from torchsummary import summary
# from thop import profile,clever_format



EXP_NUM=11
CKPT_DIR='checkpoints'
MODEL_NAME='cloth_classifier_resnet18_'+str(EXP_NUM)+'.pt'
IMG_DIR='Dataset/KaggleClothing/images_compressed'
IMG_META_DIR='Dataset/KaggleClothing/images.csv'
# TRAIN_META='Dataset/FashionMnist/fashion-mnist_train.csv'
# VALID_META='Dataset/FashionMnist/fashion-mnist_test.csv'
# IMG_WIDTH=28
# IMG_HEIGHT=28
IMG_WIDTH=224
IMG_HEIGHT=224
BATCH_SIZE=64
SHUFFLE=True
NUM_WORKERS=8
PIN_MEMORY=True
EPOCHS=30
LR=1e-4
TRAIN_SPLIT=0.7
device='cuda' if torch.cuda.is_available() else 'cpu'
config={
    'experiment_number':EXP_NUM,
    'checkpoint_name':MODEL_NAME,
    # 'train_meta':TRAIN_META,
    # 'valid_meta':VALID_META,
    'image_dir':IMG_DIR,
    'img_meta_file':IMG_META_DIR,
    'image_width':IMG_WIDTH,
    'image_height':IMG_HEIGHT,
    'batch_size':BATCH_SIZE,
    'shuffle':SHUFFLE,
    'number_workers':NUM_WORKERS,
    'pin_memory':PIN_MEMORY,
    'device':device,
    'epochs':EPOCHS,
    'learning_rate':LR,
    'train_split':TRAIN_SPLIT
}
wandb.init(name='cloth_classification_resnet18_'+str(EXP_NUM),config=config,job_type='classification',project="clothing-articles-classification", entity="hazem45")
transform_dict={
    'train':get_transform_pipeline(IMG_WIDTH,IMG_HEIGHT),
    'valid':get_transform_pipeline(IMG_WIDTH,IMG_HEIGHT,False)
}
dataset_splitter=ClothDatasetSplitter(img_dir=IMG_DIR,meta_csv=IMG_META_DIR,transforms_dict=transform_dict)
train_dataset,valid_dataset=next(iter(dataset_splitter.generate_train_valid_dataset(TRAIN_SPLIT)))
train_transform=transforms.Compose([
    transforms.Resize((IMG_WIDTH,IMG_HEIGHT)),
    transforms.GaussianBlur(3),
    transforms.RandomHorizontalFlip(),
    transforms.RandomPerspective(),
    transforms.RandomInvert(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])
valid_transform=transforms.Compose([
    transforms.Resize((IMG_WIDTH,IMG_HEIGHT)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])
# train_dataset=FashionMnist(TRAIN_META,train_transform)
# valid_dataset=FashionMnist(VALID_META,valid_transform)

train_loader=DataLoader(dataset=train_dataset,batch_size=BATCH_SIZE,shuffle=SHUFFLE,num_workers=NUM_WORKERS,pin_memory=PIN_MEMORY)
valid_loader=DataLoader(dataset=valid_dataset,batch_size=BATCH_SIZE,shuffle=SHUFFLE,num_workers=NUM_WORKERS,pin_memory=PIN_MEMORY)

model=ClothClassifier(dataset_splitter.get_classes_num())
# model=FashionMnistClassifier(train_dataset.n_classes)
model.to(device)

print(summary(model,input_data=(3,IMG_WIDTH,IMG_HEIGHT)))


criterion=CrossEntropyLoss()
optimizer=Adam(params=model.parameters(),lr=LR)
# schedular=StepLR(optimizer=optimizer,step_size=5)
metric=Accuracy()
running_loss=RunningLoss()
ckpt_callback=CheckpointCallback(os.path.join(CKPT_DIR,MODEL_NAME),'max',verbose=1)

wandb.watch(model,criterion=criterion,log_freq=1,log_graph=True)

for e in range(EPOCHS):
    model.train()
    metric.reset()
    running_loss.reset()
    log_dict={}
    iter_loop=tqdm(enumerate(train_loader),total=len(train_loader))
    # running_loss=0
    for ii,(img_batch,label_batch) in iter_loop:
        optimizer.zero_grad(set_to_none=True)
        img_batch=img_batch.cuda()
        label_batch=label_batch.cuda()
        output=model(img_batch)
        # print(output.shape,label_batch.shape)
        loss=criterion(output,label_batch)
        loss.backward()
        optimizer.step()
        
        # running_loss+=loss.detach().cpu().item()
        running_loss.update(loss)
        metric.update(label_batch,output)
        iter_loop.set_description('TRAIN LOOP E: '+str(e))
        iter_loop.set_postfix({'LOSS':running_loss.get_value(),"ACC":metric.get_value()})
    log_dict['loss/train']=running_loss.get_value()
    log_dict['accuracy/train']=metric.get_value()
    model.eval()
    metric.reset()
    running_loss.reset()
    with torch.no_grad():
        iter_loop=tqdm(enumerate(valid_loader),total=len(valid_loader))
        for ii,(img_batch,label_batch) in iter_loop:
            img_batch=img_batch.cuda()
            label_batch=label_batch.cuda()
            output=model(img_batch)
            loss=criterion(output,label_batch)

            
            # loss+=loss.detach().cpu().numpy()
            running_loss.update(loss)
            metric.update(label_batch,output)
            iter_loop.set_description('VALID LOOP E: '+str(e))
            iter_loop.set_postfix({'LOSS':running_loss.get_value(),"ACC":metric.get_value()})
        ckpt_callback.check_and_save(model,metric.get_value())   
    # schedular.step()
                 
    log_dict['loss/valid']=running_loss.get_value()
    log_dict['accuracy/valid']=metric.get_value()
    log_dict['epochs']=e
    # log_dict['lr']=schedular.get_last_lr()[-1]
    wandb.log(log_dict)

wandb.finish()