import torch
from torch.optim.lr_scheduler import StepLR
from torch.utils import data
from model import FashionMnistClassifier
from dataloader import FashionMnist
from tqdm import tqdm
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from augmentation import get_transform_pipeline
from metric import Accuracy,RunningLoss,Precision,Recall,F1Score
from callbacks import CheckpointCallback
from torchsummary import summary
import os
import wandb
from torchvision.transforms import transforms
# from torchsummary import summary
# from thop import profile,clever_format



EXP_NUM=4
CKPT_DIR='checkpoints'
MODEL_NAME='fashion_mnist_'+str(EXP_NUM)+'.pt'

TRAIN_META='Dataset/FashionMnist/fashion-mnist_train.csv'
VALID_META='Dataset/FashionMnist/fashion-mnist_test.csv'
IMG_WIDTH=28
IMG_HEIGHT=28

BATCH_SIZE=128
SHUFFLE=True
NUM_WORKERS=8
PIN_MEMORY=True
EPOCHS=30
LR=3e-3
device='cuda' if torch.cuda.is_available() else 'cpu'
config={
    'experiment_number':EXP_NUM,
    'checkpoint_name':MODEL_NAME,
    'train_meta':TRAIN_META,
    'valid_meta':VALID_META,
    'image_width':IMG_WIDTH,
    'image_height':IMG_HEIGHT,
    'batch_size':BATCH_SIZE,
    'shuffle':SHUFFLE,
    'number_workers':NUM_WORKERS,
    'pin_memory':PIN_MEMORY,
    'device':device,
    'epochs':EPOCHS,
    'learning_rate':LR,
}
wandb.init(name='fashion_mnist_classification_'+str(EXP_NUM),config=config,job_type='classification',project="clothing-articles-classification", entity="hazem45")
transform_dict={
    'train':get_transform_pipeline(IMG_WIDTH,IMG_HEIGHT),
    'valid':get_transform_pipeline(IMG_WIDTH,IMG_HEIGHT,False)
}

train_transform=transforms.Compose([
    transforms.Resize((IMG_WIDTH,IMG_HEIGHT)),
    # transforms.GaussianBlur(3),
    # transforms.RandomHorizontalFlip(),
    # transforms.RandomPerspective(),
    # transforms.RandomInvert(),
    # transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])
valid_transform=transforms.Compose([
    transforms.Resize((IMG_WIDTH,IMG_HEIGHT)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])
train_dataset=FashionMnist(TRAIN_META,train_transform)
valid_dataset=FashionMnist(VALID_META,valid_transform)

train_loader=DataLoader(dataset=train_dataset,batch_size=BATCH_SIZE,shuffle=SHUFFLE,num_workers=NUM_WORKERS,pin_memory=PIN_MEMORY)
valid_loader=DataLoader(dataset=valid_dataset,batch_size=BATCH_SIZE,shuffle=SHUFFLE,num_workers=NUM_WORKERS,pin_memory=PIN_MEMORY)

model=FashionMnistClassifier(train_dataset.n_classes)
model.to(device)

print(summary(model,input_data=(3,IMG_WIDTH,IMG_HEIGHT)))


criterion=CrossEntropyLoss()
optimizer=Adam(params=model.parameters(),lr=LR)
# schedular=StepLR(optimizer=optimizer,step_size=5)
acc_metric=Accuracy()
prec_metric=Precision()
recall_metric=Recall()
f1_metric=F1Score()
running_loss=RunningLoss()
ckpt_callback=CheckpointCallback(os.path.join(CKPT_DIR,MODEL_NAME),'max',verbose=1)

wandb.watch(model,criterion=criterion,log_freq=1,log_graph=True)

for e in range(EPOCHS):
    model.train()
    acc_metric.reset()
    running_loss.reset()
    prec_metric.reset()
    recall_metric.reset()
    f1_metric.reset()
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
        running_loss.update(batch_loss=loss)
        acc_metric.update(y_true=label_batch,y_pred=output)
        prec_metric.update(y_true=label_batch,y_pred=output)
        recall_metric.update(y_true=label_batch,y_pred=output)
        f1_metric.update(y_true=label_batch,y_pred=output)
        iter_loop.set_description('TRAIN LOOP E: '+str(e))
        iter_loop.set_postfix({
            running_loss.name:running_loss.get_value(),
            acc_metric.name:acc_metric.get_value(),
            prec_metric.name:prec_metric.get_value(),
            recall_metric.name:recall_metric.get_value(),
            f1_metric.name:f1_metric.get_value()
            })
    log_dict['loss/train']=running_loss.get_value()
    log_dict['accuracy/train']=acc_metric.get_value()
    log_dict['precision/train']=prec_metric.get_value()
    log_dict['recall/train']=recall_metric.get_value()
    log_dict['f1score/train']=f1_metric.get_value()
    model.eval()
    acc_metric.reset()
    running_loss.reset()
    prec_metric.reset()
    recall_metric.reset()
    f1_metric.reset()
    with torch.no_grad():
        iter_loop=tqdm(enumerate(valid_loader),total=len(valid_loader))
        for ii,(img_batch,label_batch) in iter_loop:
            img_batch=img_batch.cuda()
            label_batch=label_batch.cuda()
            output=model(img_batch)
            loss=criterion(output,label_batch)

            
            # loss+=loss.detach().cpu().numpy()
            running_loss.update(batch_loss=loss)
            acc_metric.update(y_true=label_batch,y_pred=output)
            prec_metric.update(y_true=label_batch,y_pred=output)
            recall_metric.update(y_true=label_batch,y_pred=output)
            f1_metric.update(y_true=label_batch,y_pred=output)
            iter_loop.set_description('VALID LOOP E: '+str(e))
            iter_loop.set_postfix({
                running_loss.name:running_loss.get_value(),
                acc_metric.name:acc_metric.get_value(),
                prec_metric.name:prec_metric.get_value(),
                recall_metric.name:recall_metric.get_value(),
                f1_metric.name:f1_metric.get_value()
            })
        ckpt_callback.check_and_save(model,acc_metric.get_value())   
    # schedular.step()
                 
    log_dict['loss/valid']=running_loss.get_value()
    log_dict['accuracy/valid']=acc_metric.get_value()
    log_dict['precision/valid']=prec_metric.get_value()
    log_dict['recall/valid']=recall_metric.get_value()
    log_dict['f1score/valid']=f1_metric.get_value()
    log_dict['epochs']=e
    # log_dict['lr']=schedular.get_last_lr()[-1]
    wandb.log(log_dict)

wandb.finish()