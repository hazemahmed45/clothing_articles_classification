# from re import split
import torch
from torch.utils import data
from torch.utils.data import Dataset
import os
import pandas as pd
import json
from sklearn.model_selection import train_test_split,KFold
from augmentation import get_transform_pipeline
import cv2
import numpy as np

CATEGORY_TO_LABEL_PATH='categories_to_label.json'
LABEL_TO_CATEGORY_PATH='label_to_categories.json'
class ClothDataset(Dataset):
    def __init__(self,img_dir:str,dataframe:pd.DataFrame,transforms=None):
        super().__init__()
        self.img_dir=img_dir
        self.dataframe=dataframe
        self.transform=transforms
        with open (CATEGORY_TO_LABEL_PATH,'r') as f:
            self.cat_to_label=json.load(f)
        # print(self.cat_to_label)
        return 
    def __getitem__(self, index):
        img_name=self.dataframe.iloc[index]['image']+'.jpg'
        label_cat=self.dataframe.iloc[index]['label']
        # print(label_cat)
        is_kid=self.dataframe.iloc[index]['kids']
        label=self.cat_to_label[label_cat]
        img_path=os.path.join(self.img_dir,img_name)
        img=cv2.imread(img_path)
        if(self.transform is not None):
            img=self.transform(image=img)['image']

        # print(label,self.cat_to_label)
        return img.float(),torch.tensor(label,dtype=torch.int64)
    def __len__(self):
        return self.dataframe.shape[0]
    
    
class ClothDatasetSplitter():
    def __init__(self,img_dir,meta_csv:str,transforms_dict:dict):
        self.img_dir=img_dir
        self.data_df=pd.read_csv(meta_csv)
        self.transform_dict=transforms_dict
        self.data_df.drop(columns='sender_id',inplace=True)
        self.data_df['label']=self.data_df['label'].astype('category')
        # print(self.data_df)
        self.cat_to_label={key:i for i,key in enumerate(self.data_df['label'].cat.categories)}
        self.label_to_cat=dict(enumerate(self.data_df['label'].cat.categories))
        if(not os.path.exists(CATEGORY_TO_LABEL_PATH)):
            with open(CATEGORY_TO_LABEL_PATH,'w') as f:
                json.dump(self.cat_to_label,f)
        else:
            with open (CATEGORY_TO_LABEL_PATH,'r') as f:
                self.cat_to_label=json.load(f)
        if(not os.path.exists(LABEL_TO_CATEGORY_PATH)):
            with open(LABEL_TO_CATEGORY_PATH,'w') as f:
                json.dump(self.label_to_cat,f)
        else:
            with open(LABEL_TO_CATEGORY_PATH,'r') as f:
                self.label_to_cat=json.load(f)
        self.X,self.Y=self.data_df[['image','kids']],self.data_df['label']
        self.k_folder=KFold()
        # print("HERE")
        return 
    def generate_train_valid_dataset(self,train_split=0.8):

        for train_idx,valid_idx in self.k_folder.split(self.data_df):
            yield ClothDataset(self.img_dir,self.data_df.iloc[train_idx],self.transform_dict.get('train',None)),ClothDataset(self.img_dir,self.data_df.iloc[valid_idx],self.transform_dict.get('valid',None))
    def get_classes_num(self):
        return len(self.cat_to_label.keys())
    
    
    
class FashionMnist(Dataset):
    def __init__(self,images_csv) -> None:
        super().__init__()
        self.df=pd.read_csv(images_csv)
        self.n_classes=self.df['label'].nunique()
        return 
    def __getitem__(self, index):
        label=self.df.iloc[index,0]
        img_raveled=self.df.iloc[index,1:].to_numpy()
        img_size=np.int(np.sqrt(img_raveled.shape[0]))
        img=np.float32(np.reshape(img_raveled,(img_size,img_size)))
        # print(img_raveled)
        img/=255.0
        # print(img_raveled.shape,img.shape)
        return torch.tensor(img).float(),torch.tensor(label).long()
    def __len__(self):
        return self.df.shape[0]


# dataset=FashionMnist('Dataset/FashionMnist/fashion-mnist_train.csv')
# print(dataset[0])
# img,label=dataset[0]
# print(img.shape,label)
# print(dataset.n_classes)