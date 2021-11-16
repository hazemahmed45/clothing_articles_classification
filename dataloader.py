from re import split
import torch
from torch.utils.data import Dataset
import os
import pandas as pd
import json
from sklearn.model_selection import train_test_split,KFold
from augmentation import get_transform_pipeline
import cv2


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
        print(self.cat_to_label)
        return 
    def __getitem__(self, index):
        img_name=self.dataframe.iloc[index]['image']+'.jpg'
        label_cat=self.dataframe.iloc[index]['label']
        # print(label_cat)
        is_kid=self.dataframe.iloc[index]['kids']
        label=self.cat_to_label[label_cat]
        print(label)
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
    
transform_dict={
    'train':get_transform_pipeline(224,224),
    'valid':get_transform_pipeline(224,224,False)
}
splitter=ClothDatasetSplitter('Dataset/images_compressed','Dataset/images.csv',transforms_dict=transform_dict)
for train_dataset,valid_dataset in splitter.generate_train_valid_dataset():
    print(len(train_dataset),len(valid_dataset))
    img,label=train_dataset[0]
    print(img.shape , label)
    pass