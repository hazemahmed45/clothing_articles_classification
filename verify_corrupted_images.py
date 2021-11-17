import os
from PIL import Image
import pandas as pd

img_dir='Dataset/images_compressed'
meta_csv='Dataset/images.csv'

df=pd.read_csv(meta_csv)
df_copy=df.copy()
print(df_copy.shape)
for idx,row in df_copy.iterrows():
    image,_,_,_=row
    img_path=os.path.join(img_dir,image+'.jpg')
    try:
        img=Image.open(img_path)
        img.verify()
    except:
        df.drop(index=idx,inplace=True)
        print(idx,image)
        continue
    # break
print(df.shape)
df.to_csv(meta_csv,index=None)