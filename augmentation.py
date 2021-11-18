from albumentations import (
    Compose, ShiftScaleRotate, RandomBrightness, HueSaturationValue, RandomContrast, HorizontalFlip,
    Rotate, Resize, Lambda, CLAHE, ColorJitter, RandomBrightnessContrast, GaussianBlur, Blur, MedianBlur,
    GridDistortion, Downscale, ChannelShuffle, Normalize, OneOf, IAAAdditiveGaussianNoise, GaussNoise,
    RandomScale, ISONoise
)
from albumentations import ImageOnlyTransform
from albumentations.pytorch import ToTensorV2
from PIL import Image
import numpy as np

class GrayToRGB(ImageOnlyTransform):
    def __init__(self, always_apply: bool = False, p: float = 0.5):
        super().__init__(always_apply=always_apply, p=p)
        
    @property
    def targets_as_params(self):
        return ["image"]

    def apply(self, img, **params):
        img=Image.fromarray(img)
        img.convert('rgb')
        return np.array(img)

    # def get_params_dependent_on_targets(self, params):
    #     img = params["image"]
    #     ch_arr = list(range(img.shape[2]))
    #     random.shuffle(ch_arr)
    #     return {"channels_shuffled": ch_arr}

    def get_transform_init_args_names(self):
        return ()


def get_transform_pipeline(width, height, is_train=True):
    if(is_train):
        return Compose([
            
            OneOf([
                ShiftScaleRotate(rotate_limit=15, p=0.87),
                Rotate(limit=15, p=0.78),
                RandomScale(p=0.78)
            ], 0.85),
            HorizontalFlip(),
            GaussNoise(p=0.67),
            OneOf([
                Blur(),
                GaussianBlur(),
                MedianBlur()
            ], p=0.75),
            OneOf([
                OneOf([
                    RandomBrightness(p=0.78),
                    RandomContrast(p=0.78)
                ], p=0.75),
                OneOf([
                    CLAHE(),
                    ColorJitter(),
                    HueSaturationValue(),
                    ChannelShuffle(),
                ], p=0.75),
            ],0.5),

            # Resize(height=height, width=width),
            # Downscale(scale_min=0.85, scale_max=0.95, p=0.25),
            Resize(height=height, width=width),
            Normalize(),
            ToTensorV2()
        ])
    else:
        return Compose([
            Resize(height=height, width=width),
            Normalize(),
            ToTensorV2()
        ])
