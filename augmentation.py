from albumentations import (
    Compose, ShiftScaleRotate, RandomBrightness, HueSaturationValue, RandomContrast, HorizontalFlip,
    Rotate, Resize, Lambda, CLAHE, ColorJitter, RandomBrightnessContrast, GaussianBlur, Blur, MedianBlur,
    GridDistortion, Downscale, ChannelShuffle, Normalize, OneOf, IAAAdditiveGaussianNoise, GaussNoise,
    RandomScale, ISONoise
)
from albumentations.pytorch import ToTensorV2

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

            Resize(height=height, width=width),
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
