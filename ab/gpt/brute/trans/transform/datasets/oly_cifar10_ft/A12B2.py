<code>
import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomVerticalFlip(p=0.85),
    transforms.RandomAdjustSharpness(sharpness_factor=1.86, p=0.85),
    transforms.RandomMotionBlur(ps=(12, 0.39)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.47], std=[0.29], inplace=True)
])
</code>