import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomVerticalFlip(p=0.83),
    transforms.Pad(padding=2, fill=(170, 244, 107), padding_mode='constant'),
    transforms.RandomResizedCrop(size=32, scale=(0.61, 0.97), ratio=(1.1, 1.72)),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
