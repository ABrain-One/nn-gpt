import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=4, fill=(98, 247, 36), padding_mode='reflect'),
    transforms.RandomResizedCrop(size=32, scale=(0.6, 0.84), ratio=(1.27, 1.58)),
    transforms.RandomVerticalFlip(p=0.12),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
