import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=0, fill=(177, 215, 124), padding_mode='edge'),
    transforms.RandomResizedCrop(size=32, scale=(0.8, 0.94), ratio=(0.83, 2.72)),
    transforms.RandomCrop(size=26),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
