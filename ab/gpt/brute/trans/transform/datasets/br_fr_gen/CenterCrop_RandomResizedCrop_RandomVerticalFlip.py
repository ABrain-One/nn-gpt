import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.CenterCrop(size=32),
    transforms.RandomResizedCrop(size=32, scale=(0.58, 0.85), ratio=(1.3, 1.59)),
    transforms.RandomVerticalFlip(p=0.83),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
