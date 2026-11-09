import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomVerticalFlip(p=0.29),
    transforms.RandomAutocontrast(p=0.87),
    transforms.RandomResizedCrop(size=32, scale=(0.74, 0.9), ratio=(1.1, 1.47)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
