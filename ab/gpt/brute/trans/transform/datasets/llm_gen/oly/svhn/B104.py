import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomVerticalFlip(p=0.11), transforms.RandomAdjustSharpness(1.45, p=0.12), transforms.RandomResizedCrop(size=224, scale=(0.55, 0.96), ratio=(0.91, 2.13)), transforms.ToTensor(), transforms.Normalize(*norm)])