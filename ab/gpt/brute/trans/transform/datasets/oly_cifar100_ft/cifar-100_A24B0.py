import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomResizedCrop(size=32, scale=(0.68, 0.84), ratio=(0.79, 2.13)), transforms.RandomAutocontrast(p=0.11), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])