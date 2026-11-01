import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomResizedCrop(size=32, scale=(0.71, 0.92), ratio=(1.2, 1.53)), transforms.RandomHorizontalFlip(p=0.4), transforms.RandomAutocontrast(p=0.82), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])