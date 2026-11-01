import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomVerticalFlip(p=0.18), transforms.RandomAutocontrast(p=0.52), transforms.RandomResizedCrop(size=32, scale=(0.8, 0.91), ratio=(0.85, 1.48)), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])