import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomResizedCrop(size=32, scale=(0.5, 0.8), ratio=(0.9, 2.6)), transforms.RandomHorizontalFlip(p=0.2), transforms.RandomRotation(degrees=(-15, 15)), transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1), transforms.ToTensor(), transforms.Normalize(*norm)])