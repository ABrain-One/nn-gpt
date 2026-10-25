import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomResizedCrop(size=32, scale=(0.8, 0.9), ratio=(1.0, 1.3)), transforms.RandomHorizontalFlip(p=0.2), transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05), transforms.ToTensor(), transforms.Normalize(*norm)])