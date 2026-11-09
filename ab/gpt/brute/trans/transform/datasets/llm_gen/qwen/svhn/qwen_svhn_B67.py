import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomResizedCrop(size=32, scale=(0.8, 1.2), ratio=(0.9, 1.1)), transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.05), transforms.RandomHorizontalFlip(p=0.5), transforms.ToTensor(), transforms.Normalize(*norm)])