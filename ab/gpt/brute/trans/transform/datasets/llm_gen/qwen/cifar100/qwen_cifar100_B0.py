import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomHorizontalFlip(p=0.5), transforms.RandomRotation(degrees=(-15, 15)), transforms.RandomResizedCrop(size=32, scale=(0.8, 1.0), ratio=(0.9, 1.1)), transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02), transforms.ToTensor(), transforms.Normalize(*norm)])