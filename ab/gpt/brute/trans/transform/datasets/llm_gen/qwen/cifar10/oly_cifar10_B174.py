import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomResizedCrop(size=32, scale=(0.8, 1.0), ratio=(0.7, 1.2)), transforms.RandomApply([transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05)], p=0.5), transforms.RandomHorizontalFlip(p=0.5), transforms.RandomRotation(degrees=(-5, 5)), transforms.ToTensor(), transforms.Normalize(*norm)])