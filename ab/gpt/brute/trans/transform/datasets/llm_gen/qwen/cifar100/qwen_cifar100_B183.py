import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomRotation(degrees=(-15, 15)), transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), transforms.RandomResizedCrop(size=32, scale=(0.7, 1.0), ratio=(0.8, 1.2)), transforms.ToTensor(), transforms.Normalize(*norm)])