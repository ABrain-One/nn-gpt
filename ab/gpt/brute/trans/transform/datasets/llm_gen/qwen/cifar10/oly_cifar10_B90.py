import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomRotation(degrees=15, p=0.25), transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, p=0.2), transforms.RandomResizedCrop(size=32, scale=(0.8, 1.0), ratio=(0.9, 1.1), interpolation=torchvision.transforms.InterpolationMode.BILINEAR, antialias=False, p=0.2), transforms.RandomHorizontalFlip(p=0.5), transforms.RandomVerticalFlip(p=0.1), transforms.ToTensor(), transforms.Normalize(*norm)])