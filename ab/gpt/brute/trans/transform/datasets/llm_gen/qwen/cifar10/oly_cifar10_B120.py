import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomHorizontalFlip(p=0.5), transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.8), transforms.RandomRotation(degrees=15, p=0.7), transforms.RandomResizedCrop(size=(64, 64), scale=(0.8, 1.0), ratio=(0.9, 1.1), interpolation=torchvision.transforms.InterpolationMode.BILINEAR, p=0.6), transforms.ToTensor(), transforms.Normalize(*norm)])