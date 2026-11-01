import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomHorizontalFlip(p=0.4), transforms.RandomRotation(degrees=15, p=0.75), transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.6), transforms.RandomErasing(scale=(0.02, 0.15), ratio=(0.3, 3.3), p=0.5), transforms.RandomResizedCrop(size=32, scale=(0.8, 1.2), ratio=(0.8, 1.2)), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])