import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomResizedCrop(size=32, scale=(0.75, 0.85), ratio=(0.9, 2.0)), transforms.RandomHorizontalFlip(p=0.4), transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), transforms.RandomRotation(degrees=(-15, 15)), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])