import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomResizedCrop(size=32, scale=(0.7, 0.85), ratio=(0.7, 2.75)), transforms.RandomHorizontalFlip(p=0.85), transforms.RandomVerticalFlip(p=0.2), transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])