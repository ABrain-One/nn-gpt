import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomResizedCrop(size=32, scale=(0.67, 0.85), ratio=(0.9, 1.43)), transforms.RandomHorizontalFlip(p=0.65), transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.05), transforms.ToTensor(), transforms.Normalize(*norm)])