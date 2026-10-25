import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomResizedCrop(size=32, scale=(0.85, 0.9), ratio=(0.98, 2.0)), transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05), transforms.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05), shear=2), transforms.RandomHorizontalFlip(p=0.4), transforms.ToTensor(), transforms.Normalize(*norm)])