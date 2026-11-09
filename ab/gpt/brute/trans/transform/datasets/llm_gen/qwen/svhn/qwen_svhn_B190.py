import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomResizedCrop(size=32, scale=(0.5, 0.9), ratio=(0.8, 1.2)), transforms.ColorJitter(brightness=0.7, contrast=1.05, saturation=0.8, hue=0.04), transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)), transforms.RandomHorizontalFlip(p=0.25), transforms.ToTensor(), transforms.Normalize(*norm)])