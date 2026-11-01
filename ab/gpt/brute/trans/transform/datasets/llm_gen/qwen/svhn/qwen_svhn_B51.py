import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomHorizontalFlip(p=0.5), transforms.RandomRotation(degrees=15), transforms.ColorJitter(brightness=0.8, contrast=1.05, saturation=1.05, hue=0.05), transforms.RandomResizedCrop(size=32, scale=(0.6, 1.0), ratio=(0.8, 1.3)), transforms.RandomAffine(degrees=5, translate=(0.1, 0.1), scale=(0.9, 1.1)), transforms.ToTensor(), transforms.Normalize(*norm)])