import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.ColorJitter(brightness=1.15, contrast=1.1, saturation=1.15, hue=0.08), transforms.RandomResizedCrop(size=32, scale=(0.8, 1.0), ratio=(0.9, 1.1)), transforms.RandomHorizontalFlip(p=0.7), transforms.RandomAffine(degrees=5, translate=(0.1, 0.1), scale=(0.95, 1.05), shear=(-5, 5)), transforms.ToTensor(), transforms.Normalize(*norm)])