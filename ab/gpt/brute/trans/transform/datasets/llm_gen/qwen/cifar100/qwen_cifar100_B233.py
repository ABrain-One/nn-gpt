import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomHorizontalFlip(p=0.5), transforms.RandomResizedCrop(size=32, scale=(0.75, 1.0), ratio=(0.8, 1.2)), transforms.ColorJitter(brightness=0.4, contrast=0.9, saturation=0.6, hue=0.05), transforms.RandomAffine(degrees=(-5, 5), translate=(0.1, 0.1), scale=(0.9, 1.1), shear=(-2, 2)), transforms.ToTensor(), transforms.Normalize(*norm)])