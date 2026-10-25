import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=(-10, 10)), transforms.ColorJitter(brightness=0.7, contrast=0.85, saturation=1.05, hue=0.03), transforms.RandomHorizontalFlip(p=0.2), transforms.RandomResizedCrop(size=32, scale=(0.7, 0.9), ratio=(0.9, 1.1)), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])