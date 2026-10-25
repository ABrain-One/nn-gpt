import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomResizedCrop(size=32, scale=(0.5, 0.9), ratio=(0.8, 1.2)), transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.05), transforms.RandomHorizontalFlip(), transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=(-5, 5)), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])