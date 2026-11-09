import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomResizedCrop(size=32, scale=(0.7, 0.85), ratio=(0.9, 2.5)), transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), transforms.RandomHorizontalFlip(p=0.4), transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10), transforms.CenterCrop(size=28), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])