import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomAffine(degrees=5, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=(-5, 5)), transforms.RandomVerticalFlip(p=0.2), transforms.ColorJitter(brightness=0.7, contrast=1.05, saturation=1.05, hue=0.03), transforms.RandomResizedCrop(size=32, scale=(0.8, 1.0), ratio=(0.8, 1.2)), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])