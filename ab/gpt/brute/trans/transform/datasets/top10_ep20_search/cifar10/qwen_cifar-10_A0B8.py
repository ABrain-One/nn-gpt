import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomHorizontalFlip(p=0.75), transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=5), transforms.ColorJitter(brightness=0.7, contrast=0.8, saturation=0.8, hue=0.05), transforms.RandomResizedCrop(32, scale=(0.8, 1.0), ratio=(0.9, 1.1)), transforms.ToTensor(), transforms.Normalize(*norm)])