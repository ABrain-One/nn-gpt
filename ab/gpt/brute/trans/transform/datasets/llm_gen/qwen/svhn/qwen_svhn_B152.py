import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomHorizontalFlip(p=0.15), transforms.RandomAffine(degrees=(-10, 10), translate=(0.1, 0.1), scale=(0.9, 1.1)), transforms.ColorJitter(brightness=0.8, contrast=0.9, saturation=0.9, hue=0.04), transforms.RandomResizedCrop(64, scale=(0.8, 1.0)), transforms.ToTensor(), transforms.Normalize(*norm)])