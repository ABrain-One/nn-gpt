import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomHorizontalFlip(p=0.5), transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.1), transforms.RandomResizedCrop(64, scale=(0.8, 1.0), ratio=(0.9, 1.1)), transforms.RandomRotation(5), transforms.ToTensor(), transforms.Normalize(*norm)])