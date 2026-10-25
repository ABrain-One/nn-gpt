import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)), transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), transforms.RandomAffine(degrees=5, translate=(0.1, 0.1), scale=(0.9, 1.1)), transforms.RandomHorizontalFlip(p=0.2), transforms.RandomVerticalFlip(p=0.1), transforms.ToTensor(), transforms.Normalize(*norm)])