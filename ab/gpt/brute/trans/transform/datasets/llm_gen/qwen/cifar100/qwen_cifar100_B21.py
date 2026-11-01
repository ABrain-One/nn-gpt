import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomResizedCrop(size=32, scale=(0.8, 0.9), ratio=(1.0, 1.1)), transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), transforms.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.9, 1.1)), transforms.RandomHorizontalFlip(p=0.5), transforms.RandomVerticalFlip(p=0.3), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])