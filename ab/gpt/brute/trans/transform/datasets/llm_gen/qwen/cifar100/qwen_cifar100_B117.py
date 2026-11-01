import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomResizedCrop(size=32, scale=(0.65, 0.9), ratio=(1.1, 2.7)), transforms.RandomApply([transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)], p=0.7), transforms.RandomHorizontalFlip(p=0.6), transforms.RandomVerticalFlip(p=0.2), transforms.RandomRotation(degrees=(-5, 5)), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])