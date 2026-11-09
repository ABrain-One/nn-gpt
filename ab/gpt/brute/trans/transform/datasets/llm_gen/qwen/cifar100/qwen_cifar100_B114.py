import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomResizedCrop(size=32, scale=(0.8, 0.9), ratio=(1.1, 2.7)), transforms.RandomHorizontalFlip(p=0.5), transforms.RandomApply([transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)], p=0.8), transforms.RandomRotation(degrees=(-10, 10)), transforms.RandomCrop(size=27), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])