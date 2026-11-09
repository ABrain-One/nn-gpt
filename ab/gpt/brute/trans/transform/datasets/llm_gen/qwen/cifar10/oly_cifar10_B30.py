import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomResizedCrop(size=32, scale=(0.5, 0.87), ratio=(0.75, 1.33)), transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.05), transforms.RandomHorizontalFlip(p=0.2), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])