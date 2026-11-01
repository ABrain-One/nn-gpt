import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomResizedCrop(size=32, scale=(0.6, 0.9), ratio=(1.2, 2.8)), transforms.RandomHorizontalFlip(p=0.5), transforms.ColorJitter(brightness=0.7, contrast=0.8, saturation=0.7, hue=0.02), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])