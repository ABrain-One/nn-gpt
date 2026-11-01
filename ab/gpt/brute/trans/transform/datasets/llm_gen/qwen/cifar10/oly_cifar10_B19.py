import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomResizedCrop(size=32, scale=(0.5, 0.95), ratio=(0.8, 3.0)), transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), transforms.RandomHorizontalFlip(p=0.4), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])