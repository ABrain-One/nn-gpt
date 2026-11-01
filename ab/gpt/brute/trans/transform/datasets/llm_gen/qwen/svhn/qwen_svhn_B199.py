import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomResizedCrop(size=32, scale=(0.7, 0.85), ratio=(1.0, 1.3)), transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1), transforms.RandomHorizontalFlip(p=0.5), transforms.RandomVerticalFlip(p=0.1), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])