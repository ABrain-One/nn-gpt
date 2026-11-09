import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomResizedCrop(size=32, scale=(0.7, 0.96), ratio=(0.75, 1.57)), transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), transforms.RandomHorizontalFlip(p=0.5), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])