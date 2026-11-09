import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomHorizontalFlip(p=0.5), transforms.RandomResizedCrop(size=32, scale=(0.5, 0.85), ratio=(0.75, 1.33)), transforms.RandomApply([transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1)], p=0.4), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])