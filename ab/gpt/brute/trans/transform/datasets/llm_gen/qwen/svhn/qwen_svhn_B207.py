import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomResizedCrop(size=32, scale=(0.5, 1.0), ratio=(0.75, 1.25)), transforms.RandomHorizontalFlip(p=0.5), transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), transforms.CenterCrop(size=28), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])