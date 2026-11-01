import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomResizedCrop(32, scale=(0.6, 1.0), ratio=(0.8, 2.0)), transforms.RandomRotation(degrees=(-20, 20)), transforms.RandomHorizontalFlip(p=0.5), transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05), transforms.RandomEqualize(p=0.2), transforms.RandomInvert(p=0.2), transforms.Resize(64), transforms.ToTensor(), transforms.Normalize(*norm)])