import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomResizedCrop(32, scale=(0.7, 1.0), ratio=(0.8, 2.0)), transforms.RandomHorizontalFlip(p=0.3), transforms.RandomRotation(15), transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.1), transforms.RandomAdjustSharpness(0.5, p=0.2), transforms.CenterCrop(32), transforms.Resize(64), transforms.ToTensor(), transforms.Normalize(*norm)])