import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomVerticalFlip(p=0.5), transforms.RandomResizedCrop(32, scale=(0.8, 1.0), ratio=(0.8, 1.25)), transforms.RandomRotation(10), transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1), transforms.ToTensor(), transforms.Normalize(**norm)])