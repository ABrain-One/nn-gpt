import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomResizedCrop(size=32, scale=(0.8, 1.0)), transforms.ColorJitter(brightness=0.9, contrast=0.9, saturation=0.9, hue=0.1), transforms.RandomHorizontalFlip(p=0.5), transforms.RandomVerticalFlip(p=0.2), transforms.ToTensor(), transforms.Normalize(*norm)])