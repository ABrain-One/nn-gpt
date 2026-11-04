import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomVerticalFlip(p=0.5), transforms.RandomResizedCrop(64, scale=(0.8, 1.0), ratio=(0.9, 1.1)), transforms.RandomRotation(15), transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1), transforms.ToTensor(), transforms.Normalize(*norm)])