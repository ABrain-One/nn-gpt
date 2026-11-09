import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.ColorJitter(brightness=0.98, contrast=1.18, saturation=1.13, hue=0.07), transforms.RandomResizedCrop(64, scale=(0.68, 1.0), ratio=(0.94, 2.9)), transforms.RandomHorizontalFlip(p=0.36), transforms.RandomRotation(degrees=13), transforms.ToTensor(), transforms.Normalize(*norm)])