import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomHorizontalFlip(p=0.75), transforms.RandomResizedCrop(size=32, scale=(0.61, 0.87), ratio=(0.83, 1.52)), transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1), transforms.RandomRotation(15), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])