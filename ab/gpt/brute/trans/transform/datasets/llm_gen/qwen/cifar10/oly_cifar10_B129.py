import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomHorizontalFlip(p=0.5), transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05), transforms.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.75, 1.33)), transforms.ToTensor(), transforms.Normalize(*norm)])