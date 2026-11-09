import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.9, 1.1)), transforms.RandomVerticalFlip(p=0.2), transforms.RandomHorizontalFlip(p=0.3), transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2), transforms.RandomAutocontrast(p=0.5), transforms.ToTensor(), transforms.Normalize(*norm)])