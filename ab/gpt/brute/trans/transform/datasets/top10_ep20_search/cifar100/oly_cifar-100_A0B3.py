import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomResizedCrop(64, scale=(0.8, 2.0), ratio=(1.0, 1.0)), transforms.RandomHorizontalFlip(p=0.5), transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1), transforms.RandomApply([transforms.GaussianBlur(3)], p=0.2), transforms.RandomRotation(degrees=(-15, 15)), transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)), transforms.ToTensor(), transforms.Normalize(*norm)])