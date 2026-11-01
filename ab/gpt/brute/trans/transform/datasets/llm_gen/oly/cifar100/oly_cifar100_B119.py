import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.ColorJitter(brightness=0.8, contrast=1.2, saturation=0.8, hue=0.1), transforms.RandomResizedCrop(64, scale=(0.8, 1.0), ratio=(1.0, 1.0)), transforms.RandomHorizontalFlip(p=0.2), transforms.RandomRotation(degrees=(-15, 15)), transforms.GaussianBlur(kernel_size=3, sigma=1.0), transforms.ToTensor(), transforms.Normalize(*norm)])