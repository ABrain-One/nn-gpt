import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomHorizontalFlip(p=0.5), transforms.RandomRotation(degrees=15), transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), transforms.RandomResizedCrop(size=32, scale=(0.85, 0.95), ratio=(0.9, 1.1)), transforms.GaussianBlur(kernel_size=3), transforms.ToTensor(), transforms.Normalize(*norm)])