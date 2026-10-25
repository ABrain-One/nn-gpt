import torch
from torchvision import transforms

def transform(norm):
    return transforms.Compose([transforms.RandomResizedCrop(64, scale=(0.75 ** 2, 1.0), ratio=(0.75, 1.7777)), transforms.RandomRotation(degrees=(-10, 10)), transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), transforms.RandomGaussianBlur(kernel_size=5, sigma=(0.5, 1.5)), transforms.RandomHorizontalFlip(p=0.6), transforms.ToTensor(), transforms.Normalize(**norm)])