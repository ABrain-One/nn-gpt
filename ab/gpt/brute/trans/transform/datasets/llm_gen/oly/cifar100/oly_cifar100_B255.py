import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomResizedCrop(32, scale=(0.8, 1.2), ratio=(0.9, 1.1)), transforms.ColorJitter(brightness=0.8, contrast=1.2, saturation=0.8, hue=0.1), transforms.RandomRotation(5), transforms.RandomHorizontalFlip(p=0.5), transforms.GaussianBlur(kernel_size=3, sigma=1.5), transforms.Resize(64), transforms.ToTensor(), transforms.Normalize(*norm)])