import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomAffine(degrees=(-10, 10), translate=(0.1, 0.1), scale=(0.9, 1.1)), transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05), transforms.RandomResizedCrop(size=64, scale=(0.8, 1.0)), transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 2.0)), transforms.ToTensor(), transforms.Normalize(*norm)])