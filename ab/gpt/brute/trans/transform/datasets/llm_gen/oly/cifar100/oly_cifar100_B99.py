import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomHorizontalFlip(p=0.5), transforms.RandomResizedCrop(64, scale=(0.8, 1.0), ratio=(0.9, 1.1)), transforms.ColorJitter(brightness=0.9, contrast=1.0, saturation=1.2, hue=0.1), transforms.RandomRotation(degrees=(-15, 15)), transforms.GaussianBlur(kernel_size=3, sigma=1.0, p=0.5), transforms.ToTensor(), transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])])