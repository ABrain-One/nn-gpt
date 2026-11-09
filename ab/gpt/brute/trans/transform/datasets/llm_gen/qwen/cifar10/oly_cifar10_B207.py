import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomRotation(degrees=(-15, 15), p=0.5), transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.4), transforms.RandomResizedCrop(size=32, scale=(0.7, 0.8), ratio=(0.9, 1.1)), transforms.RandomHorizontalFlip(p=0.5), transforms.RandomVerticalFlip(p=0.2), transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 0.5), p=0.3), transforms.ToTensor(), transforms.Normalize(*norm)])