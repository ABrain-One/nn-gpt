import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomHorizontalFlip(p=0.75), transforms.RandomResizedCrop(size=32, scale=(0.8, 0.9), ratio=(0.9, 1.1)), transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05), transforms.GaussianBlur(kernel_size=(3, 3), sigma=(1.5, 1.5)), transforms.RandomRotation(degrees=(-5, 5)), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])