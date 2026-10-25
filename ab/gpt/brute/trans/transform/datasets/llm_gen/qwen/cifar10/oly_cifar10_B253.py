import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomResizedCrop(size=32, scale=(0.8, 0.9), ratio=(0.9, 1.1)), transforms.RandomAffine(degrees=10, translate=(0.04, 0.05), scale=(1.15, 1.68), shear=(1.5, 6.4)), transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05), transforms.GaussianBlur(kernel_size=3, sigma=(0.6, 1.0)), transforms.RandomHorizontalFlip(p=0.2), transforms.RandomVerticalFlip(p=0.1), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])