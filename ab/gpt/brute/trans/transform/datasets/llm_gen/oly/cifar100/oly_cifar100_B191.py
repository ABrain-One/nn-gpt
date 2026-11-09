import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomResizedCrop(32, scale=(0.8, 1.0), ratio=(0.8, 1.28)), transforms.RandomHorizontalFlip(p=0.5), transforms.RandomAffine(degrees=5, translate=(0.1, 0.1), scale=(0.95, 1.05), shear=(-5.0, 5.0)), transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), transforms.GaussianBlur(kernel_size=3, sigma=1.0), transforms.ToTensor(), transforms.Normalize(**norm)])