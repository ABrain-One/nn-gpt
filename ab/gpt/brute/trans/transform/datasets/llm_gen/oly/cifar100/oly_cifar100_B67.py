import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomResizedCrop(64, scale=(0.8, 1.0), ratio=(0.9, 1.1)), transforms.RandomHorizontalFlip(p=0.32), transforms.RandomApply([transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)], p=0.5), transforms.GaussianBlur(kernel_size=3, sigma=(0.3, 0.8)), transforms.ToTensor(), transforms.Normalize(*norm)])