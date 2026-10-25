import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomResizedCrop(32, scale=(0.6, 1.0), ratio=(0.9, 1.1)), transforms.RandomHorizontalFlip(p=0.7), transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1), transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)), transforms.Resize(64), transforms.ToTensor(), transforms.Normalize(*norm)])