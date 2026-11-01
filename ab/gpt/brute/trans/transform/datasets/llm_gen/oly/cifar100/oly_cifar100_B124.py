import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomResizedCrop(64, scale=(0.5, 1.5), ratio=(0.8, 1.2)), transforms.RandomHorizontalFlip(p=0.7), transforms.RandomInvert(p=0.3), transforms.ColorJitter(brightness=0.5, contrast=0.4, saturation=0.6, hue=0.05), transforms.GaussianBlur(kernel_size=3, sigma=1.0), transforms.ToTensor(), transforms.Normalize(**norm)])