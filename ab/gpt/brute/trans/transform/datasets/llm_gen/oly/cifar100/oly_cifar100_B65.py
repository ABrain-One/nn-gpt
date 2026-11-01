import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomResizedCrop(32, scale=(0.7, 0.9), ratio=(1.2, 1.5)), transforms.RandomHorizontalFlip(), transforms.RandomRotation(15), transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), transforms.GaussianBlur(kernel_size=3, sigma=(0.5, 1.5)), transforms.Resize(64), transforms.ToTensor(), transforms.Normalize(**norm)])