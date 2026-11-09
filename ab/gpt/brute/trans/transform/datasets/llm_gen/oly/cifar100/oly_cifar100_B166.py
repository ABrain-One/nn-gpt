import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.Resize(96), transforms.RandomResizedCrop(64, scale=(0.62, 0.9), ratio=(1.32, 2.13)), transforms.RandomVerticalFlip(p=0.16), transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), transforms.GaussianBlur(kernel_size=5, sigma=(0.4, 1.5)), transforms.ToTensor(), transforms.Normalize(**norm)])