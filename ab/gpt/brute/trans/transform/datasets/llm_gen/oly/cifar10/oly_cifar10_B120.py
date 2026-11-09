import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomResizedCrop(32, scale=(0.5, 1.0), ratio=(0.8, 2.0)), transforms.RandomVerticalFlip(p=0.67), transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])