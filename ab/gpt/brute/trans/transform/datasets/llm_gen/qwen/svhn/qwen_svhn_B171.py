import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomHorizontalFlip(p=0.5), transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.01), transforms.GaussianBlur(kernel_size=3, sigma=(0.5, 1.0)), transforms.RandomResizedCrop(64, scale=(0.8, 1.0)), transforms.ToTensor(), transforms.Normalize(*norm)])