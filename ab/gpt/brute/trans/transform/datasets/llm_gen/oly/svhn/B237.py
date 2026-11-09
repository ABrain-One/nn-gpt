import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomResizedCrop(224, scale=(0.8, 1.2), ratio=(0.8, 1.3)), transforms.RandomRotation(degrees=(-15, 15)), transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), transforms.GaussianBlur(kernel_size=3, sigma=1.5), transforms.ToTensor(), transforms.Normalize(*norm)])