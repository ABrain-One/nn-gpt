import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomResizedCrop(size=32, scale=(0.65, 0.9), ratio=(0.8, 1.2)), transforms.RandomHorizontalFlip(p=0.3), transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 1.0)), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])