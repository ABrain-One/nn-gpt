import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomHorizontalFlip(p=0.5), transforms.RandomResizedCrop(32, scale=(0.8, 1.0), ratio=(0.8, 1.2)), transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), transforms.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.9, 1.1)), transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5)), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])