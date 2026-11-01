import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomHorizontalFlip(p=0.79), transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), transforms.RandomResizedCrop(size=32, scale=(0.8, 0.85), ratio=(0.95, 1.05)), transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(1.12, 1.65), shear=(1.5, 5.5)), transforms.GaussianBlur(kernel_size=3, sigma=(0.75, 1.15)), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])