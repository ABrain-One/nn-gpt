import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomResizedCrop(size=32, scale=(0.7, 0.96), ratio=(1.32, 2.84)), transforms.RandomHorizontalFlip(p=0.46), transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), transforms.RandomAffine(degrees=19, translate=(0.1, 0.08), scale=(0.84, 1.21), shear=(2.23, 7.0)), transforms.GaussianBlur(kernel_size=3, sigma=(0.54, 1.66)), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])