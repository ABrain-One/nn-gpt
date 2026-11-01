import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomHorizontalFlip(p=0.75), transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=(-5, 5)), transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)), transforms.RandomResizedCrop(size=32, scale=(0.8, 1.0), ratio=(0.75, 1.25)), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])