import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomResizedCrop(size=32, scale=(0.6, 0.9), ratio=(0.8, 1.2)), transforms.ColorJitter(brightness=0.6, contrast=1.2, saturation=0.6, hue=0.05), transforms.RandomHorizontalFlip(p=0.7), transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)), transforms.RandomAffine(degrees=(-15, 15), translate=(0.1, 0.1), scale=(0.9, 1.1)), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])