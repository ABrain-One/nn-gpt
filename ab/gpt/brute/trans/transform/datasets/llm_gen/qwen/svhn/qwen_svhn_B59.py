import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomResizedCrop(size=32, scale=(0.8, 1.2)), transforms.ColorJitter(brightness=0.7, contrast=0.8, saturation=1.1, hue=0.05), transforms.RandomHorizontalFlip(p=0.5), transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)), transforms.ToTensor(), transforms.Normalize(*norm)])