import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomHorizontalFlip(p=0.5), transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), transforms.RandomRotation(degrees=(-10, 10)), transforms.RandomResizedCrop(size=224, scale=(0.8, 1.0)), transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 2.0)), transforms.ToTensor(), transforms.Normalize(*norm)])