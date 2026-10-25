import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomResizedCrop(size=224, scale=(0.8, 1.2), ratio=(0.9, 1.1)), transforms.RandomHorizontalFlip(p=0.5), transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02), transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)), transforms.RandomRotation(degrees=(-15, 15)), transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(*norm)])