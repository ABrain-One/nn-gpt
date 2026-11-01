import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomResizedCrop(size=32, scale=(0.75, 0.99), ratio=(0.8, 2.8)), transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05), transforms.RandomHorizontalFlip(p=0.5), transforms.RandomApply([transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.2), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])