import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomResizedCrop(size=32, scale=(0.5, 0.89), ratio=(0.94, 2.85)), transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1), transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5)), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])