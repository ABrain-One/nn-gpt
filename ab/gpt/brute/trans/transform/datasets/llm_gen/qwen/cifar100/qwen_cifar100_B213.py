import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomHorizontalFlip(p=0.5), transforms.RandomCrop(size=32, padding=4), transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1), transforms.RandomRotation(degrees=(-15, 15)), transforms.GaussianBlur(kernel_size=3, sigma=(0.2, 1.0)), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])