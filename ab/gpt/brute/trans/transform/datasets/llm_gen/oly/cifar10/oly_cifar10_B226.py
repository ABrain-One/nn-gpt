import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomHorizontalFlip(p=0.82), transforms.RandomRotation(degrees=(-15, 15)), transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1), transforms.GaussianBlur(kernel_size=5, sigma=1.0), transforms.Resize(224), transforms.ToTensor(), transforms.Normalize(*norm)])