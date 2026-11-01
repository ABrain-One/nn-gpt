import torch
import torchvision.transforms as transforms
from torchvision.transforms import RandomApply

def transform(norm):
    gaussian_blur = transforms.GaussianBlur(kernel_size=3, sigma=1.0)
    apply_gb = RandomApply([gaussian_blur], p=0.25)
    return transforms.Compose([transforms.Pad(padding=4, fill=(194, 244, 182), padding_mode='constant'), transforms.RandomHorizontalFlip(p=0.5), transforms.RandomRotation(degrees=(-15, 15)), transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1), apply_gb, transforms.Resize(64), transforms.ToTensor(), transforms.Normalize(*norm)])