import torch
import torchvision.transforms as transforms
from torchvision import transforms

def transform(norm):
    return transforms.Compose([transforms.ColorJitter(brightness=1.5, contrast=1.2, saturation=1.0, hue=0.03), transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=1.0)], p=0.5), transforms.RandomAffine(degrees=10, translate=(0.2, 0.3), scale=(0.8, 1.3), shear=(-15.0, 15.0)), transforms.RandomVerticalFlip(p=0.5), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])