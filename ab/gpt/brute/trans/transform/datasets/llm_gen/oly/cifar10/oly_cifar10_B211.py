import torch
import torchvision.transforms as transforms
from PIL import Image

def transform(norm):
    return transforms.Compose([transforms.RandomHorizontalFlip(p=0.8), transforms.RandomResizedCrop(32, scale=(0.8, 1.0), interpolation=Image.BICUBIC), transforms.RandomRotation(degrees=(-5, 5)), transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1), transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.5, 1.5))], p=0.2), transforms.ToTensor(), transforms.Normalize(*norm)])