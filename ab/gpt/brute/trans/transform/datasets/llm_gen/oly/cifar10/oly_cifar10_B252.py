import torch
from torchvision import transforms

def transform(norm):
    return transforms.Compose([transforms.Resize(256), transforms.RandomCrop(224), transforms.RandomHorizontalFlip(p=0.5), transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=1.0)], p=0.5), transforms.ToTensor(), transforms.Normalize(**norm)])