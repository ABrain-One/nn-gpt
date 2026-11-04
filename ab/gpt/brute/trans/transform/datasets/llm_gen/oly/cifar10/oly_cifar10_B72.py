import torch
from torchvision import transforms

def transform(norm):
    return transforms.Compose([transforms.RandomHorizontalFlip(p=0.5), transforms.Pad(padding=3, fill=(238, 7, 23), padding_mode='reflect'), transforms.RandomResizedCrop(size=32, scale=(0.8, 1.0), ratio=(0.9, 1.1)), transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), transforms.GaussianBlur(kernel_size=3), transforms.RandomAutocontrast(p=0.3), transforms.Resize(64), transforms.ToTensor(), transforms.Normalize(**norm)])