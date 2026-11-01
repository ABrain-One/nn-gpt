import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomResizedCrop(32, scale_range=(0.8, 1.0), aspect_ratio=1.0), transforms.RandomHorizontalFlip(p=0.5), transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1), transforms.RandomApply([transforms.Autocontrast()], p=0.5), transforms.ToTensor(), transforms.Normalize(**norm)])