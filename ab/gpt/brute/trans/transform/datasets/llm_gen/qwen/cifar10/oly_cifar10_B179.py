import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, p=0.4), transforms.RandomHorizontalFlip(p=0.2), transforms.RandomAffine(degrees=(-5, 5), translate=(0.1, 0.1), scale=(0.9, 1.1), shear=(-5, 5), fillcolor=(128, 128, 128), p=0.3), transforms.RandomResizedCrop(size=32, scale=(0.8, 0.9), ratio=(0.8, 1.2), interpolation=torchvision.transforms.InterpolationMode.BILINEAR), transforms.ToTensor(), transforms.Normalize(*norm)])