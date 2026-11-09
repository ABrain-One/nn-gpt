import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomHorizontalFlip(p=0.75), transforms.RandomAffine(translate=(0.1, 0.1), scale=(0.9, 1.1), shear=(-10, 10), p=0.4), transforms.RandomResizedCrop(size=32, scale=(0.8, 1.0), ratio=(0.9, 1.1), interpolation=torchvision.transforms.InterpolationMode.BILINEAR), transforms.Resize((64, 64)), transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5), transforms.ToTensor(), transforms.Normalize(*norm)])