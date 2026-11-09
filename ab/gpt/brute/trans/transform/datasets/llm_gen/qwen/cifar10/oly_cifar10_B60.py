import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomHorizontalFlip(p=0.4), transforms.RandomRotation(degrees=(-10, 10), resample=torchvision.transforms.InterpolationMode.BILINEAR), transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), transforms.RandomErasing(scale=(0.02, 0.1), ratio=(0.3, 3.3), value=(0.485, 0.456, 0.406)), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])