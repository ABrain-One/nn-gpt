import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.75, 1.333)), transforms.RandomHorizontalFlip(p=0.5), transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1), transforms.RandomRotation(degrees=15), transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.5), transforms.ToTensor(), transforms.Normalize(**norm)])