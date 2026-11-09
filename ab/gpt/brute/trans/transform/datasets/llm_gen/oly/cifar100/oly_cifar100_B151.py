import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomResizedCrop(256, scale=(0.8, 1.0), ratio=(0.9, 1.1)), transforms.RandomHorizontalFlip(p=0.5), transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05), transforms.RandomRotation(degrees=(-15, 15)), transforms.RandomApply([transforms.GaussianBlur(radius_max=1.5)], p=0.25), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(**norm)])