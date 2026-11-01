import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomResizedCrop(32, scale=(0.8, 1.0), ratio=(0.9, 1.1)), transforms.RandomApply([transforms.GaussianBlur(5, sigma=(0.25, 1.22))], p=0.5), transforms.RandomHorizontalFlip(p=0.5), transforms.Resize(64), transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), transforms.ToTensor(), transforms.Normalize(*norm)])