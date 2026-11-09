import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomResizedCrop(224, scale=(0.75, 1.0)), transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=0.5)], p=0.5), transforms.RandomHorizontalFlip(p=0.5), transforms.RandomRotation(degrees=(-15, 15)), transforms.ToTensor(), transforms.Normalize(**norm)])