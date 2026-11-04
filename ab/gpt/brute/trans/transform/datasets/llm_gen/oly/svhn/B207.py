import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomResizedCrop(32, scale=(0.8, 1.0), ratio=(0.9, 1.1)), transforms.RandomHorizontalFlip(p=0.8), transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), transforms.RandomInvert(p=0.2), transforms.GaussianBlur(kernel_size=3, sigma=1.0), transforms.Resize(64), transforms.ToTensor(), transforms.Normalize(mean=[0.449, 0.487, 0.478], std=[0.263, 0.255, 0.252])])