import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomVerticalFlip(p=0.8), transforms.RandomResizedCrop(224, scale=(0.79, 0.91), ratio=(1.21, 1.66)), transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05), transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.5), transforms.Resize(224), transforms.ToTensor(), transforms.Normalize(*norm)])