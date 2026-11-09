import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomVerticalFlip(p=0.36), transforms.RandomHorizontalFlip(p=0.5), transforms.RandomResizedCrop(30, scale=(0.8, 1.0), ratio=(0.9, 1.1)), transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5), transforms.RandomApply([transforms.GaussianBlur(kernel_size=5)], p=0.5), transforms.RandomAutocontrast(p=0.9), transforms.Resize(64), transforms.ToTensor(), transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.201])])