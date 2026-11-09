import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.Pad(padding=4, padding_mode='edge', fill=(177, 215, 124)), transforms.RandomResizedCrop(32, scale=(0.8, 1.0), ratio=(0.83, 2.72)), transforms.RandomCrop(26), transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=0.5)], p=0.5), transforms.Resize(64), transforms.ToTensor(), transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.201])])