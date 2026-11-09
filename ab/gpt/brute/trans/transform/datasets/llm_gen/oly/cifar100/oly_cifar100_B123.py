import torch
import torchvision.transforms as transforms

def transform():
    return transforms.Compose([transforms.RandomVerticalFlip(p=0.1), transforms.RandomResizedCrop(32, scale=(0.8, 1.2), ratio=(0.8, 1.2)), transforms.RandomRotation(degrees=10, p=0.5), transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), transforms.RandomApply([transforms.GaussianBlur(3)], p=0.25), transforms.Resize(224), transforms.ToTensor(), transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.201])])