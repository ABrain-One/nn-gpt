import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomHorizontalFlip(p=0.5), transforms.RandomVerticalFlip(p=0.2), transforms.RandomResizedCrop(size=(64, 64), scale=(0.8, 1.2), ratio=(0.75, 1.33)), transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.01), transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 0.5)), transforms.RandomRotation(degrees=(-15, 15)), transforms.ToTensor(), transforms.Normalize(*norm)])