import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomResizedCrop(size=32, scale=(0.8, 1.2), ratio=(0.75, 1.33)), transforms.RandomApply([transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)], p=0.8), transforms.RandomHorizontalFlip(p=0.5), transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)), transforms.RandomRotation(degrees=(-15, 15)), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])