import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomResizedCrop(32, scale=(0.8, 1.2), ratio=(1.0, 1.0)), transforms.RandomHorizontalFlip(p=0.5), transforms.RandomRotation(degrees=(-10, 10)), transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1), transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.2), transforms.RandomEqualize(p=0.2), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])