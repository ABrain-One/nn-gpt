import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1), transforms.RandomHorizontalFlip(p=0.5), transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)), transforms.RandomRotation(degrees=(-10, 10)), transforms.ToTensor(), transforms.Normalize(*norm)])