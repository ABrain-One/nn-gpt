import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomCrop(size=32), transforms.RandomHorizontalFlip(p=0.4), transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1), transforms.RandomRotation(degrees=(-10, 10)), transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10), transforms.ToTensor(), transforms.Normalize(*norm)])