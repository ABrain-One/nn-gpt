import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomResizedCrop(size=32, scale=(0.6, 0.8), ratio=(1.1, 1.4)), transforms.RandomHorizontalFlip(p=0.3), transforms.RandomVerticalFlip(p=0.1), transforms.ColorJitter(brightness=0.7, contrast=0.8, saturation=0.7, hue=0.06), transforms.RandomAffine(degrees=5, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=(-10, 10)), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])