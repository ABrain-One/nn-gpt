import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomHorizontalFlip(p=0.45), transforms.RandomAffine(degrees=15, translate=(0.08, 0.12), scale=(0.9, 1.1), shear=(-10, 10)), transforms.ColorJitter(brightness=0.8, contrast=0.9, saturation=1.05, hue=0.02), transforms.RandomResizedCrop(size=32, scale=(0.8, 1.0), ratio=(0.9, 1.1)), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])