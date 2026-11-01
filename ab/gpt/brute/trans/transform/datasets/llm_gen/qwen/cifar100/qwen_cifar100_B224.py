import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomResizedCrop(size=32, scale=(0.85, 0.95), ratio=(0.8, 2.7)), transforms.RandomAffine(degrees=(-15, 15), translate=(0.1, 0.1), scale=(0.8, 1.2), shear=(-5, 5)), transforms.RandomHorizontalFlip(p=0.2), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])