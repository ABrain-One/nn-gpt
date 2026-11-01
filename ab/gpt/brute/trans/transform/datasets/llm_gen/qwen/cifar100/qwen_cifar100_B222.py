import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomHorizontalFlip(p=0.3), transforms.RandomAffine(degrees=(-15, 15), translate=(0.1, 0.1), scale=(0.8, 1.2), shear=(-5, 5)), transforms.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.9, 1.1)), transforms.ToTensor(), transforms.Normalize(*norm)])