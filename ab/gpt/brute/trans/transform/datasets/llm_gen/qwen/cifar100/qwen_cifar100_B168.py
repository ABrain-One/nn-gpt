import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomResizedCrop(32), transforms.RandomHorizontalFlip(p=0.5), transforms.RandomAffine(degrees=(-10, 10), translate=(0.1, 0.1), scale=(0.9, 1.1), shear=(-5, 5)), transforms.ToTensor(), transforms.Normalize(*norm)])