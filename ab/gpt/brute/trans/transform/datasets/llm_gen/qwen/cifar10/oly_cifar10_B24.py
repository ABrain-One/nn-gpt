import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomRotation(15), transforms.RandomResizedCrop(size=32, scale=(0.8, 1.2), ratio=(0.9, 1.1)), transforms.RandomBrightness(0.1), transforms.RandomContrast(0.1), transforms.ToTensor(), transforms.Normalize(*norm)])