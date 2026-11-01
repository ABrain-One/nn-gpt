import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomResizedCrop(32, scale=(0.8, 1.0), ratio=(0.8, 1.2)), transforms.RandomHorizontalFlip(p=0.5), transforms.RandomRotation(15), transforms.ColorJitter(hue=0.1), transforms.ToTensor(), transforms.Normalize(*norm)])