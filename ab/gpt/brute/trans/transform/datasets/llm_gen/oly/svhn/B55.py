import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomResizedCrop(32, scale=(0.6, 1.0), ratio=(0.9, 1.5)), transforms.RandomAutocontrast(p=0.6), transforms.RandomInvert(p=0.7), transforms.RandomHorizontalFlip(p=0.5), transforms.Resize(64), transforms.ToTensor(), transforms.Normalize(*norm)])