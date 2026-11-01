import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.ColorJitter(brightness=0.9, contrast=0.9, saturation=0.8, hue=0.1), transforms.RandomResizedCrop(64, scale=(0.75, 1.0), ratio=(0.8, 1.25)), transforms.RandomHorizontalFlip(), transforms.RandomApply([transforms.GaussianBlur(3)], p=0.5), transforms.ToTensor(), transforms.Normalize(*norm)])