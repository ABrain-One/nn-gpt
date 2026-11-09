import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomHorizontalFlip(p=0.57), transforms.RandomResizedCrop(64, scale=(0.8, 1.0), ratio=(1.0, 1.0)), transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), transforms.RandomApply([transforms.GaussianBlur(3)], p=0.4), transforms.ToTensor(), transforms.Normalize(*norm)])