import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomCrop(size=32), transforms.RandomHorizontalFlip(p=0.4), transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.1), transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])