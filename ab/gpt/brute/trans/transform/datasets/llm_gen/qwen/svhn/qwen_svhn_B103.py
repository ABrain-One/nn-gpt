import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomResizedCrop(size=32, scale=(0.6, 0.85), ratio=(0.8, 3.1)), transforms.RandomAffine(degrees=(-15, 15), translate=(0.1, 0.1), scale=(0.9, 1.1), shear=(-5, 5)), transforms.RandomHorizontalFlip(p=0.5), transforms.RandomGrayscale(p=0.2), transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05), transforms.ToTensor(), transforms.Normalize(*norm)])