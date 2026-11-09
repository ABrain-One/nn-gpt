import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.ColorJitter(brightness=1.05, contrast=0.85, saturation=0.95, hue=0.01), transforms.RandomGrayscale(p=0.32), transforms.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.8, 1.3)), transforms.RandomHorizontalFlip(p=0.21), transforms.RandomRotation(degrees=(-15, 15)), transforms.ToTensor(), transforms.Normalize(*norm)])