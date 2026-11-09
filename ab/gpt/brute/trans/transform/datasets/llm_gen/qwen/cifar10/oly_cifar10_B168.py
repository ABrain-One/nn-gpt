import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomHorizontalFlip(p=0.3), transforms.RandomRotation(degrees=15), transforms.RandomResizedCrop(size=32, scale=(0.8, 1.0), ratio=(0.8, 1.2)), transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1), transforms.RandomAffine(degrees=(-5, 5), translate=(0.05, 0.05), scale=(0.95, 1.05), shear=(-2, 2)), transforms.RandomGrayscale(p=0.2), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])