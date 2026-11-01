import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=(-5, 5)), transforms.ColorJitter(brightness=0.7, contrast=0.8, saturation=0.7, hue=0.05), transforms.RandomGrayscale(p=0.2), transforms.RandomHorizontalFlip(p=0.6), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])