import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomResizedCrop(size=32, scale=(0.5, 0.7), ratio=(1.1, 1.5)), transforms.RandomHorizontalFlip(p=0.6), transforms.ColorJitter(brightness=0.9, contrast=0.9, saturation=0.7, hue=0.02), transforms.RandomAffine(degrees=15, translate=(0.02, 0.1), scale=(1.05, 1.3), shear=(0.5, 5.0)), transforms.RandomGrayscale(p=0.75), transforms.Resize((128, 128)), transforms.ToTensor(), transforms.Normalize(*norm)])