import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomAffine(degrees=(-10, 10), translate=(0.1, 0.1), scale=(0.9, 1.1), shear=(-5, 5)), transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), transforms.RandomRotation(degrees=(-15, 15)), transforms.RandomGrayscale(p=0.2), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])