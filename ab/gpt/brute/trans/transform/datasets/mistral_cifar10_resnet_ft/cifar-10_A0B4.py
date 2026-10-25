import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomHorizontalFlip(p=0.5), transforms.RandomCrop(size=32, padding=4), transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2), transforms.RandomRotation(degrees=10), transforms.ToTensor(), transforms.Normalize(*norm), transforms.RandomAffine(degrees=5, translate=(0.1, 0.1), scale=(0.9, 1.1)), transforms.RandomGrayscale(p=0.2)])