import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1), transforms.RandomAffine(degrees=(-10, 10), translate=(0.1, 0.1), scale=(0.9, 1.1)), transforms.RandomHorizontalFlip(p=0.2), transforms.RandomGrayscale(p=0.2), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])