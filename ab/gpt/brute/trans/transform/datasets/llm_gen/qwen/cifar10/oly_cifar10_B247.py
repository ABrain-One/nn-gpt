import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomResizedCrop(size=32, scale=(0.8, 1.0)), transforms.RandomHorizontalFlip(p=0.5), transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1), transforms.RandomGrayscale(p=0.2), transforms.ToTensor(), transforms.Normalize(*norm)])