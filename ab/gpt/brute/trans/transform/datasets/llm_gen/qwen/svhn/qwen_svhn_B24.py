import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomRotation(degrees=10), transforms.RandomResizedCrop(size=(64, 64), scale=(0.8, 1.2)), transforms.RandomGrayscale(p=0.2), transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), transforms.RandomHorizontalFlip(p=0.7), transforms.RandomVerticalFlip(p=0.3), transforms.ToTensor(), transforms.Normalize(*norm)])