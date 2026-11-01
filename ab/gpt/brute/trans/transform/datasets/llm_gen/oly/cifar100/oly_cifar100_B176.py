import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.Resize(80), transforms.RandomCrop(64), transforms.RandomHorizontalFlip(p=0.2), transforms.RandomVerticalFlip(p=0.2), transforms.RandomRotation(degrees=(-15, 15)), transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05), transforms.RandomGrayscale(p=0.2), transforms.ToTensor(), transforms.Normalize(**norm)])