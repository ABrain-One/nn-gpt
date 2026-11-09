import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomHorizontalFlip(p=0.4), transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1), transforms.RandomRotation(degrees=15), transforms.RandomResizedCrop(size=(256, 256), scale=(0.8, 1.0), ratio=(0.75, 1.3333333333333333)), transforms.RandomGrayscale(p=0.2), transforms.ToTensor(), transforms.Normalize(*norm)])