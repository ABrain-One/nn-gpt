import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomHorizontalFlip(p=0.3), transforms.RandomGrayscale(p=0.1), transforms.RandomResizedCrop(size=32, scale=(0.6, 0.9), ratio=(0.8, 1.2)), transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])