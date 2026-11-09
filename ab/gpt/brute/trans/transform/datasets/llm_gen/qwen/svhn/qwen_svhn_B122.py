import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomHorizontalFlip(p=0.3), transforms.RandomVerticalFlip(p=0.15), transforms.RandomRotation(degrees=15), transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])