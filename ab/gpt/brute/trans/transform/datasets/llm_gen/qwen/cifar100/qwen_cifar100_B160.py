import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomHorizontalFlip(p=0.75), transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1), transforms.RandomRotation(degrees=10), transforms.RandomResizedCrop(size=32, scale=(0.8, 1.2)), transforms.Pad(padding=4, fill=(128, 128, 128), padding_mode='reflect'), transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)), transforms.ToTensor(), transforms.Normalize(*norm)])