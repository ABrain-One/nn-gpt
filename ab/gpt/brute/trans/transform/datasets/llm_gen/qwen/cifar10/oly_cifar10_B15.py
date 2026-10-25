import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomRotation(degrees=(-10, 10), p=0.4), transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.3), transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=(0, 10), resample=False, fillcolor=None, padding=None, pad_if_needed=False, fillvalue=0, antialias=None, interpolation=2), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])