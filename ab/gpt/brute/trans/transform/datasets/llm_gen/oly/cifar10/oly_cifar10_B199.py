import torch
from torchvision import transforms

def transform():
    return transforms.Compose([transforms.Pad(padding=2, fill=(125, 123, 97)), transforms.RandomCrop(32), transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), transforms.RandomRotation(15), transforms.Resize(64), transforms.ToTensor(), transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.201])])