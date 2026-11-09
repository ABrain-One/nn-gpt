import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomCrop(30), transforms.Pad(2, fill=(191, 85, 157)), transforms.ColorJitter(brightness=0.85, contrast=0.92, saturation=1.11, hue=0.08), transforms.Resize(64), transforms.ToTensor(), transforms.Normalize(*norm)])