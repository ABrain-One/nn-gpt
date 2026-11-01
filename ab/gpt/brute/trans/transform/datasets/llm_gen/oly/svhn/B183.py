import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.ColorJitter(brightness=0.9, contrast=0.85, saturation=1.0, hue=0.05), transforms.Pad(padding=4, padding_mode='edge'), transforms.RandomRotation(degrees=15), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])