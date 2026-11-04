import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.Pad(padding=4, fill=(168, 190, 153), padding_mode='reflect'), transforms.RandomRotation(degrees=15), transforms.ColorJitter(brightness=0.9, contrast=0.9, saturation=0.9, hue=0.1), transforms.RandomInvert(p=0.2), transforms.RandomResizedCrop(size=32, scale=(0.8, 1.0)), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])