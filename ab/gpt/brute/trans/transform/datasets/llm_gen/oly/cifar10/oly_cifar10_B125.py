import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomVerticalFlip(p=0.9), transforms.RandomEqualize(p=0.1), transforms.RandomInvert(p=0.85), transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), transforms.RandomRotation(degrees=(-10, 10), p=0.5), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])