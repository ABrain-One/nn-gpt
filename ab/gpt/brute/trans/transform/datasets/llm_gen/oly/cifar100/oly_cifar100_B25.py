import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomVerticalFlip(p=0.5), transforms.RandomHorizontalFlip(p=0.5), transforms.Pad(padding=4, mode='reflect', fill=(86, 147, 3)), transforms.RandomAdjustSharpness(sharpness_factor=0.5, p=0.5), transforms.Resize((64, 64)), transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), transforms.ToTensor(), transforms.Normalize(*norm)])