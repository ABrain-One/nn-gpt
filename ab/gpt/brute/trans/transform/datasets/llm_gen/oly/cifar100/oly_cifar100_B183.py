import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomHorizontalFlip(p=0.6), transforms.RandomVerticalFlip(p=0.59), transforms.RandomRotation(degrees=(-15, 15)), transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), transforms.Pad(padding=4, fill=(49, 209, 210), padding_mode='reflect'), transforms.RandomCrop(size=31), transforms.Resize((64, 64), interpolation=transforms.InterpolationMode.BICUBIC), transforms.ToTensor(), transforms.Normalize(*norm)])