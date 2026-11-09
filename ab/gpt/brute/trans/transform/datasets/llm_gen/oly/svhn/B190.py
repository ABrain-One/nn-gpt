import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.Pad(padding=3, padding_mode='reflect', fill=(147, 89, 233)), transforms.RandomInvert(p=0.41), transforms.RandomVerticalFlip(p=0.74), transforms.ColorJitter(brightness=1.07, contrast=1.15, saturation=0.83, hue=0.04), transforms.Resize(64), transforms.ToTensor(), transforms.Normalize(mean=[0.4377, 0.4505, 0.402], std=[0.2654, 0.2431, 0.2575])])