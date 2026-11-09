import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.Resize(80, interpolation=transforms.InterpolationMode.BICUBIC), transforms.RandomCrop(64), transforms.RandomRotation(degrees=18), transforms.RandomAdjustSharpness(sharpness_factor=0.7, p=0.7), transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5), transforms.ToTensor(), transforms.Normalize(**norm)])