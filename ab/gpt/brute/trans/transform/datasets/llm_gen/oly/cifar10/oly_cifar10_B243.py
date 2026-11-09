import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.Resize(448, interpolation=transforms.InterpolationMode.BICUBIC), transforms.RandomCrop(224), transforms.RandomHorizontalFlip(p=0.47), transforms.RandomRotation(6), transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), transforms.ToTensor(), transforms.Normalize(*norm)])