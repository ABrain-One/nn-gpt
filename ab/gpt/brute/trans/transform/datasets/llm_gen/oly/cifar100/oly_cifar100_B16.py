import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomResizedCrop(32, scale=(0.8, 1.0), ratio=(1.3, 1.7)), transforms.RandomPosterize(8, p=0.2), transforms.RandomAdjustSharpness(1.0, p=0.5), transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1), transforms.RandomRotation(6), transforms.Resize(64), transforms.ToTensor(), transforms.Normalize(*norm)])