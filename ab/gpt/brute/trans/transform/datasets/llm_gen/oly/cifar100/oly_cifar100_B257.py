import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomResizedCrop(32, scale=(0.6, 0.9), ratio=(0.8, 2.2)), transforms.ColorJitter(brightness=0.9, contrast=1.1, saturation=1.2, hue=0.1), transforms.RandomRotation(22), transforms.RandomPosterize(4, p=0.2), transforms.RandomCrop(30), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(**norm)])