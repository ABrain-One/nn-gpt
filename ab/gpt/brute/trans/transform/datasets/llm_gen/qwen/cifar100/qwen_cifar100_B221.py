import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomVerticalFlip(p=0.15), transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.1, hue=0.05), transforms.RandomRotation(degrees=(-15, 15)), transforms.RandomPerspective(distortion_scale=0.1, p=0.1), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])