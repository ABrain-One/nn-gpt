import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomAutocontrast(p=0.5), transforms.RandomPerspective(distortion_scale=0.1, p=0.5), transforms.RandomRotation(degrees=(-5, 5), p=0.5), transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), transforms.Resize((256, 256)), transforms.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.9, 1.1)), transforms.ToTensor(), transforms.Normalize(**norm)])