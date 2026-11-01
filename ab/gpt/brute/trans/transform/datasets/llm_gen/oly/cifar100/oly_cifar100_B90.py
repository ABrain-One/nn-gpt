import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomResizedCrop(64, scale=(0.75, 1.3), ratio=(0.8, 2.4)), transforms.RandomRotation(degrees=(-5, 5)), transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1), transforms.RandomPerspective(distortion_scale=0.25, p=0.5), transforms.GaussianBlur(5, sigma=(0.5, 1.5)), transforms.ToTensor(), transforms.Normalize(**norm)])