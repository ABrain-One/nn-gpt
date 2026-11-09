import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.9, 1.1)), transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), transforms.RandomRotation(degrees=(-5, 5)), transforms.ToTensor(), transforms.Normalize(mean=[0.4325, 0.451, 0.3805], std=[0.2884, 0.2742, 0.277])])