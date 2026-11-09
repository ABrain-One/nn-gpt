import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomHorizontalFlip(p=0.7), transforms.RandomRotation(degrees=15), transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), transforms.RandomErasing(scale=(0.02, 0.1), ratio=(0.3, 3.3)), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])