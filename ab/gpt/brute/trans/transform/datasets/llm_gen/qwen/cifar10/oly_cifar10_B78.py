import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomHorizontalFlip(p=0.35), transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1), transforms.RandomAffine(degrees=(-15, 15), translate=(0.1, 0.1), scale=(0.8, 1.2)), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])