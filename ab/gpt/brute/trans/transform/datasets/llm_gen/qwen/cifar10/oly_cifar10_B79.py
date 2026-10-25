import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomHorizontalFlip(p=0.25), transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)), transforms.RandomRotation(15), transforms.CenterCrop(size=32), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])