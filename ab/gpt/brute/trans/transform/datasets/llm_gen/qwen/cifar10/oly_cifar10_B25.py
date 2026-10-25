import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.CenterCrop(size=32), transforms.RandomHorizontalFlip(p=0.45), transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1), transforms.RandomRotation(degrees=10), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])