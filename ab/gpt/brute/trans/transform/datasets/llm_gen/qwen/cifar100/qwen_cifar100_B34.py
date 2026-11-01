import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomRotation(degrees=(-15, 15)), transforms.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)), transforms.RandomHorizontalFlip(p=0.4), transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.01), transforms.RandomAffine(degrees=(-5, 5), translate=(0.1, 0.1), scale=(0.9, 1.1)), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])