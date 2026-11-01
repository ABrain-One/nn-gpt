import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomRotation(15), transforms.RandomResizedCrop(size=32, scale=(0.8, 0.95), ratio=(0.75, 1.33)), transforms.ColorJitter(brightness=0.7, contrast=0.8, saturation=0.9, hue=0.02), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])