import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomHorizontalFlip(p=0.15), transforms.RandomAffine(degrees=15, translate=(0.08, 0.12), scale=(0.95, 1.43), shear=(1.25, 6.25)), transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.1), transforms.RandomRotation(15), transforms.RandomResizedCrop(size=29, scale=(0.8, 1.0), ratio=(0.75, 1.33)), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])