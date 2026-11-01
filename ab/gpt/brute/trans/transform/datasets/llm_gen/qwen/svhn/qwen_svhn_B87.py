import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomResizedCrop(size=32, scale=(0.65, 0.9), ratio=(1.0, 2.5)), transforms.ColorJitter(brightness=0.95, contrast=0.98, saturation=0.92, hue=0.07), transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=(-10, 10)), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])