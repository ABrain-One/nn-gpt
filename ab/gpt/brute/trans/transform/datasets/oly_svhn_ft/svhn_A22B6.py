import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.ColorJitter(brightness=0.84, contrast=1.18, saturation=0.9, hue=0.05), transforms.RandomResizedCrop(size=32, scale=(0.76, 1.0), ratio=(1.2, 1.88)), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])