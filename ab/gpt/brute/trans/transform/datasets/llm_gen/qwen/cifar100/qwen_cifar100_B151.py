import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomResizedCrop(size=32, scale=(0.65, 0.9), ratio=(0.8, 2.0)), transforms.RandomHorizontalFlip(p=0.48), transforms.RandomAffine(degrees=12, translate=(0.1, 0.03), scale=(1.1, 1.6), shear=(2.9, 6.4)), transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])