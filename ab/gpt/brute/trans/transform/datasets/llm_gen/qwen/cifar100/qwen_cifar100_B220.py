import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomResizedCrop(size=32, scale=(0.75, 0.98), ratio=(0.95, 2.33)), transforms.RandomHorizontalFlip(p=0.45), transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), transforms.RandomGrayscale(p=0.15), transforms.RandomAffine(degrees=2, translate=(0.05, 0.15), scale=(0.98, 1.05), shear=(-2.5, 2.5)), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])