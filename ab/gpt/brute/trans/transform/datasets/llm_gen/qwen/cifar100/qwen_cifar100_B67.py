import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomHorizontalFlip(p=0.7), transforms.ColorJitter(brightness=0.8, contrast=0.7, saturation=0.6, hue=0.02), transforms.RandomAffine(degrees=15, translate=(0.03, 0.07), scale=(0.98, 1.5), shear=(-0.6, 7.5)), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])