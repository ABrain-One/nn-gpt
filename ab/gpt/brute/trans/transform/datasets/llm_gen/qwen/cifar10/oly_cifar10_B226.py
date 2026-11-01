import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomHorizontalFlip(p=0.75), transforms.ColorJitter(brightness=0.75, contrast=0.9, saturation=1.05, hue=0.1), transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=(-10, 10), fillcolor=(128, 128, 128)), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])