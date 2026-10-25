import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomAffine(10, scale=(0.9, 1.1), shear=5), transforms.RandomHorizontalFlip(p=0.3), transforms.ColorJitter(brightness=0.7, contrast=1.1, saturation=0.8, hue=0.01), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])