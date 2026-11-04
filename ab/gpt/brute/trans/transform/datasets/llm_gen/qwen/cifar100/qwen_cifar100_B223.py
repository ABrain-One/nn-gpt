import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomHorizontalFlip(p=0.5), transforms.RandomAffine(degrees=10, translate=(0.02, 0.1), scale=(0.9, 1.1), shear=(-2, 2)), transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1), transforms.RandomResizedCrop(size=224), transforms.ToTensor(), transforms.Normalize(*norm)])