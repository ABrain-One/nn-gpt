import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomResizedCrop(224, scale=(0.8, 1.2), ratio=(0.75, 1.33)), transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.01), transforms.RandomGrayscale(p=0.2), transforms.RandomHorizontalFlip(p=0.5), transforms.RandomAffine(degrees=5, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=5), transforms.ToTensor(), transforms.Normalize(*norm)])