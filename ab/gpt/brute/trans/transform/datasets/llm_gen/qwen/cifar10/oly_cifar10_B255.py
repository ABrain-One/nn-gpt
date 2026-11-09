import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomResizedCrop(size=32, scale=(0.64, 0.92), ratio=(0.82, 2.0)), transforms.RandomHorizontalFlip(p=0.47), transforms.RandomAffine(degrees=13, translate=(0.09, 0.04), scale=(1.12, 1.57), shear=(2.83, 6.36)), transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), transforms.RandomGrayscale(p=0.2), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])