import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomHorizontalFlip(p=0.5), transforms.RandomResizedCrop(size=(32, 32), scale=(0.6, 1.0), ratio=(0.75, 1.33)), transforms.RandomRotation(degrees=5), transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2), transforms.RandomGrayscale(p=0.3), transforms.RandomAutocontrast(p=0.6), transforms.RandomInvert(p=0.15), transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=5), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])