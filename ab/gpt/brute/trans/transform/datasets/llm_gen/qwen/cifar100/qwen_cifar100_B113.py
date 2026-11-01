import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(p=0.5), transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=5), transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.01), transforms.RandomGrayscale(p=0.2), transforms.ToTensor(), transforms.Normalize(*norm)])