import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(), transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)), transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05), transforms.RandomGrayscale(p=0.2), transforms.ToTensor(), transforms.Normalize(*norm)])