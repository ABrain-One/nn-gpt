import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomHorizontalFlip(p=0.4), transforms.RandomAffine(15, translate=(0.1, 0.1), scale=(0.9, 1.1)), transforms.ColorJitter(brightness=0.9, contrast=0.9, saturation=0.9, hue=0.05), transforms.RandomGrayscale(p=0.2), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])