import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomHorizontalFlip(p=0.5), transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), transforms.RandomGrayscale(p=0.2), transforms.RandomRotation(degrees=(-10, 10)), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])