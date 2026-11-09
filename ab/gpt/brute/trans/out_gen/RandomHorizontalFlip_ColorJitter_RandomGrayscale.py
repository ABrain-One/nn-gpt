import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.28),
    transforms.ColorJitter(brightness=1.11, contrast=0.87, saturation=1.14, hue=0.04),
    transforms.RandomGrayscale(p=0.82),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
