import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.Pad(padding=4, fill=(21, 86, 118)), transforms.RandomAffine(degrees=15, translate=(0.2, 0.2), scale=(0.85, 1.1), shear=(-5, 5)), transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1), transforms.RandomGrayscale(p=0.25), transforms.RandomInvert(p=0.2), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])