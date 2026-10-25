import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.CenterCrop(size=28), transforms.RandomAffine(degrees=(-15, 15), translate=(0.1, 0.1), scale=(0.9, 1.1), shear=(-5, 5)), transforms.RandomHorizontalFlip(p=0.4), transforms.RandomVerticalFlip(p=0.3), transforms.ColorJitter(brightness=0.8, contrast=0.7, saturation=0.8, hue=0.02), transforms.RandomPosterize(bits=6, p=0.35), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])