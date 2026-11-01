import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomHorizontalFlip(p=0.45), transforms.RandomResizedCrop(size=32, scale=(0.6, 0.9), ratio=(0.75, 2.25)), transforms.ColorJitter(brightness=0.85, contrast=0.85, saturation=1.0, hue=0.01), transforms.RandomPosterize(bits=7, p=0.55), transforms.CenterCrop(size=27), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])