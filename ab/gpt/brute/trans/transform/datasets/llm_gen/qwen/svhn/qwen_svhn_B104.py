import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomHorizontalFlip(p=0.5), transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])