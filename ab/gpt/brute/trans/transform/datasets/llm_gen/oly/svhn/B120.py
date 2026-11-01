import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomResizedCrop(32, scale=(0.8, 1.2)), transforms.RandomHorizontalFlip(), transforms.ColorJitter(brightness=0.9, contrast=0.9, saturation=1.0, hue=0.15), transforms.RandomEqualize(p=0.8), transforms.RandomResizedCrop(31), transforms.Resize(64), transforms.ToTensor(), transforms.Normalize(*norm)])