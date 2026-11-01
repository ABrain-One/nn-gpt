import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomHorizontalFlip(p=0.3), transforms.RandomResizedCrop(size=(32, 32), scale=(0.8, 1.0)), transforms.RandomRotation(degrees=5), transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), transforms.RandomVerticalFlip(p=0.1), transforms.ToTensor(), transforms.Normalize(*norm)])