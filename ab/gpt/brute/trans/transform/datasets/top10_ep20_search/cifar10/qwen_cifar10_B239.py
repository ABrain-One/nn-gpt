import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.75, 1.33)), transforms.RandomApply([transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)], p=0.8), transforms.RandomGrayscale(p=0.2), transforms.RandomHorizontalFlip(p=0.5), transforms.ToTensor(), transforms.Normalize(*norm)])