import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomResizedCrop(32, scale=(0.74, 0.85), ratio=(0.84, 1.89)), transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1), transforms.RandomGrayscale(p=0.85), transforms.RandomAdjustSharpness(1.73, p=0.82), transforms.RandomHorizontalFlip(p=0.5), transforms.Resize(64), transforms.ToTensor(), transforms.Normalize(**norm)])