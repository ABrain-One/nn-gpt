import torch
from torchvision import transforms

def transform(norm):
    return transforms.Compose([transforms.RandomResizedCrop(224, scale=(0.5, 1.0), ratio=(0.75, 1.333)), transforms.RandomHorizontalFlip(p=0.5), transforms.RandomRotation(degrees=(-15, 15)), transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), transforms.RandomGrayscale(p=0.2), transforms.GaussianBlur(kernel_size=3, sigma=1.0, p=0.5), transforms.ToTensor(), transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.201])])