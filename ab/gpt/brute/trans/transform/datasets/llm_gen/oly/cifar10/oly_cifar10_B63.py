import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomResizedCrop(64, scale_range=(0.8, 1.2), aspect_ratio_range=(0.9, 1.1)), transforms.RandomApply([transforms.ColorJitter(0.2, 0.2, 0.2, 0.1)], p=0.8), transforms.RandomGrayscale(p=0.2), transforms.RandomRotation(degrees=(-15, 15)), transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, std=(0.1, 2.0))], p=0.5), transforms.RandomHorizontalFlip(p=0.85), transforms.ToTensor(), transforms.Normalize(*norm)])