import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomRotation(degrees=(-15, 15), p=0.7), transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.6), transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=0.1, p=0.5), transforms.RandomVerticalFlip(p=0.5), transforms.RandomHorizontalFlip(p=0.5), transforms.GaussianBlur(kernel_size=3, sigma=(0.5, 1.2)), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])