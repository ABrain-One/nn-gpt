import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomCrop(size=30), transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=(0.5, 3.0)), transforms.RandomHorizontalFlip(p=0.3), transforms.RandomVerticalFlip(p=0.3), transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 2)), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])