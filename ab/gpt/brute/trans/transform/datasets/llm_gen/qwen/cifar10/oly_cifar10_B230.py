import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomHorizontalFlip(p=0.85), transforms.RandomVerticalFlip(p=0.1), transforms.RandomRotation(degrees=(-5, 5)), transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05), transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 1.5)), transforms.RandomErasing(scale=(0.02, 0.1), ratio=(0.3, 3)), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])