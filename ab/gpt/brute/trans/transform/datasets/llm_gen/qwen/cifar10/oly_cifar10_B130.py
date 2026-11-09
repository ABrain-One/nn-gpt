import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomResizedCrop(size=32, scale=(0.65, 0.95), ratio=(0.8, 1.2)), transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), transforms.RandomHorizontalFlip(p=0.2), transforms.RandomVerticalFlip(p=0.1), transforms.RandomRotation(degrees=(-10, 10)), transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5)), transforms.RandomErasing(scale=(0.02, 0.1), ratio=(0.3, 3.3)), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])