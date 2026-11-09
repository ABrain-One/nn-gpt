import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomResizedCrop(size=32, scale=(0.65, 0.92), ratio=(0.7, 1.2)), transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), transforms.RandomHorizontalFlip(), transforms.RandomAffine(degrees=5, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=(-2, 2)), transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.3)), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])