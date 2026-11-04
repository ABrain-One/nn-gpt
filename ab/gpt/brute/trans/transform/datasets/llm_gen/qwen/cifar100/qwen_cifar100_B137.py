import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomResizedCrop(256), transforms.RandomHorizontalFlip(), transforms.ColorJitter(brightness=0.85, contrast=0.88, saturation=1.05, hue=0.08), transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)), transforms.RandomRotation(degrees=15), transforms.RandomAffine(degrees=15, translate=(0.05, 0.15), scale=(0.95, 1.05), shear=(0.1, 5)), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])