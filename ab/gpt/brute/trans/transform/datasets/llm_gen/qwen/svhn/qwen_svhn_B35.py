import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.ColorJitter(brightness=1.02, contrast=1.03, saturation=0.85, hue=0.04), transforms.RandomResizedCrop(size=(256, 256), scale=(0.8, 1.2)), transforms.RandomHorizontalFlip(p=0.5), transforms.RandomRotation(degrees=(-10, 10)), transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 0.5)), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])