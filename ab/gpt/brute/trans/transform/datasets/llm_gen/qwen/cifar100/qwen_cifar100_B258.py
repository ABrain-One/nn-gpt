import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomResizedCrop(size=32, scale=(0.72, 0.98), ratio=(0.78, 2.62)), transforms.RandomHorizontalFlip(p=0.45), transforms.RandomAdjustSharpness(sharpness_factor=1.2, p=0.4), transforms.ColorJitter(brightness=0.15, contrast=0.1, saturation=0.1, hue=0.05), transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])