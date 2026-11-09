import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomResizedCrop(size=32, scale=(0.64, 0.95), ratio=(0.76, 2.08)), transforms.RandomHorizontalFlip(p=0.49), transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), transforms.GaussianBlur(kernel_size=5, sigma=(0.35, 1.77)), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])