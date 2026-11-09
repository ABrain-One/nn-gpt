import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomResizedCrop(32, scale=(0.5, 1.0), ratio=(0.8, 2.0)), transforms.RandomHorizontalFlip(p=0.5), transforms.GaussianBlur(kernel_size=3, sigma=(0.8, 1.1)), transforms.RandomAutocontrast(p=0.6), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])