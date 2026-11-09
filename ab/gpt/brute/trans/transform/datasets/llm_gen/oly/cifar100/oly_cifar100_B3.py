import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomResizedCrop(32, scale=(0.5, 1.0), ratio=(0.8, 1.2)), transforms.ColorJitter(0.2, 0.2, 0.2, 0.1), transforms.RandomHorizontalFlip(p=0.7), transforms.GaussianBlur(kernel_size=3, sigma=(0.5, 1.5)), transforms.ToTensor(), transforms.Normalize(*norm)])