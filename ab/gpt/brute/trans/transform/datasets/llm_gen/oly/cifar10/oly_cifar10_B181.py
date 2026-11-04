import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomHorizontalFlip(p=0.5), transforms.Resize(448), transforms.RandomResizedCrop(224, scale=(0.5, 1.0), ratio=(0.5, 2.0)), transforms.RandomRotation(degrees=15), transforms.GaussianBlur(kernel_size=5, sigma=(0.33, 1.04)), transforms.ColorJitter(brightness=0.2, contrast=0.2), transforms.ToTensor(), transforms.Normalize(*norm)])