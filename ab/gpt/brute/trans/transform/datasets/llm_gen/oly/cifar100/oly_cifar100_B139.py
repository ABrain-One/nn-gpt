import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomRotation(degrees=(-15, 15)), transforms.Resize(256), transforms.RandomResizedCrop(224, scale=(0.5, 1.0), ratio=(0.9, 1.1)), transforms.RandomHorizontalFlip(p=0.5), transforms.GaussianBlur(kernel_size=3, sigma=(0.5, 1.0)), transforms.ToTensor(), transforms.Normalize(*norm)])