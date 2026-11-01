import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomResizedCrop(64, scale=(0.5, 1.0), ratio=(0.8, 3.0)), transforms.RandomHorizontalFlip(p=0.24), transforms.GaussianBlur(kernel_size=5, sigma=(0.77, 1.91)), transforms.ToTensor(), transforms.Normalize(**norm)])