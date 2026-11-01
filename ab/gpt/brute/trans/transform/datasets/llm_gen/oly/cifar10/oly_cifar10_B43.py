import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomVerticalFlip(p=0.2), transforms.RandomHorizontalFlip(p=0.5), transforms.GaussianBlur(kernel_size=5, sigma=1.0), transforms.RandomResizedCrop(24, scale=(0.8, 1.0), ratio=(0.9, 1.1)), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(**norm)])