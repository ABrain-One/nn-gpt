import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomResizedCrop(64, scale=(0.5, 1.0), ratio=(0.5, 2.0)), transforms.RandomHorizontalFlip(), transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(1.5, 1.5))], p=0.5), transforms.ToTensor(), transforms.Normalize(**norm)])