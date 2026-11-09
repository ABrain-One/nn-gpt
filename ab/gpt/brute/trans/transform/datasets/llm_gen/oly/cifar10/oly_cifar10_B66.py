import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomHorizontalFlip(p=0.8), transforms.RandomResizedCrop(64, scale=(0.6, 1.0), ratio=(0.8, 1.2)), transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), transforms.GaussianBlur(kernel_size=3, sigma=(0.5, 1.5)), transforms.ToTensor(), transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.201])])