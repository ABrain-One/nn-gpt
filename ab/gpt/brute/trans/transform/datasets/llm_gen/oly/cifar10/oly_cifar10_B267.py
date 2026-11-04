import torch
import torchvision.transforms as transforms
cifar10_mean = [0.4914, 0.4822, 0.4465]
cifar10_std = [0.2023, 0.1994, 0.201]

def transform():
    return transforms.Compose([transforms.RandomResizedCrop(32, scale=(0.8, 1.0), ratio=(0.9, 1.1)), transforms.RandomHorizontalFlip(p=0.5), transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1), transforms.RandomRotation(degrees=(-15, 15)), transforms.GaussianBlur(kernel_size=3, sigma=1.5), transforms.Resize(224), transforms.ToTensor(), transforms.Normalize(mean=cifar10_mean, std=cifar10_std)])