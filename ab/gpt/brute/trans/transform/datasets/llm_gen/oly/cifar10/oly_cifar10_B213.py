import torch
import torchvision.transforms as transforms

def transform():
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.201]
    return transforms.Compose([transforms.RandomResizedCrop(224, scale=(0.75, 1.0), ratio=(0.9, 1.1)), transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), transforms.RandomHorizontalFlip(p=0.5), transforms.GaussianBlur(kernel_size=3, sigma=1.0), transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])