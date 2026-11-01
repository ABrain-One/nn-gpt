import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.9, 1.1)), transforms.RandomHorizontalFlip(p=0.5), transforms.RandomRotation(degrees=15), transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), transforms.GaussianBlur(kernel_size=3, sigma=1.5), transforms.ToTensor(), transforms.Normalize(mean=[0.4776, 0.4725, 0.4363], std=[0.2291, 0.2242, 0.2251])])