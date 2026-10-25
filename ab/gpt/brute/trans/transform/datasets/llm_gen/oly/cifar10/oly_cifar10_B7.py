import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.Pad(padding=4, fill=(240, 95, 30), padding_mode='constant'), transforms.RandomResizedCrop(size=32, scale=(0.75, 0.95), ratio=(0.9, 2.1)), transforms.RandomApply([transforms.ColorJitter(0.2, 0.2, 0.2, 0.1)], p=0.8), transforms.RandomRotation(degrees=(-10, 10)), transforms.RandomHorizontalFlip(p=0.5), transforms.GaussianBlur(kernel_size=3, sigma=1.5, p=0.5), transforms.ToTensor(), transforms.Normalize(**norm)])