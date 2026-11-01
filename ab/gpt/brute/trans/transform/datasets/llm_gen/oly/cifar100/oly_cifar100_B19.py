import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomHorizontalFlip(p=0.5), transforms.RandomRotation(degrees=(-20, 20), p=0.6), transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.15), transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=0.5)], p=0.5), transforms.Resize(72), transforms.RandomCrop(64), transforms.ToTensor(), transforms.Normalize(*norm)])