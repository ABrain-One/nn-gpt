import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomRotation(2), transforms.RandomCrop(26), transforms.GaussianBlur(kernel_size=3, sigma=1.0), transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), transforms.Resize(64), transforms.ToTensor(), transforms.Normalize(*norm)])