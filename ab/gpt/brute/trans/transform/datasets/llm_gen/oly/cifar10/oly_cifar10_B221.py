import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.Pad(padding=2, padding_mode='reflect'), transforms.RandomCrop(32), transforms.RandomHorizontalFlip(p=0.5), transforms.RandomVerticalFlip(p=0.1), transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=1.0)], p=0.5), transforms.Resize(224), transforms.ToTensor(), transforms.Normalize(*norm)])