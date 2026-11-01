import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomHorizontalFlip(p=0.5), transforms.RandomVerticalFlip(p=0.2), transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05), transforms.RandomResizedCrop(size=24, scale=(0.8, 1.2), ratio=(0.75, 1.33)), transforms.Pad(padding=4, fill=(128, 128, 128), padding_mode='reflect'), transforms.RandomRotation(degrees=(-10, 10)), transforms.GaussianBlur(kernel_size=3, sigma=(0.2, 0.5)), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])