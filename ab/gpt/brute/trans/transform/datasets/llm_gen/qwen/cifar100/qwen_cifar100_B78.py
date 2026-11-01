import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomResizedCrop(size=32, scale=(0.65, 0.85), ratio=(1.1, 1.5)), transforms.Pad(padding=4, fill=(153, 255, 153), padding_mode='reflect'), transforms.RandomHorizontalFlip(p=0.6), transforms.RandomRotation(degrees=(-15, 15)), transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05), transforms.GaussianBlur(kernel_size=3), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])