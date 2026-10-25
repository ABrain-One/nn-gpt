import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomHorizontalFlip(p=0.15), transforms.RandomRotation(degrees=(-15, 15)), transforms.Pad(padding=8, fill=(120, 112, 132), padding_mode='reflect'), transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.5, 0.5)), transforms.RandomResizedCrop(size=32, scale=(0.8, 1.2), ratio=(0.9, 1.1)), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])