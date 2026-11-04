import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.Pad(padding=4, fill=(0.507, 0.486, 0.441), padding_mode='constant'), transforms.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.1), transforms.RandomResizedCrop(size=64, scale=(0.6, 1.0), ratio=(0.8, 1.2)), transforms.GaussianBlur(kernel_size=3, sigma=(0.5, 1.5)), transforms.ToTensor(), transforms.Normalize(**norm)])