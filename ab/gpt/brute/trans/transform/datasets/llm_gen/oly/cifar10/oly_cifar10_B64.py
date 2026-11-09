import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.Pad(padding=2, fill=(125, 122, 116), padding_mode='constant'), transforms.RandomResizedCrop(32, scale=(0.8, 1.0), ratio=(0.9, 2.1)), transforms.GaussianBlur(kernel_size=3, sigma=(0.41, 1.24)), transforms.RandomHorizontalFlip(p=0.33), transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])