import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomResizedCrop(size=32, scale=(0.8, 1.0), ratio=(0.75, 1.25)), transforms.RandomHorizontalFlip(p=0.3), transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05), transforms.Pad(padding=4, fill=(128, 128, 128), padding_mode='constant'), transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])