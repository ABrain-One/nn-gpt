import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomHorizontalFlip(p=0.5), transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1), transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)), transforms.Pad(padding=8, fill=(128, 128, 128), padding_mode='reflect'), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])