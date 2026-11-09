import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomHorizontalFlip(p=0.85), transforms.Pad(padding=2, fill=(152, 144, 151), padding_mode='constant'), transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=5), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])