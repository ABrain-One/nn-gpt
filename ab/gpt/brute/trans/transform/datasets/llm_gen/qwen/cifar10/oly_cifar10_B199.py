import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomHorizontalFlip(p=0.5), transforms.RandomRotation(degrees=(-15, 15)), transforms.RandomAffine(translate=(0.1, 0.1), scale=(0.9, 1.1)), transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05), transforms.Pad(padding=4, fill=(128, 128, 128), padding_mode='reflect'), transforms.RandomErasing(scale=(0.02, 0.1), ratio=(0.3, 3.3)), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])