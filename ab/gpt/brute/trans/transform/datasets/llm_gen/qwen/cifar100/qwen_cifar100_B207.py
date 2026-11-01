import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.Pad(padding=2, fill=(127, 127, 127), padding_mode='symmetric'), transforms.RandomHorizontalFlip(p=0.5), transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=5), transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1), transforms.RandomResizedCrop(size=64, scale=(0.8, 1.0), ratio=(0.9, 1.1)), transforms.ToTensor(), transforms.Normalize(*norm)])