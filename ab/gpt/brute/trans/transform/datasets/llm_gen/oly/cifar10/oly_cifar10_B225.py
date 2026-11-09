import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomCrop(28), transforms.Pad(4, fill=(177, 244, 177), padding_mode='symmetric'), transforms.RandomAffine(degrees=10, translate=(0.15, 0.15), scale=(0.8, 1.2), shear=(-5, 5)), transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])