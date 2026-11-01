import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.Pad(padding=2, fill=(128, 128, 128), padding_mode='symmetric'), transforms.RandomRotation(degrees=5), transforms.RandomAutocontrast(p=0.75), transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05), transforms.RandomCrop(size=28), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])