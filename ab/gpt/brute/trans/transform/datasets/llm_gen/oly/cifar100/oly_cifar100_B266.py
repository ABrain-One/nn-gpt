import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.Pad(padding=4, fill=(125, 121, 115), padding_mode='constant'), transforms.RandomHorizontalFlip(p=0.6), transforms.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.1), transforms.RandomRotation(degrees=5), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(**norm)])