import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.Pad(padding=4, fill=(128, 128, 128), padding_mode='reflect'), transforms.RandomHorizontalFlip(p=0.5), transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1), transforms.RandomResizedCrop(size=224, scale=(0.8, 1.2)), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])