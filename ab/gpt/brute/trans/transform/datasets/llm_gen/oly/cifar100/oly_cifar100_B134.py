import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05), transforms.RandomHorizontalFlip(), transforms.Pad(padding=5, fill=(125, 122, 114), padding_mode='constant'), transforms.RandomCrop(size=32), transforms.Resize(224), transforms.ToTensor(), transforms.Normalize(**norm)])