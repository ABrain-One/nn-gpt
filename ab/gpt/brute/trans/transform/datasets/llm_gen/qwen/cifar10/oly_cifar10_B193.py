import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.Pad(padding=4, fill=(125, 125, 125), padding_mode='reflect'), transforms.RandomHorizontalFlip(p=0.5), transforms.RandomCrop(size=32, pad_if_needed=True), transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), transforms.RandomAdjustSharpness(sharpness_factor=1.4, p=0.4), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])