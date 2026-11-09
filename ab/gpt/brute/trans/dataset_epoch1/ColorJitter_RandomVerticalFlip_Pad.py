import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.ColorJitter(brightness=1.07, contrast=1.15, saturation=0.83, hue=0.04),
    transforms.RandomVerticalFlip(p=0.81),
    transforms.Pad(padding=3, fill=(147, 89, 233), padding_mode='reflect'),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
