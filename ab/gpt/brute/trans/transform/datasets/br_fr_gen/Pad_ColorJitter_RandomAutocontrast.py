import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=1, fill=(62, 62, 204), padding_mode='reflect'),
    transforms.ColorJitter(brightness=0.92, contrast=1.15, saturation=0.99, hue=0.01),
    transforms.RandomAutocontrast(p=0.31),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
